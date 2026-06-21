"""
registry.py — MLflow experiment tracking + model registry (plan §8).

A served artifact is the corrector PLUS its probability calibrator bundled as a
`ServedModel`, so serving applies the calibrated P(ISSR), not the raw head.

`log_run` records everything needed to reproduce a model — params, metrics, the
comparison table, diagnostic plots, the git SHA, the dataset hash, and library
versions — logs the bundle as an artifact, registers it under
`cfg.registered_model_name`, and moves the new version to Staging.

`load_model` pulls the latest registered version back for serving. The registry
defaults to a local file backend (`mlruns/`) and is configurable to a remote
tracking URI, so the same code runs on a laptop or against a team server.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .calibrate import ProbabilityCalibrator
    from .config import MLConfig
    from .model import RHiCorrector


@dataclass
class ServedModel:
    """Serving bundle: the corrector + the probability calibrator.

    `.predict(df)` mirrors RHiCorrector.predict but returns the CALIBRATED
    P(ISSR), so the whole serving path (predict.field_from_metcube) is unchanged
    whether it gets a raw corrector or a calibrated bundle.
    """

    corrector: RHiCorrector
    calibrator: ProbabilityCalibrator | None = None
    conformal_half_width: float = float("nan")

    def predict(self, df) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rhi_hat, rhi_std, p_raw = self.corrector.predict(df)
        p = self.calibrator.transform(p_raw) if self.calibrator is not None else p_raw
        return rhi_hat, rhi_std, np.asarray(p, dtype=float)

    def predict_proba_issr(self, df) -> np.ndarray:
        return self.predict(df)[2]

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> ServedModel:
        import joblib

        obj = joblib.load(path)
        if not isinstance(obj, ServedModel):
            raise TypeError(f"{path} did not contain a ServedModel")
        return obj


# ===========================================================================
# MLflow helpers
# ===========================================================================

_ARTIFACT_DIR = "model"
_ARTIFACT_NAME = "served_model.joblib"


def set_tracking(cfg: MLConfig) -> None:
    import os

    import mlflow

    # The local file backend (the spec's default) is in maintenance mode on
    # newer MLflow and raises unless this opt-out is set. Honour it for file
    # URIs; a remote/db tracking URI is unaffected.
    if cfg.mlflow_tracking_uri.startswith(("file:", "./", "../", "mlruns")):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.registered_model_name)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _lib_versions() -> dict[str, str]:
    import importlib

    out = {}
    for p in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "mlflow"):
        try:
            out[p] = importlib.import_module(p).__version__  # type: ignore[attr-defined]
        except Exception:
            out[p] = "absent"
    return out


def log_run(
    cfg: MLConfig,
    served_model: ServedModel,
    *,
    params: dict[str, Any],
    metrics: dict[str, float],
    comparison: pd.DataFrame | None = None,
    figures: dict[str, Any] | None = None,
    dataset_hash: str = "unknown",
    register: bool = True,
) -> dict[str, Any]:
    """Log a training run to MLflow and (optionally) register + stage it.

    Returns {run_id, model_version (or None)}.
    """
    import json
    import os
    import tempfile

    import mlflow

    set_tracking(cfg)
    result: dict[str, Any] = {"run_id": None, "model_version": None}

    with mlflow.start_run() as run:
        result["run_id"] = run.info.run_id
        mlflow.log_params({k: _short(v) for k, v in params.items()})
        mlflow.set_tags(
            {"git_sha": _git_sha(), "dataset_hash": dataset_hash, **_lib_versions()}
        )
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                            if v is not None and np.isfinite(v)})

        # Log the bundle as a proper pyfunc MODEL (so it is registrable on
        # modern MLflow, which requires a logged model — a raw artifact path is
        # no longer enough).
        info = mlflow.pyfunc.log_model(
            artifact_path=_ARTIFACT_DIR,
            python_model=_served_pyfunc(served_model),
        )

        with tempfile.TemporaryDirectory() as td:
            if comparison is not None:
                cpath = os.path.join(td, "comparison_table.csv")
                comparison.to_csv(cpath)
                mlflow.log_artifact(cpath)
                try:  # markdown is nicer but needs `tabulate`; CSV is the source of truth
                    mlflow.log_text(comparison.to_markdown(), "comparison_table.md")
                except ImportError:
                    pass

            meta = {"dataset_hash": dataset_hash, "git_sha": _git_sha(),
                    "config": _config_dict(cfg)}
            jpath = os.path.join(td, "run_metadata.json")
            with open(jpath, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, default=str)
            mlflow.log_artifact(jpath)

        for name, fig in (figures or {}).items():
            mlflow.log_figure(fig, f"plots/{name}.png")

        if register:
            mv = mlflow.register_model(info.model_uri, cfg.registered_model_name)
            result["model_version"] = mv.version
            _transition(cfg, mv.version, "Staging")

    return result


def _served_pyfunc(served: ServedModel):
    """Wrap a ServedModel as an mlflow PythonModel (predict -> DataFrame).

    Defined inside a factory so `import mlflow` stays lazy; cloudpickle captures
    the served bundle (the fitted estimators) with the model.
    """
    import mlflow

    class _ServedPyfunc(mlflow.pyfunc.PythonModel):  # type: ignore[misc,name-defined]
        def predict(self, context, model_input, params=None):  # noqa: ARG002
            rhi_hat, rhi_std, p = served.predict(model_input)
            return pd.DataFrame({"rhi_hat": rhi_hat, "rhi_std": rhi_std, "p_issr": p})

    return _ServedPyfunc()


def _transition(cfg: MLConfig, version: str, stage: str) -> None:
    from mlflow.tracking import MlflowClient

    MlflowClient().transition_model_version_stage(
        name=cfg.registered_model_name, version=version, stage=stage,
        archive_existing_versions=False,
    )


def promote(cfg: MLConfig, version: str, stage: str = "Production") -> None:
    """Promote a registered version to a stage (Staging/Production/Archived)."""
    set_tracking(cfg)
    _transition(cfg, version, stage)


@dataclass
class _PyfuncAdapter:
    """Adapts a loaded pyfunc model back to the `.predict -> (rhi_hat, rhi_std,
    p_issr)` tuple interface the serving code (predict.field_from_metcube) and
    the monitor expect."""

    pyfunc: Any

    def predict(self, df) -> tuple:
        out = self.pyfunc.predict(df)
        return (out["rhi_hat"].to_numpy(), out["rhi_std"].to_numpy(),
                out["p_issr"].to_numpy())

    def predict_proba_issr(self, df):
        return self.predict(df)[2]


def load_model(cfg: MLConfig, stage: str = "Staging") -> _PyfuncAdapter:
    """Load the latest registered model for a stage (default Staging) and adapt
    it to the native serving interface."""
    import mlflow
    from mlflow.tracking import MlflowClient

    set_tracking(cfg)
    client = MlflowClient()
    versions = client.get_latest_versions(cfg.registered_model_name, stages=[stage])
    if not versions:
        raise RuntimeError(
            f"no registered model '{cfg.registered_model_name}' in stage {stage!r}. "
            f"Train one first: python -m contrail_ml train"
        )
    mv = versions[0]
    loaded = mlflow.pyfunc.load_model(f"models:/{cfg.registered_model_name}/{mv.version}")
    return _PyfuncAdapter(loaded)


def _short(v: Any) -> Any:
    s = str(v)
    return s[:250]


def _config_dict(cfg: MLConfig) -> dict[str, Any]:
    return {f.name: getattr(cfg, f.name)
            for f in cfg.__dataclass_fields__.values()  # type: ignore[attr-defined]
            if f.name != "extra"}
