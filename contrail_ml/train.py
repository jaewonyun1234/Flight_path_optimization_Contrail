"""
train.py — Cross-validate, fit, calibrate, evaluate, and register (plan §6).

Pipeline
========
1. Temporal split: train on older years, hold out recent years untouched, so
   the headline metrics are measured on data the model never saw.
2. Grouped CV: GroupKFold over (year, ~5x5deg spatial tile) so neither time nor
   place leaks across folds — the optimistic-CV trap for spatial data.
3. Fit the corrector on a fit-subset; calibrate P(ISSR) and fit the conformal
   RHi interval on a disjoint calibration-subset (no calibration leakage).
4. Evaluate the calibrated model AND every baseline on the temporal holdout;
   build the comparison table.
5. Log params, metrics, the table, and diagnostic plots to MLflow; register the
   model and move it to Staging.

`run_training(..., log_mlflow=False)` runs everything except the MLflow side, so
the training logic is unit-testable with no tracking server.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .baselines import default_baselines
from .calibrate import ConformalRHi, ProbabilityCalibrator
from .data.schema import LABEL_DELTA, LABEL_ISSR, LABEL_RHI
from .evaluate import comparison_table, plot_reliability, plot_rhi_scatter, score_predictions
from .model import RHiCorrector
from .registry import ServedModel

if TYPE_CHECKING:
    from .config import MLConfig


@dataclass
class TrainResult:
    served: ServedModel
    conformal: ConformalRHi
    comparison: pd.DataFrame
    cv_metrics: dict[str, float]
    holdout_metrics: dict[str, float]
    dataset_hash: str
    mlflow: dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# splits & grouping
# ===========================================================================


def _year(df: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(df["time"]).dt.year.to_numpy()


def temporal_split(df: pd.DataFrame, cfg: MLConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Older years -> train, recent years -> held-out test. Falls back to a
    random 75/25 split if the year ranges don't select anything."""
    yr = _year(df)
    tr = (yr >= cfg.train_years[0]) & (yr <= cfg.train_years[1])
    te = (yr >= cfg.test_years[0]) & (yr <= cfg.test_years[1])
    if tr.sum() < 20 or te.sum() < 20:
        rng = np.random.default_rng(cfg.random_state)
        mask = rng.random(len(df)) < 0.75
        return df.loc[mask].reset_index(drop=True), df.loc[~mask].reset_index(drop=True)
    return df.loc[tr].reset_index(drop=True), df.loc[te].reset_index(drop=True)


def cv_groups(df: pd.DataFrame) -> np.ndarray:
    """Group label = year + ~5x5deg spatial tile (no time/space leakage)."""
    yr = _year(df)
    tile_lat = np.floor(df["lat"].to_numpy() / 5.0).astype(int)
    tile_lon = np.floor(df["lon"].to_numpy() / 5.0).astype(int)
    return np.array([f"{a}_{b}_{c}" for a, b, c in zip(yr, tile_lat, tile_lon, strict=False)])


def dataset_hash(df: pd.DataFrame) -> str:
    """Content hash of the training table for provenance."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


# ===========================================================================
# CV & fit
# ===========================================================================


def cross_validate(cfg: MLConfig, train_df: pd.DataFrame, n_splits: int = 4) -> dict[str, float]:
    """Grouped CV; returns mean RHi MAE and ISSR ETS across folds. Uses a
    single-member corrector (no ensemble) for speed — CV measures signal, not
    predictive spread."""
    from sklearn.model_selection import GroupKFold

    groups = cv_groups(train_df)
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)

    maes, etss = [], []
    for tr_idx, va_idx in gkf.split(train_df, groups=groups):
        tr, va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        m = _new_corrector(cfg, n_ensemble=1)
        m.fit(tr, tr[LABEL_DELTA].to_numpy())
        rhi_hat, _s, p = m.predict(va)
        s = score_predictions(va[LABEL_RHI].to_numpy(), va[LABEL_ISSR].to_numpy(),
                              rhi_hat, p)
        maes.append(s["rhi_mae"])
        etss.append(s["ets"])
    return {"rhi_mae": float(np.mean(maes)), "ets": float(np.mean(etss)),
            "n_splits": float(n_splits)}


def fit_and_calibrate(
    cfg: MLConfig, train_df: pd.DataFrame, calib_frac: float = 0.25
) -> tuple[ServedModel, ConformalRHi]:
    """Fit the corrector on a fit-subset, then calibrate P(ISSR) and the
    conformal RHi interval on a disjoint, group-held-out calibration subset."""
    from sklearn.model_selection import GroupShuffleSplit

    groups = cv_groups(train_df)
    gss = GroupShuffleSplit(n_splits=1, test_size=calib_frac, random_state=cfg.random_state)
    fit_idx, cal_idx = next(gss.split(train_df, groups=groups))
    fit_df, cal_df = train_df.iloc[fit_idx], train_df.iloc[cal_idx]

    corrector = _new_corrector(cfg)
    corrector.fit(fit_df, fit_df[LABEL_DELTA].to_numpy())

    rhi_hat, _s, p_raw = corrector.predict(cal_df)
    calibrator = ProbabilityCalibrator(method="isotonic").fit(
        p_raw, cal_df[LABEL_ISSR].to_numpy()
    )
    conformal = ConformalRHi(alpha=0.1).fit(cal_df[LABEL_RHI].to_numpy(), rhi_hat)

    served = ServedModel(corrector=corrector, calibrator=calibrator,
                         conformal_half_width=conformal.half_width)
    return served, conformal


def _new_corrector(cfg: MLConfig, n_ensemble: int | None = None) -> RHiCorrector:
    m = RHiCorrector.from_config(cfg)
    if n_ensemble is not None:
        m.n_ensemble = n_ensemble
    return m


# ===========================================================================
# top-level
# ===========================================================================


def run_training(
    cfg: MLConfig,
    df: pd.DataFrame,
    *,
    log_mlflow: bool = True,
    do_cv: bool = True,
) -> TrainResult:
    """Full training run. Returns a TrainResult; logs to MLflow unless disabled."""
    from .data.schema import validate_table

    validate_table(df)
    dh = dataset_hash(df)
    train_df, test_df = temporal_split(df, cfg)

    cv_metrics = cross_validate(cfg, train_df) if do_cv else {}
    served, conformal = fit_and_calibrate(cfg, train_df)

    baselines = [b.fit(train_df) for b in default_baselines(cfg.issr_rhi_threshold)]
    comparison = comparison_table(test_df, served, baselines,
                                  p_threshold=cfg.issr_p_threshold)

    # Headline metrics = the model's holdout row.
    rhi_hat, _std, p = served.predict(test_df)
    holdout = score_predictions(test_df[LABEL_RHI].to_numpy(),
                                test_df[LABEL_ISSR].to_numpy(), rhi_hat, p,
                                p_threshold=cfg.issr_p_threshold)

    result = TrainResult(served=served, conformal=conformal, comparison=comparison,
                         cv_metrics=cv_metrics, holdout_metrics=holdout, dataset_hash=dh)

    if log_mlflow:
        from .registry import log_run

        figures = {
            "reliability": plot_reliability(p, test_df[LABEL_ISSR].to_numpy()),
            "rhi_scatter": plot_rhi_scatter(test_df[LABEL_RHI].to_numpy(), rhi_hat,
                                            test_df["rhi"].to_numpy()),
        }
        params = {
            "regime_split_rhi": cfg.regime_split_rhi,
            "n_ensemble": cfg.n_ensemble,
            "xgb_n_estimators": cfg.xgb_n_estimators,
            "xgb_max_depth": cfg.xgb_max_depth,
            "issr_p_threshold": cfg.issr_p_threshold,
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
        }
        metrics = {**{f"cv_{k}": v for k, v in cv_metrics.items()},
                   **{f"holdout_{k}": v for k, v in holdout.items()}}
        result.mlflow = log_run(cfg, served, params=params, metrics=metrics,
                                comparison=comparison, figures=figures,
                                dataset_hash=dh)
    return result
