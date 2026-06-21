"""
config.py — Typed configuration for the contrail_ml pipeline.

One dataclass (`MLConfig`) carries every knob: the geographic region and
pressure levels to pull, the temporal train/test split, model hyperparameters,
the MLflow tracking URI, and the geographic anchor that maps the local sim
frame to real (lon, lat). It loads from environment variables (12-factor) or a
YAML file, and hardcodes NO secrets — credentials/paths come from the
environment.

Why a single config object?
    Both the training-table builder and the live predictor read the SAME
    region/levels/anchor from here, which (together with the shared
    `features.py`) is what guarantees train/serve consistency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

from .features import GeoAnchor

# The local sim domain, copied from contrail_env.synthetic_issr.ISSRField.domain
# (x_km, y_km, z_m). Kept here so the ML grid and the optimizer agree on extent.
SIM_DOMAIN: tuple[float, float, float, float, float, float] = (
    0.0, 1500.0, 0.0, 800.0, 9000.0, 12500.0
)


@dataclass(frozen=True)
class MLConfig:
    """All configuration for building, training, serving and monitoring."""

    # ---- Geographic region (degrees) and pressure levels (hPa) ----------
    # Europe / North-Atlantic box covering the cruise region.
    lon_min: float = -30.0
    lon_max: float = 30.0
    lat_min: float = 30.0
    lat_max: float = 70.0
    pressure_levels_hpa: tuple[int, ...] = (150, 175, 200, 225, 250, 300)

    # ---- Temporal split (inclusive year bounds) -------------------------
    train_years: tuple[int, int] = (2011, 2019)
    test_years: tuple[int, int] = (2020, 2022)

    # ---- Geographic anchor: local (x=0, y=0) -> (lat, lon), + ref time ---
    # Anchors the 1500x800 km sim box over south-west -> central Europe.
    origin_lat: float = 43.0
    origin_lon: float = -5.0
    ref_time: str = "2019-07-01T12:00:00"  # ISO; used for serving snapshots

    # ---- Model hyperparameters ------------------------------------------
    regime_split_rhi: float = 85.0   # % RHi boundary between dry/humid sub-models
    n_ensemble: int = 10             # bootstrap re-fits for predictive spread
    issr_rhi_threshold: float = 100.0  # RHi % at/above which a cell is ISSR
    issr_p_threshold: float = 0.5      # calibrated P(ISSR) cut for is_inside()
    xgb_max_depth: int = 6
    xgb_n_estimators: int = 300
    xgb_learning_rate: float = 0.05
    mlp_hidden: tuple[int, ...] = (64, 32)
    random_state: int = 42

    # ---- Paths / MLflow --------------------------------------------------
    data_dir: str = "data"
    mlflow_tracking_uri: str = "file:./mlruns"
    registered_model_name: str = "contrail-issr-rhi-corrector"

    # ---- Credentials / data paths (read from env, never hardcoded) ------
    iagos_dir: str | None = None     # local dir of IAGOS NetCDF files

    # Free-form extra knobs without widening the schema.
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #

    @property
    def anchor(self) -> GeoAnchor:
        """The geographic anchor used by BOTH the model grid and the GUI."""
        return GeoAnchor(origin_lat=self.origin_lat, origin_lon=self.origin_lon)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(lon_min, lon_max, lat_min, lat_max) for the weather loaders."""
        return (self.lon_min, self.lon_max, self.lat_min, self.lat_max)

    def with_overrides(self, **kwargs: Any) -> MLConfig:
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)

    # ------------------------------------------------------------------ #
    # Loaders
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls, prefix: str = "CONTRAIL_ML_") -> MLConfig:
        """Build a config, overriding any field from `${PREFIX}FIELD` env vars.

        Example: CONTRAIL_ML_MLFLOW_TRACKING_URI, CONTRAIL_ML_IAGOS_DIR.
        Only fields present in the environment are overridden; types are
        coerced from the dataclass defaults.
        """
        base = cls()
        overrides: dict[str, Any] = {}
        for f in base.__dataclass_fields__.values():  # type: ignore[attr-defined]
            if f.name == "extra":
                continue
            env_key = f"{prefix}{f.name.upper()}"
            if env_key in os.environ:
                overrides[f.name] = _coerce(getattr(base, f.name), os.environ[env_key])
        return replace(base, **overrides)

    @classmethod
    def from_yaml(cls, path: str) -> MLConfig:
        """Load a config from a YAML file (only known fields are applied)."""
        import yaml  # lazy: PyYAML is a transitive dep, not always present

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        clean = {k: v for k, v in data.items() if k in known}
        return cls(**clean)


def _coerce(reference: Any, raw: str) -> Any:
    """Coerce an env-var string to the type of the dataclass default value."""
    if isinstance(reference, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(raw)
    if isinstance(reference, float):
        return float(raw)
    if isinstance(reference, tuple):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        # Preserve element type from the first reference element if available.
        if reference and isinstance(reference[0], int):
            return tuple(int(p) for p in parts)
        if reference and isinstance(reference[0], float):
            return tuple(float(p) for p in parts)
        return tuple(parts)
    return raw


DEFAULT_CONFIG = MLConfig()
