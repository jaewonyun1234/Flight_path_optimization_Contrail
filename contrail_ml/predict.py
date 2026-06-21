"""
predict.py — Turn a trained model + a weather snapshot into an MLIssrField.

Serving runs the model over a `(lon, lat, pressure)` grid and packs the
corrected-RHi-excess and calibrated P(ISSR) cubes into an `MLIssrField`
(issr_field.py), which the optimizer consumes through the ISSRField interface.

By default we serve from a GFS FORECAST — operationally you want to know where
ISSR *will* be when the aircraft arrives, not where it is now. ERA5 replay is
available for retrospective studies. Every feature is computed by the shared
`features.add_derived_features`, restricted to the ERA5∩GFS variable set, so no
feature is ever missing at serve time.

A guarded `synthetic` path builds a clearly-labelled fake met cube and a model
trained on the synthetic fallback, so `serve-check` and the integration tests
exercise the entire seam with no network and no credentials.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .features import RAW_NWP_COLUMNS, add_derived_features, rhi_excess
from .issr_field import MLIssrField

if TYPE_CHECKING:
    from .config import MLConfig


@dataclass
class MetCube:
    """A weather snapshot on an ascending `(lon, lat, pressure)` grid.

    `fields` maps each raw NWP variable name (features.RAW_NWP_COLUMNS) to an
    array of shape (n_lon, n_lat, n_pressure). `time` is a single valid time
    applied to every cell.
    """

    lon_axis: np.ndarray
    lat_axis: np.ndarray
    pressure_axis: np.ndarray
    fields: dict[str, np.ndarray]
    time: Any  # np.datetime64 / str / pandas Timestamp

    def shape(self) -> tuple[int, int, int]:
        return (len(self.lon_axis), len(self.lat_axis), len(self.pressure_axis))


# ===========================================================================
# Core: met cube + model -> MLIssrField
# ===========================================================================


def field_from_metcube(
    model: Any,
    cube: MetCube,
    cfg: MLConfig,
    *,
    mode: str = "prob",
) -> MLIssrField:
    """Run `model` over every cell of `cube` and build an MLIssrField."""
    nlon, nlat, npr = cube.shape()
    lon3, lat3, p3 = np.meshgrid(cube.lon_axis, cube.lat_axis, cube.pressure_axis,
                                 indexing="ij")
    rows = {
        "lon": lon3.ravel(),
        "lat": lat3.ravel(),
        "pressure_hpa": p3.ravel(),
        "time": pd.to_datetime(cube.time),
    }
    for name in RAW_NWP_COLUMNS:
        if name not in cube.fields:
            raise KeyError(f"met cube missing required field {name!r}")
        rows[name] = np.asarray(cube.fields[name], dtype=float).ravel()

    df = add_derived_features(pd.DataFrame(rows), pressure_col="pressure_hpa")
    rhi_hat, _rhi_std, p_issr = model.predict(df)

    rhi_exc_cube = np.asarray(rhi_excess(rhi_hat), dtype=float).reshape(nlon, nlat, npr)
    p_issr_cube = np.asarray(p_issr, dtype=float).reshape(nlon, nlat, npr)

    return MLIssrField(
        lon_axis=np.asarray(cube.lon_axis, dtype=float),
        lat_axis=np.asarray(cube.lat_axis, dtype=float),
        pressure_axis=np.asarray(cube.pressure_axis, dtype=float),
        rhi_excess_cube=rhi_exc_cube,
        p_issr_cube=p_issr_cube,
        anchor=cfg.anchor,
        threshold=0.3,  # RHi-excess cut for the pure-RHi mode (parity w/ ISSRField)
        p_threshold=cfg.issr_p_threshold,
        mode=mode,
    )


# ===========================================================================
# Real serving path (GFS forecast / ERA5 replay)
# ===========================================================================


def real_ml_issr_field(
    cfg: MLConfig,
    met_source: str = "gfs",
    time: str | None = None,
    *,
    model: Any = None,
    grid_res_deg: float = 1.0,
    mode: str = "prob",
) -> MLIssrField:
    """Load a registered model + real weather and build the field.

    Not hermetic: needs the `[ml]` geo stack and network/credentials. Kept thin
    so the offline tests don't depend on it.
    """
    if model is None:
        from .registry import load_model

        model = load_model(cfg)

    if met_source == "gfs":
        from .data.gfs import load_forecast_cube

        cube = load_forecast_cube(cfg, time=time, grid_res_deg=grid_res_deg)
    elif met_source == "era5":
        from .data.era5 import load_met_cube

        cube = load_met_cube(cfg, time=time, grid_res_deg=grid_res_deg)
    else:
        raise ValueError(f"unknown met_source {met_source!r} (use gfs/era5/synthetic)")

    return field_from_metcube(model, cube, cfg, mode=mode)


# ===========================================================================
# Hermetic synthetic path (tests / serve-check demo) — clearly labelled
# ===========================================================================


def synthetic_ml_issr_field(
    cfg: MLConfig,
    *,
    model: Any = None,
    allow_synthetic: bool = False,
    grid_res_deg: float = 2.0,
    seed: int = 0,
    mode: str = "prob",
) -> MLIssrField:
    """Build an MLIssrField from a fake met cube + fallback-trained model.

    For tests and offline demos ONLY; guarded so fake weather can't slip into a
    real run.
    """
    if not allow_synthetic:
        raise RuntimeError(
            "synthetic_ml_issr_field refused: pass allow_synthetic=True. The "
            "synthetic met cube is for tests/serve-check demos only."
        )
    if model is None:
        model = _quick_fallback_model(cfg, seed=seed)
    cube = synthetic_met_cube(cfg, grid_res_deg=grid_res_deg, seed=seed,
                              allow_synthetic=True)
    return field_from_metcube(model, cube, cfg, mode=mode)


def _quick_fallback_model(cfg: MLConfig, seed: int = 0) -> Any:
    """Train a small RHiCorrector on the synthetic fallback (tests/demo)."""
    from .data.schema import LABEL_DELTA
    from .data.synthetic_fallback import make_synthetic_training_table
    from .model import RHiCorrector

    df = make_synthetic_training_table(n=3000, seed=seed, allow_synthetic=True)
    m = RHiCorrector(
        regime_split_rhi=cfg.regime_split_rhi,
        issr_rhi_threshold=cfg.issr_rhi_threshold,
        n_ensemble=3,
        xgb_n_estimators=80,
        min_regime_samples=30,
        random_state=cfg.random_state,
    )
    m.fit(df, df[LABEL_DELTA].to_numpy())
    return m


def synthetic_met_cube(
    cfg: MLConfig,
    grid_res_deg: float = 2.0,
    seed: int = 0,
    *,
    allow_synthetic: bool = False,
) -> MetCube:
    """Fabricate a physically-flavoured fake weather cube (tests/demo ONLY).

    Builds a few smooth humid 'blobs' in (lon, lat, pressure) whose NWP RHi
    peaks just below saturation; the model's dry-bias correction then pushes
    the blob cores past 100 % — a nice illustration of recovering hidden ISSR
    from a dry-biased forecast. Auxiliary fields (cloud ice, vertical velocity,
    cloud cover) are correlated with humidity the same way the training
    fallback is, so the learned relationship transfers.
    """
    if not allow_synthetic:
        raise RuntimeError("synthetic_met_cube refused: pass allow_synthetic=True.")
    warnings.warn(
        "USING SYNTHETIC MET CUBE — fabricated weather, not a real forecast. "
        "Valid for tests/serve-check demos only.",
        stacklevel=2,
    )
    from .config import SIM_DOMAIN
    from .features import EPSILON, e_si_pa

    rng = np.random.default_rng(seed)
    lon_axis = np.arange(cfg.lon_min, cfg.lon_max + 1e-6, grid_res_deg)
    lat_axis = np.arange(cfg.lat_min, cfg.lat_max + 1e-6, grid_res_deg)
    pressure_axis = np.array(sorted(cfg.pressure_levels_hpa), dtype=float)
    lon3, lat3, p3 = np.meshgrid(lon_axis, lat_axis, pressure_axis, indexing="ij")

    # Temperature: colder aloft, smooth latitude gradient.
    T = 225.0 - 0.05 * (p3 - 225.0) - 0.15 * (lat3 - 50.0)

    # Place the humid blobs INSIDE the geographic footprint of the local sim
    # box (so the optimizer actually sees ISSR to route around), not scattered
    # across the whole continental bbox. The footprint comes from the SAME
    # anchor the field/GUI use.
    x0, x1, y0, y1, _z0, _z1 = SIM_DOMAIN
    corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    geo = [cfg.anchor.local_to_geo(cx, cy) for cx, cy in corners]
    box_lon_lo = min(g[0] for g in geo)
    box_lon_hi = max(g[0] for g in geo)
    box_lat_lo = min(g[1] for g in geo)
    box_lat_hi = max(g[1] for g in geo)

    # Cruise pressures: only the levels the FL340-400 band actually maps onto
    # (~187-250 hPa) carry blob cores, so the aircraft fly through the ISSR.
    cruise_levels = [p for p in pressure_axis if 180.0 <= p <= 255.0] or [225.0]
    # Keep cores comfortably inside the box footprint (margin > blob sigma).
    lon_lo, lon_hi = box_lon_lo + 2.0, box_lon_hi - 2.0
    lat_lo, lat_hi = box_lat_lo + 1.5, box_lat_hi - 1.5

    # A handful of Gaussian humid blobs -> a "true" RHi field.
    true_rhi = np.full(lon3.shape, 35.0)
    for _ in range(6):
        c_lon = rng.uniform(lon_lo, lon_hi)
        c_lat = rng.uniform(lat_lo, lat_hi)
        c_p = float(rng.choice(cruise_levels))
        amp = rng.uniform(70.0, 100.0)
        true_rhi += amp * np.exp(
            -(((lon3 - c_lon) / 6.0) ** 2
              + ((lat3 - c_lat) / 4.0) ** 2
              + ((p3 - c_p) / 30.0) ** 2)
        )
    true_rhi = np.clip(true_rhi, 5.0, 150.0)

    humid = np.clip(true_rhi - 90.0, 0.0, None)
    cloud_ice = np.clip(8e-6 * humid / 60.0, 0.0, None)
    w = -0.05 * np.clip(true_rhi - 100.0, 0.0, None) / 50.0
    cloud_cover = np.clip(0.15 + 0.6 * np.clip(true_rhi - 80.0, 0.0, None) / 70.0, 0.0, 1.0)
    u = np.full(lon3.shape, 12.0)
    v = np.zeros(lon3.shape)

    # Apply the SAME dry bias the training fallback uses, so the NWP cube is
    # dry-biased and the model has the correction to apply.
    bias = 2.0 + 0.28 * np.clip(true_rhi - 85.0, 0.0, None) + 1.5e6 * cloud_ice
    nwp_rhi = np.clip(true_rhi - bias, 1.0, 200.0)

    esi = np.asarray(e_si_pa(T), dtype=float)
    e = (nwp_rhi / 100.0) * esi
    q_nwp = EPSILON * e / (p3 * 100.0 - (1.0 - EPSILON) * e)

    fields = {
        "T": T, "specific_humidity": q_nwp, "u": u, "v": v, "w": w,
        "cloud_cover": cloud_cover, "cloud_ice": cloud_ice,
    }
    return MetCube(lon_axis=lon_axis, lat_axis=lat_axis, pressure_axis=pressure_axis,
                   fields=fields, time=cfg.ref_time)


# ===========================================================================
# Tabular prediction (used by the monitor)
# ===========================================================================


def predict_table(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Append model outputs to a feature dataframe (corrected RHi, std, p_issr)."""
    feat = add_derived_features(df) if "rhi" not in df.columns else df
    rhi_hat, rhi_std, p_issr = model.predict(feat)
    out = feat.copy()
    out["rhi_hat"] = rhi_hat
    out["rhi_std"] = rhi_std
    out["p_issr"] = p_issr
    return out
