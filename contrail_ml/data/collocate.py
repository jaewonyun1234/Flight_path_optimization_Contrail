"""
collocate.py — Put ERA5 onto the IAGOS waypoints (plan §4).

For each IAGOS measurement we need the NWP fields AT THAT point in space-time, so
the model input row lines up with the truth label. This is a 4-D (time, level,
lat, lon) linear interpolation of the ERA5 dataset onto the waypoint coordinates
via xarray `.interp`.

Known caveat (documented, not hidden): IAGOS aircraft avoid deep convection, so
the sample is biased away from the most intense humidity. We surface this as an
optional `sample_weight` column and call it out in docs/ML.md rather than
pretending the sample is unbiased.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..features import RAW_NWP_COLUMNS

if TYPE_CHECKING:
    from ..config import MLConfig


def collocate_era5(iagos_df: pd.DataFrame, cfg: MLConfig) -> pd.DataFrame:
    """Interpolate ERA5 raw NWP fields onto IAGOS waypoints.

    Returns the IAGOS frame with the RAW_NWP_COLUMNS added (the model inputs),
    so the caller can compute features and attach labels. Heavy imports are
    deferred; fails loudly via era5.open_era5 if the geo stack is missing.
    """
    import xarray as xr

    from .era5 import ERA5_VARIABLES, open_era5

    if iagos_df.empty:
        raise ValueError("collocate_era5 got an empty IAGOS frame")

    times = pd.to_datetime(iagos_df["time"])
    t0, t1 = times.min(), times.max()
    met = open_era5(cfg, (np.datetime64(t0), np.datetime64(t1)))
    ds = getattr(met, "data", met)

    pts = {
        "time": xr.DataArray(times.values, dims="points"),
        "level": xr.DataArray(iagos_df["pressure_hpa"].to_numpy(), dims="points"),
        "latitude": xr.DataArray(iagos_df["lat"].to_numpy(), dims="points"),
        "longitude": xr.DataArray(iagos_df["lon"].to_numpy(), dims="points"),
    }
    interp = ds.interp(**pts)

    out = iagos_df.copy().reset_index(drop=True)
    for name in RAW_NWP_COLUMNS:
        cf = ERA5_VARIABLES[name]
        out[name] = np.asarray(interp[cf].values, dtype=float)

    # The IAGOS T/q are the truth; the ERA5 T/q are the model inputs. Keep the
    # ERA5 versions under the canonical feature names, and the measured RHi as
    # the label (already on iagos_df as rhi_iagos).
    return out.dropna(subset=list(RAW_NWP_COLUMNS)).reset_index(drop=True)


def add_sampling_weight(df: pd.DataFrame, enable: bool = False) -> pd.DataFrame:
    """Optionally add a `sample_weight` correcting (crudely) for IAGOS avoiding
    the most humid/convective air. Off by default; documented in ML.md."""
    out = df.copy()
    if not enable:
        out["sample_weight"] = 1.0
        return out
    # Up-weight the under-sampled humid tail so the loss isn't dominated by the
    # over-represented dry regime.
    rhi = out["rhi"].to_numpy() if "rhi" in out.columns else out["rhi_iagos"].to_numpy()
    w = 1.0 + np.clip(rhi - 90.0, 0.0, None) / 30.0
    out["sample_weight"] = w
    return out
