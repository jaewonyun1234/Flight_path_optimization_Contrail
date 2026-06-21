"""
iagos.py — Load IAGOS in-situ aircraft humidity as ground truth (plan §4).

IAGOS (In-service Aircraft for a Global Observing System) equips commercial
aircraft with research-grade humidity sensors. Its measured RHi near the
tropopause is the closest thing to truth for ISSR — and the label this whole
project trains against. IAGOS distributes one NetCDF per flight; access needs a
(free) registration on the IAGOS portal (see docs/DATA.md). The data path comes
from config/env, never hardcoded, and no credentials live in the repo.

We RECOMPUTE RHi from the measured T, humidity and pressure with the SAME
thermodynamics as the features (contrail_ml.features), rather than trusting a
pre-computed RHi column, so the label and the features share one definition (no
label skew). The function fails loudly if the data directory is absent — it
never invents measurements.

VERIFY AGAINST DOCS: IAGOS variable names differ between products (IAGOS-CORE,
MOZAIC, CARIBIC). The mapping below is a documented default; override via config.
"""

from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..features import EPSILON, rhi_percent

if TYPE_CHECKING:
    from ..config import MLConfig

# Default IAGOS NetCDF variable names -> our canonical columns. Override per
# product through cfg.extra["iagos_variables"].
IAGOS_VARIABLES: dict[str, str] = {
    "time": "UTC_time",
    "lat": "lat",
    "lon": "lon",
    "pressure_pa": "air_press_AC",     # static pressure, Pa
    "temperature_k": "air_temp_AC",    # air temperature, K
    "h2o_vmr_ppmv": "H2O_gas_P",       # water-vapour volume mixing ratio, ppmv
    "qc": "H2O_gas_P_val",             # validation/quality flag (product-specific)
}

# Validation flags that we keep (product-specific; documented in DATA.md).
GOOD_QC_VALUES: tuple[int, ...] = (0, 1)


def load_iagos_waypoints(cfg: MLConfig) -> pd.DataFrame:
    """Load every IAGOS flight NetCDF under cfg.iagos_dir into one waypoint
    dataframe with columns: lat, lon, pressure_hpa, time, T, specific_humidity,
    rhi_iagos, qc_flag.

    Raises a clear error if the directory is missing/empty — never fabricates.
    """
    data_dir = cfg.iagos_dir or os.environ.get("CONTRAIL_ML_IAGOS_DIR")
    if not data_dir:
        raise FileNotFoundError(
            "IAGOS directory not set. Set cfg.iagos_dir or CONTRAIL_ML_IAGOS_DIR "
            "to a folder of IAGOS NetCDF files. See docs/DATA.md for registration."
        )
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"IAGOS directory does not exist: {data_dir}")
    files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    if not files:
        raise FileNotFoundError(f"no IAGOS *.nc files found in {data_dir}")

    var = {**IAGOS_VARIABLES, **(cfg.extra.get("iagos_variables", {}))}
    frames = [_load_one(f, var, cfg) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=["T", "specific_humidity", "pressure_hpa"]).reset_index(drop=True)


def _load_one(path: str, var: dict[str, str], cfg: MLConfig) -> pd.DataFrame:
    import xarray as xr  # lazy: part of the [ml] extra

    ds = xr.open_dataset(path)
    try:
        pressure_pa = np.asarray(ds[var["pressure_pa"]].values, dtype=float)
        temperature = np.asarray(ds[var["temperature_k"]].values, dtype=float)
        vmr_ppmv = np.asarray(ds[var["h2o_vmr_ppmv"]].values, dtype=float)
        lat = np.asarray(ds[var["lat"]].values, dtype=float)
        lon = np.asarray(ds[var["lon"]].values, dtype=float)
        time = pd.to_datetime(np.asarray(ds[var["time"]].values))
        qc = (np.asarray(ds[var["qc"]].values) if var["qc"] in ds else
              np.zeros_like(temperature, dtype=int))
    finally:
        ds.close()

    pressure_hpa = pressure_pa / 100.0
    # ppmv volume mixing ratio -> specific humidity (kg/kg).
    vmr = vmr_ppmv * 1e-6
    mmr = EPSILON * vmr            # mass mixing ratio
    q = mmr / (1.0 + mmr)

    rhi = np.asarray(rhi_percent(temperature, q, pressure_hpa), dtype=float)

    df = pd.DataFrame(
        {
            "lat": lat, "lon": lon, "pressure_hpa": pressure_hpa, "time": time,
            "T": temperature, "specific_humidity": q, "rhi_iagos": rhi,
            "qc_flag": np.asarray(qc).astype(int),
        }
    )
    # Quality screening + region/level window.
    df = df[df["qc_flag"].isin(GOOD_QC_VALUES)]
    lon_min, lon_max, lat_min, lat_max = cfg.bbox
    df = df[(df["lon"].between(lon_min, lon_max)) & (df["lat"].between(lat_min, lat_max))]
    p_lo, p_hi = min(cfg.pressure_levels_hpa), max(cfg.pressure_levels_hpa)
    df = df[df["pressure_hpa"].between(p_lo - 25, p_hi + 25)]
    return df.reset_index(drop=True)
