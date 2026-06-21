"""
features.py — SHARED feature engineering (training AND serving).

This is the anti-skew guarantee (plan §0.2): every feature the model ever sees
is computed here, once, by code used identically by the training-table builder
(`data/build_dataset.py`) and the live predictor (`predict.py`). If a feature
were computed one way at train time and another at serve time, the model would
silently degrade in production. Centralizing it makes that class of bug
impossible.

Contents
========
1. Thermodynamics — the physics that turns (T, q, p) into relative humidity
   over ice (RHi). Murphy & Koop (2005) ice-saturation vapour pressure.
2. Standard-atmosphere altitude <-> pressure (ISA barometric formula).
3. Cyclical time encodings (solar time, day-of-year) as sin/cos pairs.
4. The geographic anchor mapping the local sim frame <-> real (lon, lat).
5. `FEATURES` — the exact, ordered list of model-input columns, restricted
   to variables present in BOTH ERA5 and GFS so nothing is missing at serve
   time.

All thermodynamic functions accept floats or numpy arrays and return the same
shape, so they vectorize over a whole grid or a whole training table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pandas is in the [ml] extra; keep import-time deps light.
    import pandas as pd

# ===========================================================================
# 1. THERMODYNAMICS
# ===========================================================================

# Dry-air / water-vapour molecular-mass ratio (Rd/Rv).
EPSILON = 0.621981


def e_si_pa(temperature_k: np.ndarray | float) -> np.ndarray | float:
    """Saturation vapour pressure OVER ICE, in Pa. Murphy & Koop (2005).

        e_si = exp(9.550426 - 5723.265/T + 3.53068*ln(T) - 0.00728332*T)

    Valid for T in roughly [110, 273.16] K, which covers the entire upper
    troposphere where ISSRs live. T in kelvin.
    """
    t = np.asarray(temperature_k, dtype=float)
    out = np.exp(
        9.550426
        - 5723.265 / t
        + 3.53068 * np.log(t)
        - 0.00728332 * t
    )
    return _maybe_scalar(out, temperature_k)


def vapor_pressure_pa(
    specific_humidity: np.ndarray | float,
    pressure_pa: np.ndarray | float,
) -> np.ndarray | float:
    """Water-vapour partial pressure (Pa) from specific humidity q and total
    pressure p:

        e = q * p / (EPSILON + (1 - EPSILON) * q)

    q is dimensionless (kg water / kg moist air); p and e share units (Pa).
    """
    q = np.asarray(specific_humidity, dtype=float)
    p = np.asarray(pressure_pa, dtype=float)
    e = q * p / (EPSILON + (1.0 - EPSILON) * q)
    return _maybe_scalar(e, specific_humidity)


def rhi_percent(
    temperature_k: np.ndarray | float,
    specific_humidity: np.ndarray | float,
    pressure_hpa: np.ndarray | float,
) -> np.ndarray | float:
    """Relative humidity over ice, in PERCENT, from (T[K], q, p[hPa]).

        RHi = 100 * e / e_si(T)

    RHi >= 100 % is the supersaturation that lets a contrail persist.
    """
    p_pa = np.asarray(pressure_hpa, dtype=float) * 100.0
    e = np.asarray(vapor_pressure_pa(specific_humidity, p_pa), dtype=float)
    esi = np.asarray(e_si_pa(temperature_k), dtype=float)
    rhi = 100.0 * e / esi
    return _maybe_scalar(rhi, temperature_k)


def rhi_excess(rhi_pct: np.ndarray | float) -> np.ndarray | float:
    """ISSRField semantics: max(0, RHi/100 - 1).

    0 outside supersaturation; grows with how far above ice-saturation the
    air is. This is exactly the quantity the synthetic ISSR blobs return, so
    the ML field is a drop-in.
    """
    r = np.asarray(rhi_pct, dtype=float)
    out = np.clip(r / 100.0 - 1.0, 0.0, None)
    return _maybe_scalar(out, rhi_pct)


# ===========================================================================
# 2. STANDARD ATMOSPHERE — altitude <-> pressure
# ===========================================================================

_T0 = 288.15        # sea-level standard temperature (K)
_L = 0.0065         # tropospheric lapse rate (K/m)
_P0 = 1013.25       # sea-level standard pressure (hPa)
_G = 9.80665        # gravity (m/s^2)
_R = 287.05         # specific gas constant for dry air (J/kg/K)
_H_TROP = 11_000.0  # tropopause altitude (m)
_EXP = _G / (_R * _L)                       # ~5.2559
_P_TROP = _P0 * (1.0 - _L * _H_TROP / _T0) ** _EXP  # ~226.32 hPa
_T_TROP = _T0 - _L * _H_TROP                # 216.65 K


def altitude_to_pressure_hpa(altitude_m: np.ndarray | float) -> np.ndarray | float:
    """ISA pressure (hPa) at a geometric altitude (m). Piecewise: lapse-rate
    troposphere below 11 km, isothermal stratosphere above."""
    h = np.asarray(altitude_m, dtype=float)
    trop = _P0 * np.clip(1.0 - _L * h / _T0, 1e-9, None) ** _EXP
    strat = _P_TROP * np.exp(-_G * (h - _H_TROP) / (_R * _T_TROP))
    out = np.where(h <= _H_TROP, trop, strat)
    return _maybe_scalar(out, altitude_m)


def pressure_to_altitude_m(pressure_hpa: np.ndarray | float) -> np.ndarray | float:
    """Inverse of altitude_to_pressure_hpa (ISA)."""
    p = np.asarray(pressure_hpa, dtype=float)
    trop = (_T0 / _L) * (1.0 - (p / _P0) ** (1.0 / _EXP))
    strat = _H_TROP - (_R * _T_TROP / _G) * np.log(np.clip(p / _P_TROP, 1e-9, None))
    out = np.where(p >= _P_TROP, trop, strat)
    return _maybe_scalar(out, pressure_hpa)


# ===========================================================================
# 3. CYCLICAL TIME ENCODINGS
# ===========================================================================
# A model must not learn "11pm and midnight are 23 hours apart". Encoding a
# periodic quantity as a (sin, cos) pair puts the two ends of the cycle next
# to each other in feature space.


def solar_time_encoding(
    utc_hour: np.ndarray | float,
    lon_deg: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """(sin, cos) of LOCAL solar time. Local solar hour ~= UTC + lon/15."""
    h = np.asarray(utc_hour, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    local = np.mod(h + lon / 15.0, 24.0)
    ang = 2.0 * np.pi * local / 24.0
    return _maybe_scalar(np.sin(ang), utc_hour), _maybe_scalar(np.cos(ang), utc_hour)


def day_of_year_encoding(
    day_of_year: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """(sin, cos) of the day-of-year (seasonal cycle)."""
    d = np.asarray(day_of_year, dtype=float)
    ang = 2.0 * np.pi * d / 365.25
    return _maybe_scalar(np.sin(ang), day_of_year), _maybe_scalar(np.cos(ang), day_of_year)


# ===========================================================================
# 4. GEOGRAPHIC ANCHOR — local sim frame <-> real (lon, lat)
# ===========================================================================


@dataclass(frozen=True)
class GeoAnchor:
    """Maps the local sim frame (x_km east, y_km north) to geography.

    The sim is a flat 1500x800 km Cartesian box; real weather lives on
    (lon, lat). One small-angle anchor ties them together so the model, the
    predicted field, and the GUI map all agree on where things are.

        lat = origin_lat + y_km / KM_PER_DEG_LAT
        lon = origin_lon + x_km / (KM_PER_DEG_LAT * cos(lat))
    """

    origin_lat: float
    origin_lon: float
    km_per_deg_lat: float = 111.0

    def local_to_geo(
        self, x_km: np.ndarray | float, y_km: np.ndarray | float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """(x_km, y_km) -> (lon, lat) in degrees."""
        x = np.asarray(x_km, dtype=float)
        y = np.asarray(y_km, dtype=float)
        lat = self.origin_lat + y / self.km_per_deg_lat
        lon = self.origin_lon + x / (self.km_per_deg_lat * np.cos(np.radians(lat)))
        return _maybe_scalar(lon, x_km), _maybe_scalar(lat, y_km)

    def geo_to_local(
        self, lon_deg: np.ndarray | float, lat_deg: np.ndarray | float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """(lon, lat) -> (x_km, y_km). Inverse of local_to_geo."""
        lon = np.asarray(lon_deg, dtype=float)
        lat = np.asarray(lat_deg, dtype=float)
        y = (lat - self.origin_lat) * self.km_per_deg_lat
        x = (lon - self.origin_lon) * self.km_per_deg_lat * np.cos(np.radians(lat))
        return _maybe_scalar(x, lon_deg), _maybe_scalar(y, lat_deg)


# ===========================================================================
# 5. THE FEATURE LIST + the shared table builder
# ===========================================================================

# Raw NWP columns expected on an input row (present in BOTH ERA5 and GFS).
RAW_NWP_COLUMNS: tuple[str, ...] = (
    "T",                  # temperature, K
    "specific_humidity",  # kg/kg
    "u", "v", "w",        # wind components (m/s; w = vertical velocity Pa/s)
    "cloud_cover",        # fraction [0,1]
    "cloud_ice",          # specific cloud ice water content, kg/kg
)

# The exact, ordered model-input columns. `rhi` is the NWP relative humidity
# over ice (the quantity the model corrects) — computed from THIS row's T/q/p,
# whether that row came from ERA5 (train) or GFS (serve). One name, no skew.
FEATURES: tuple[str, ...] = (
    "T",
    "specific_humidity",
    "rhi",
    "u",
    "v",
    "w",
    "cloud_cover",
    "cloud_ice",
    "altitude_m",
    "sin_solar",
    "cos_solar",
    "sin_doy",
    "cos_doy",
)


def add_derived_features(df: pd.DataFrame, pressure_col: str = "pressure_hpa") -> pd.DataFrame:
    """Add every derived feature column to `df` IN PLACE-SAFE fashion.

    Requires raw columns: T, specific_humidity, <pressure_col>, u, v, w,
    cloud_cover, cloud_ice, lon, and a `time` column (datetime-like or a
    numeric UTC-hour fallback). Returns a NEW frame with the FEATURES columns
    present, so the same call site serves both the training table and the
    live grid.
    """
    out = df.copy()

    out["rhi"] = rhi_percent(out["T"].to_numpy(), out["specific_humidity"].to_numpy(),
                             out[pressure_col].to_numpy())
    out["rhi_excess"] = rhi_excess(out["rhi"].to_numpy())
    out["altitude_m"] = pressure_to_altitude_m(out[pressure_col].to_numpy())

    utc_hour, doy = _time_parts(out)
    sin_solar, cos_solar = solar_time_encoding(utc_hour, out["lon"].to_numpy())
    sin_doy, cos_doy = day_of_year_encoding(doy)
    out["sin_solar"] = sin_solar
    out["cos_solar"] = cos_solar
    out["sin_doy"] = sin_doy
    out["cos_doy"] = cos_doy
    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the FEATURES columns, in order. Raises if any is absent
    (a loud failure beats silently training on the wrong columns)."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(
            f"feature_matrix: missing columns {missing}; "
            f"did you call add_derived_features() first?"
        )
    return df.loc[:, list(FEATURES)]


# ===========================================================================
# Internal helpers
# ===========================================================================


def _maybe_scalar(arr: np.ndarray, like: np.ndarray | float) -> np.ndarray | float:
    """Return a Python float when the input was scalar, else the array."""
    if np.isscalar(like) or (isinstance(like, np.ndarray) and like.ndim == 0):
        return float(arr)
    return arr


def _time_parts(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract (utc_hour, day_of_year) arrays from a `time` column.

    Accepts a real datetime column or, as a fallback for synthetic tests,
    explicit `utc_hour`/`day_of_year` numeric columns.
    """
    import pandas as pd

    if "time" in df.columns:
        t = pd.to_datetime(df["time"])
        utc_hour = (t.dt.hour + t.dt.minute / 60.0).to_numpy()
        doy = t.dt.dayofyear.to_numpy().astype(float)
        return utc_hour, doy
    if "utc_hour" in df.columns and "day_of_year" in df.columns:
        return df["utc_hour"].to_numpy(), df["day_of_year"].to_numpy()
    raise KeyError("add_derived_features needs a 'time' column (or utc_hour + day_of_year).")
