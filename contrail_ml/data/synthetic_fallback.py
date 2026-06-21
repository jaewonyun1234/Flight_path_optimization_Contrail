"""
synthetic_fallback.py — Hermetic, clearly-labelled fake training data.

PURPOSE AND THE HARD RULE
=========================
The whole project's credibility rests on never passing synthetic data off as
real observations (plan §0.1). This module is the *single* exception: a tiny
labelled table generated from a KNOWN function, used SOLELY by unit tests / CI
so the model, calibration, and integration code can be exercised with no
network and no credentials.

Two guards enforce that it can't leak into real work:
  1. It refuses to run unless the caller passes `allow_synthetic=True`.
  2. It emits a loud warning every time it is used, and stamps every row with
     source == "synthetic_fallback".

WHAT IT MODELS
==============
A physically-flavoured bias-correction problem. Each row has a "true"
(IAGOS-like) relative humidity over ice and a DRY-BIASED NWP humidity — the
real, documented failure mode of weather models near the tropopause. The bias
is small in dry air and grows in the humid regime (RHi > ~85%), and depends on
cloud-ice / vertical-velocity too, so:
  * the residual `delta = rhi_iagos - rhi(nwp)` is genuinely learnable,
  * the regime split at 85% RHi separates two different bias behaviours,
  * noise keeps P(ISSR) genuinely uncertain near the threshold (so calibration
    has something to do).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..features import (
    EPSILON,
    add_derived_features,
    e_si_pa,
)
from .schema import LABEL_DELTA, LABEL_ISSR, LABEL_RHI, TABLE_COLUMNS, issr_label

SOURCE_TAG = "synthetic_fallback"


def make_synthetic_training_table(
    n: int = 4000,
    seed: int = 0,
    *,
    allow_synthetic: bool = False,
    rhi_threshold: float = 100.0,
) -> pd.DataFrame:
    """Generate a labelled training table that matches the real schema.

    Parameters
    ----------
    n, seed : size and reproducibility.
    allow_synthetic : MUST be True. The guard exists so production code can
        never call this by accident.
    rhi_threshold : RHi % used to derive the binary ISSR label.

    Returns a DataFrame with exactly `schema.TABLE_COLUMNS`.
    """
    if not allow_synthetic:
        raise RuntimeError(
            "synthetic_fallback refused: pass allow_synthetic=True. This data "
            "is for tests/CI only and must never stand in for real IAGOS/ERA5."
        )
    warnings.warn(
        "USING SYNTHETIC FALLBACK DATA — not real observations. "
        "Valid for tests/CI only; never report metrics from this as science.",
        stacklevel=2,
    )

    rng = np.random.default_rng(seed)

    # ---- keys & geometry --------------------------------------------------
    pressure_hpa = rng.choice([150, 175, 200, 225, 250, 300], size=n).astype(float)
    lon = rng.uniform(-30.0, 30.0, size=n)
    lat = rng.uniform(30.0, 70.0, size=n)
    # Spread timestamps across several years and times of day.
    base = np.datetime64("2015-01-01T00:00:00")
    offsets = rng.integers(0, 6 * 365 * 24 * 60, size=n)  # minutes over ~6 yr
    time = base + offsets.astype("timedelta64[m]")

    # ---- temperature: colder higher up, with weather noise ----------------
    # ~225 K near 225 hPa, falling ~0.05 K/hPa toward lower pressure.
    T = 225.0 - 0.05 * (pressure_hpa - 225.0) + rng.normal(0.0, 4.0, size=n)
    T = np.clip(T, 200.0, 245.0)

    # ---- pick a TRUE RHi spanning dry..supersaturated ---------------------
    # Mixture: mostly sub-saturated, a meaningful humid/ISSR tail.
    true_rhi = rng.uniform(20.0, 150.0, size=n)

    # Auxiliary fields, partly correlated with humidity (physical flavour).
    cloud_ice = np.clip(
        rng.normal(0.0, 1.0, size=n) * 1e-6 + 8e-6 * np.clip(true_rhi - 90.0, 0.0, None) / 60.0,
        0.0, None,
    )
    w = rng.normal(0.0, 0.1, size=n) - 0.05 * np.clip(true_rhi - 100.0, 0.0, None) / 50.0
    cloud_cover = np.clip(0.15 + 0.6 * np.clip(true_rhi - 80.0, 0.0, None) / 70.0
                          + rng.normal(0.0, 0.1, size=n), 0.0, 1.0)
    u = rng.normal(10.0, 12.0, size=n)
    v = rng.normal(0.0, 12.0, size=n)

    # ---- the DRY BIAS: model underestimates humidity, worse when humid ----
    humid_excess = np.clip(true_rhi - 85.0, 0.0, None)
    bias = (
        2.0                                   # small baseline dry bias
        + 0.28 * humid_excess                 # grows in the humid regime
        + 1.5e6 * cloud_ice                   # ice-laden air biased more
        - 12.0 * np.clip(-w, 0.0, None)       # strong updraughts biased more
        + rng.normal(0.0, 4.0, size=n)        # irreducible noise
    )
    nwp_rhi = np.clip(true_rhi - bias, 1.0, 200.0)

    # Back out the NWP specific humidity consistent with nwp_rhi (so that the
    # `rhi` feature recomputed by features.py reproduces nwp_rhi exactly).
    q_nwp = _q_from_rhi(nwp_rhi, T, pressure_hpa)

    raw = pd.DataFrame(
        {
            "lat": lat,
            "lon": lon,
            "pressure_hpa": pressure_hpa,
            "time": time,
            "T": T,
            "specific_humidity": q_nwp,
            "u": u,
            "v": v,
            "w": w,
            "cloud_cover": cloud_cover,
            "cloud_ice": cloud_ice,
        }
    )

    df = add_derived_features(raw, pressure_col="pressure_hpa")

    # ---- labels (truth) + residual + meta --------------------------------
    df[LABEL_RHI] = true_rhi
    df[LABEL_DELTA] = true_rhi - df["rhi"].to_numpy()  # = bias the model learns
    df[LABEL_ISSR] = issr_label(true_rhi, rhi_threshold)
    df["qc_flag"] = "ok"
    df["source"] = SOURCE_TAG
    df["sample_weight"] = 1.0

    return df.loc[:, list(TABLE_COLUMNS)].reset_index(drop=True)


def _q_from_rhi(rhi_pct: np.ndarray, temperature_k: np.ndarray,
                pressure_hpa: np.ndarray) -> np.ndarray:
    """Invert RHi -> specific humidity q (kg/kg), the inverse of
    features.vapor_pressure_pa composed with rhi_percent."""
    esi = np.asarray(e_si_pa(temperature_k), dtype=float)
    e = (rhi_pct / 100.0) * esi               # vapour pressure, Pa
    p_pa = pressure_hpa * 100.0
    return EPSILON * e / (p_pa - (1.0 - EPSILON) * e)
