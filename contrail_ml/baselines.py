"""
baselines.py — The honest competition (plan §0.3, §6).

Before claiming the ML model "beats physics", it has to beat the cheap things.
These three baselines all expose the same `predict_rhi` / `predict_proba`
surface as the model, so `evaluate.py` can score them side by side:

  RawERA5Baseline           : use the NWP RHi as-is (the uncorrected forecast).
  MultiplicativeFactorBaseline : one global factor f minimising ||f*rhi - truth||.
  QuantileMappingBaseline   : bivariate (T, RHi) empirical quantile mapping — the
                              standard statistical bias-correction the ML must beat.

A baseline's classification score is its corrected RHi passed through a fixed
logistic centred at the ISSR threshold, so ROC/PR/Brier are all well defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Logistic scale (in % RHi) used to turn a corrected-RHi baseline into a
# probability. Deliberately fixed (uncalibrated) — the point of the baselines
# is to be simple; calibration is the model's advantage.
_LOGISTIC_SCALE = 8.0


def _logistic_from_rhi(rhi: np.ndarray, threshold: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.asarray(rhi, dtype=float) - threshold) / _LOGISTIC_SCALE))


@dataclass
class RawERA5Baseline:
    """The uncorrected NWP forecast: corrected RHi == raw RHi."""

    issr_threshold: float = 100.0
    name: str = "raw_era5"

    def fit(self, df: pd.DataFrame) -> RawERA5Baseline:
        return self

    def predict_rhi(self, df: pd.DataFrame) -> np.ndarray:
        return df["rhi"].to_numpy(dtype=float)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return _logistic_from_rhi(self.predict_rhi(df), self.issr_threshold)


@dataclass
class MultiplicativeFactorBaseline:
    """Scale the NWP RHi by a single least-squares factor f."""

    issr_threshold: float = 100.0
    name: str = "x_factor"
    factor: float = 1.0

    def fit(self, df: pd.DataFrame) -> MultiplicativeFactorBaseline:
        rhi = df["rhi"].to_numpy(dtype=float)
        truth = df["rhi_iagos"].to_numpy(dtype=float)
        denom = float(np.sum(rhi * rhi))
        self.factor = float(np.sum(rhi * truth) / denom) if denom > 0 else 1.0
        return self

    def predict_rhi(self, df: pd.DataFrame) -> np.ndarray:
        return self.factor * df["rhi"].to_numpy(dtype=float)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return _logistic_from_rhi(self.predict_rhi(df), self.issr_threshold)


@dataclass
class QuantileMappingBaseline:
    """Bivariate (T, RHi) empirical quantile mapping.

    Stratify by temperature bin; within each bin learn the monotone map from
    the NWP-RHi quantiles to the IAGOS-RHi quantiles, then apply it at predict
    time. This is the conventional statistical bias-correction the ML version
    has to outperform to justify itself.
    """

    issr_threshold: float = 100.0
    n_temp_bins: int = 8
    n_quantiles: int = 50
    name: str = "quantile_map"
    _t_edges: np.ndarray = field(default_factory=lambda: np.array([]))
    _src_q: list = field(default_factory=list)
    _dst_q: list = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> QuantileMappingBaseline:
        t = df["T"].to_numpy(dtype=float)
        rhi = df["rhi"].to_numpy(dtype=float)
        truth = df["rhi_iagos"].to_numpy(dtype=float)

        self._t_edges = np.quantile(t, np.linspace(0, 1, self.n_temp_bins + 1))
        self._t_edges[0] -= 1e-6
        self._t_edges[-1] += 1e-6
        qs = np.linspace(0, 1, self.n_quantiles)
        self._src_q, self._dst_q = [], []
        for lo, hi in zip(self._t_edges[:-1], self._t_edges[1:], strict=False):
            m = (t >= lo) & (t < hi)
            if m.sum() < 5:
                # too few points: identity map for this bin
                self._src_q.append(None)
                self._dst_q.append(None)
                continue
            self._src_q.append(np.quantile(rhi[m], qs))
            self._dst_q.append(np.quantile(truth[m], qs))
        return self

    def predict_rhi(self, df: pd.DataFrame) -> np.ndarray:
        t = df["T"].to_numpy(dtype=float)
        rhi = df["rhi"].to_numpy(dtype=float)
        out = rhi.copy()
        bin_idx = np.clip(np.digitize(t, self._t_edges) - 1, 0, len(self._src_q) - 1)
        for b in range(len(self._src_q)):
            if self._src_q[b] is None:
                continue
            sel = bin_idx == b
            if sel.any():
                out[sel] = np.interp(rhi[sel], self._src_q[b], self._dst_q[b])
        return out

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return _logistic_from_rhi(self.predict_rhi(df), self.issr_threshold)


def default_baselines(issr_threshold: float = 100.0) -> list:
    """The standard baseline set used by the comparison table."""
    return [
        RawERA5Baseline(issr_threshold=issr_threshold),
        MultiplicativeFactorBaseline(issr_threshold=issr_threshold),
        QuantileMappingBaseline(issr_threshold=issr_threshold),
    ]
