"""
calibrate.py — Make the probabilities mean what they say.

A classifier that outputs 0.7 should be right ~70 % of the time at that score.
Raw tree-ensemble probabilities usually aren't; we fix that with a monotone
post-hoc map fitted on a held-out fold:

    isotonic : non-parametric, monotone step function (default; flexible).
    platt    : a 1-D logistic squashing (smoother, needs less data).

We also provide a SPLIT-CONFORMAL interval on corrected RHi: a distribution-
free band rhi_hat ± q such that the true RHi lands inside ~(1 - alpha) of the
time, calibrated on held-out residuals. And `expected_calibration_error` (ECE)
to quantify how honest the probabilities are — the headline calibration metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ProbabilityCalibrator:
    """Post-hoc calibration map for P(ISSR). Fit on a held-out fold, then apply
    to every future probability."""

    method: str = "isotonic"
    _model: Any = None
    _fitted: bool = False

    def fit(self, p_raw: np.ndarray, y_true: np.ndarray) -> ProbabilityCalibrator:
        p = np.asarray(p_raw, dtype=float).reshape(-1)
        y = np.asarray(y_true, dtype=int).reshape(-1)

        if len(np.unique(y)) < 2:
            # Can't calibrate against a single class; act as identity.
            self._model = None
            self._fitted = True
            return self

        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p, y)
            self._model = iso
        elif self.method == "platt":
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(C=1e6, solver="lbfgs")
            lr.fit(p.reshape(-1, 1), y)
            self._model = lr
        else:
            raise ValueError(f"unknown calibration method {self.method!r}")
        self._fitted = True
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ProbabilityCalibrator.transform before fit().")
        p = np.asarray(p_raw, dtype=float).reshape(-1)
        if self._model is None:
            return np.clip(p, 0.0, 1.0)
        if self.method == "isotonic":
            return np.clip(self._model.predict(p), 0.0, 1.0)
        return self._model.predict_proba(p.reshape(-1, 1))[:, 1]

    # convenience
    def fit_transform(self, p_raw, y_true) -> np.ndarray:
        return self.fit(p_raw, y_true).transform(p_raw)


def expected_calibration_error(
    p: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error: mean |confidence - accuracy| over equal-width
    probability bins, weighted by bin population. Lower is better; the plan's
    target is ECE < 0.05."""
    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(y_true, dtype=float).reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        in_bin = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not in_bin.any():
            continue
        conf = p[in_bin].mean()
        acc = y[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(conf - acc)
    return float(ece)


def reliability_curve(
    p: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_confidence, bin_accuracy, bin_count) for a reliability
    diagram. Empty bins are dropped."""
    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(y_true, dtype=float).reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        in_bin = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not in_bin.any():
            continue
        conf.append(p[in_bin].mean())
        acc.append(y[in_bin].mean())
        cnt.append(int(in_bin.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


@dataclass
class ConformalRHi:
    """Split-conformal interval half-width for corrected RHi.

    Calibrate on held-out absolute residuals |rhi_true - rhi_hat|; the (1-alpha)
    empirical quantile is a distribution-free half-width q such that the true
    value lies within rhi_hat ± q about (1-alpha) of the time.
    """

    alpha: float = 0.1
    half_width: float = float("nan")

    def fit(self, rhi_true: np.ndarray, rhi_hat: np.ndarray) -> ConformalRHi:
        resid = np.abs(np.asarray(rhi_true, dtype=float) - np.asarray(rhi_hat, dtype=float))
        n = len(resid)
        # finite-sample-adjusted quantile level
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.half_width = float(np.quantile(resid, level))
        return self

    def interval(self, rhi_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(rhi_hat, dtype=float)
        return r - self.half_width, r + self.half_width
