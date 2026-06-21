"""
monitor.py — Production monitoring with a real ground-truth feedback loop.

This is not a static dashboard (plan §8). As new IAGOS flights arrive, we score
the model that was actually serving against the fresh truth (rolling ETS /
PR-AUC / ECE) AND watch the input distribution drift away from the training
reference (PSI per feature). When skill drops below a floor or drift breaches a
bound, we emit a structured `retrain_recommended` signal — the trigger an
orchestrator would act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .evaluate import score_predictions
from .features import FEATURES

if TYPE_CHECKING:
    from .config import MLConfig


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> float:
    """PSI between a reference and a current sample of one feature.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant.
    Bin edges come from the reference quantiles.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    e_frac = np.clip(e_hist / max(1, e_hist.sum()), 1e-6, None)
    a_frac = np.clip(a_hist / max(1, a_hist.sum()), 1e-6, None)
    return float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, features: tuple[str, ...] = FEATURES
) -> dict[str, float]:
    """PSI per feature between the training reference and a current window."""
    return {
        f: population_stability_index(reference[f].to_numpy(), current[f].to_numpy())
        for f in features
        if f in reference.columns and f in current.columns
    }


@dataclass
class MonitorReport:
    n_new: int
    skill: dict[str, float]
    drift: dict[str, float]
    retrain_recommended: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_new": self.n_new,
            "skill": self.skill,
            "max_drift_psi": max(self.drift.values()) if self.drift else 0.0,
            "drift": self.drift,
            "retrain_recommended": self.retrain_recommended,
            "reasons": self.reasons,
        }


def monitor(
    model: Any,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg: MLConfig,
    *,
    min_ets: float = 0.2,
    max_psi: float = 0.25,
) -> MonitorReport:
    """Score the serving model on fresh labelled data and check feature drift.

    `current` must carry the labels (rhi_iagos, y_issr) so skill is measured
    against truth, plus the FEATURES so the model can predict and drift can be
    computed.
    """
    reasons: list[str] = []

    rhi_hat, _std, p = model.predict(current)
    skill = score_predictions(
        current["rhi_iagos"].to_numpy(), current["y_issr"].to_numpy(), rhi_hat, p,
        p_threshold=cfg.issr_p_threshold,
    )
    drift = feature_drift(reference, current)

    if skill["ets"] < min_ets:
        reasons.append(f"ETS {skill['ets']:.3f} < floor {min_ets}")
    worst = max(drift.items(), key=lambda kv: kv[1], default=(None, 0.0))
    if worst[1] > max_psi:
        reasons.append(f"feature '{worst[0]}' PSI {worst[1]:.3f} > bound {max_psi}")

    return MonitorReport(
        n_new=int(len(current)),
        skill=skill,
        drift=drift,
        retrain_recommended=bool(reasons),
        reasons=reasons,
    )
