"""
evaluate.py — Metrics, baseline comparison, and diagnostic plots.

This is where the project earns the right to claim anything (plan §0.3, §6).
Nothing here trains; it scores predictions against the IAGOS truth on the
held-out set and tabulates the model next to the raw-ERA5 and quantile-mapping
baselines. The headline numbers:

  * RHi dry-bias reduction (mean error vs IAGOS) and RMSE/MAE,
  * ISSR skill: ETS, F1, ROC-AUC, PR-AUC,
  * probability honesty: Brier score and Expected Calibration Error.

Report whatever the numbers are — modest gains, reported rigorously with
calibrated probabilities, are themselves the contribution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .calibrate import expected_calibration_error, reliability_curve

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def equitable_threat_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ETS (Gilbert skill score) for a binary forecast. Chance-corrected hit
    rate; 0 = no skill, 1 = perfect. The plan's bar is beating raw ERA5's
    ~0.2-0.4."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    hits = int(np.sum((yp == 1) & (yt == 1)))
    false_alarms = int(np.sum((yp == 1) & (yt == 0)))
    misses = int(np.sum((yp == 0) & (yt == 1)))
    n = len(yt)
    pred_pos = hits + false_alarms
    obs_pos = hits + misses
    hits_random = pred_pos * obs_pos / n if n else 0.0
    denom = hits + false_alarms + misses - hits_random
    return float((hits - hits_random) / denom) if denom != 0 else 0.0


def score_predictions(
    rhi_true: np.ndarray,
    y_true: np.ndarray,
    rhi_hat: np.ndarray,
    p_issr: np.ndarray,
    *,
    p_threshold: float = 0.5,
) -> dict[str, float]:
    """All metrics for one predictor, as a flat dict."""
    rhi_true = np.asarray(rhi_true, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    rhi_hat = np.asarray(rhi_hat, dtype=float)
    p_issr = np.asarray(p_issr, dtype=float)
    y_pred = (p_issr >= p_threshold).astype(int)

    err = rhi_hat - rhi_true
    out: dict[str, float] = {
        "rhi_rmse": float(np.sqrt(np.mean(err ** 2))),
        "rhi_mae": float(np.mean(np.abs(err))),
        "rhi_bias": float(np.mean(err)),
        "ets": equitable_threat_score(y_true, y_pred),
    }

    # sklearn metrics; guard the single-class case (AUC undefined).
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, p_issr))
        out["pr_auc"] = float(average_precision_score(y_true, p_issr))
        out["brier"] = float(brier_score_loss(y_true, np.clip(p_issr, 0, 1)))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
        out["brier"] = float("nan")
    out["ece"] = expected_calibration_error(np.clip(p_issr, 0, 1), y_true)
    return out


def comparison_table(
    holdout: pd.DataFrame,
    model: Any,
    baselines: list,
    *,
    p_threshold: float = 0.5,
) -> pd.DataFrame:
    """One row per predictor (model + each baseline), one column per metric.

    `model` is an RHiCorrector-like object (.predict -> rhi_hat, rhi_std,
    p_issr); each baseline exposes predict_rhi/predict_proba. The baselines must
    already be fitted on the training split.
    """
    rhi_true = holdout["rhi_iagos"].to_numpy(dtype=float)
    y_true = holdout["y_issr"].to_numpy(dtype=int)

    rows: dict[str, dict[str, float]] = {}

    rhi_hat, _std, p_issr = model.predict(holdout)
    rows["ml_corrector"] = score_predictions(
        rhi_true, y_true, rhi_hat, p_issr, p_threshold=p_threshold
    )
    for b in baselines:
        rows[b.name] = score_predictions(
            rhi_true, y_true, b.predict_rhi(holdout), b.predict_proba(holdout),
            p_threshold=p_threshold,
        )

    table = pd.DataFrame(rows).T
    table.index.name = "predictor"
    return table


# ===========================================================================
# Diagnostic plots (logged to MLflow as artifacts)
# ===========================================================================


def plot_reliability(p_issr: np.ndarray, y_true: np.ndarray) -> Figure:
    """Reliability diagram (calibration curve) with the y=x ideal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf, acc, _cnt = reliability_curve(np.clip(p_issr, 0, 1), y_true)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(conf, acc, "o-", label="model")
    ax.set_xlabel("predicted P(ISSR)")
    ax.set_ylabel("observed ISSR frequency")
    ax.set_title("Reliability")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_rhi_scatter(rhi_true: np.ndarray, rhi_hat: np.ndarray,
                     rhi_raw: np.ndarray) -> Figure:
    """Corrected vs raw RHi against IAGOS truth — visualises the bias removal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    lim = (0.0, max(160.0, float(np.max(rhi_true)) + 5))
    ax.plot(lim, lim, "k--", lw=1)
    ax.scatter(rhi_true, rhi_raw, s=4, alpha=0.3, label="raw NWP")
    ax.scatter(rhi_true, rhi_hat, s=4, alpha=0.3, label="corrected")
    ax.axvline(100, color="grey", lw=0.7)
    ax.axhline(100, color="grey", lw=0.7)
    ax.set_xlabel("IAGOS RHi (%)")
    ax.set_ylabel("predicted RHi (%)")
    ax.set_title("RHi: raw vs corrected")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend()
    fig.tight_layout()
    return fig
