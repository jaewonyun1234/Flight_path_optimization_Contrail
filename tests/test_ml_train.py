"""Training pipeline (no MLflow): CV runs, model beats raw ERA5, table is built.

Exercises temporal split, grouped CV, fit+calibrate, and the model-vs-baselines
comparison — the core of the MLOps story — without a tracking server.
"""

import warnings

from contrail_ml.config import MLConfig
from contrail_ml.data.synthetic_fallback import make_synthetic_training_table
from contrail_ml.train import run_training


def test_run_training_builds_comparison_and_beats_raw():
    cfg = MLConfig(n_ensemble=3, xgb_n_estimators=60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = make_synthetic_training_table(n=5000, seed=11, allow_synthetic=True)

    res = run_training(cfg, df, log_mlflow=False, do_cv=True)

    # CV produced metrics.
    assert "rhi_mae" in res.cv_metrics and res.cv_metrics["rhi_mae"] > 0

    # The comparison table has the model row and every baseline.
    table = res.comparison
    for name in ("ml_corrector", "raw_era5", "x_factor", "quantile_map"):
        assert name in table.index

    ml = table.loc["ml_corrector"]
    raw = table.loc["raw_era5"]
    # Bias removed and ISSR skill not worse than the uncorrected forecast.
    assert abs(ml["rhi_bias"]) < abs(raw["rhi_bias"])
    assert ml["ets"] >= raw["ets"]
    assert ml["rhi_mae"] < raw["rhi_mae"]
    # Calibrated probabilities are honest.
    assert ml["ece"] < 0.05
