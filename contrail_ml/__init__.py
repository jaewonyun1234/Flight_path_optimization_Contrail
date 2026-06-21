"""
contrail_ml — Observation-grounded ISSR model + MLOps lifecycle.

This package replaces the synthetic ISSR generator
(`contrail_env.synthetic_issr`) with a machine-learning model that predicts
ice-supersaturated regions (ISSRs) from numerical-weather-prediction (NWP)
fields, bias-corrected against in-situ aircraft humidity (IAGOS) and served
into the existing optimizer through the SAME duck-typed `ISSRField` interface.

DESIGN INVARIANTS (see CONTRAIL_ML_IMPLEMENTATION_PLAN.md §0)
============================================================
1. Never fabricate observational data. The only synthetic data is the
   explicitly-named `synthetic_fallback` used SOLELY for tests/CI.
2. No training/serving skew: feature computation lives in ONE module
   (`contrail_ml.features`), shared by the table builder and the predictor.
3. Don't claim to beat physics until measured: `evaluate` compares against
   raw-ERA5 and quantile-mapping baselines before any "improvement".
4. Keep the deployed solver image lean: all ML deps live in the `[ml]` extra.
5. Don't break the existing system: the synthetic path stays the default.

LIGHT IMPORT SURFACE
====================
Importing `contrail_ml` pulls in only `features`/`config`, which depend on
numpy (+ optional pandas). The heavy estimators (xgboost, scikit-learn,
mlflow) and the geo stack (pycontrails, xarray) are imported lazily inside
the modules that need them, so `import contrail_ml` is cheap and the world
hook in `contrail_env` can import it without dragging in the full extra.
"""

from __future__ import annotations

from . import config, features

__all__ = ["config", "features", "MLIssrField", "ml_issr_field"]


def __getattr__(name: str):
    # Lazy re-export so `from contrail_ml import MLIssrField` works without
    # forcing scipy/sklearn to import at package import time.
    if name in ("MLIssrField", "ml_issr_field"):
        from . import issr_field

        return getattr(issr_field, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
