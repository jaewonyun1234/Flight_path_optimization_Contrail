"""
model.py — RHiCorrector: a regime-split, uncertainty-aware bias corrector.

THE IDEA (plan §5)
==================
Weather models are dry-biased near the tropopause, and the bias behaves
differently in dry vs humid air. So we split on the NWP relative-humidity-over-
ice at 85 % and fit two sub-models:

    dry regime  (rhi < 85):  XGBoost regressor   — fast, tabular.
    humid regime (rhi >= 85): MLP regressor       — where the ISSR signal and
                                                    the nonlinearity live.

We predict the RESIDUAL  delta = rhi_iagos - rhi_nwp  (the correction), not the
absolute RHi — easier and more stable. Corrected RHi = rhi_nwp + delta_hat.

A separate XGBoost CLASSIFIER head gives P(ISSR), which `calibrate.py` then
calibrates. A bootstrap ENSEMBLE of the regressor gives a predictive spread
(rhi_std), so downstream code knows where the model is unsure.

PHYSICS-INFORMED, THE LEGITIMATE WAY
====================================
The thermodynamics enter as engineered features (`rhi`, computed from T/q/p by
the shared features module), the regime split is a physical boundary, and
outputs are clamped to thermodynamic plausibility. There is no PDE-residual
loss — there is no governing PDE for this threshold problem, so a PINN would be
cargo-culting.

The class exposes a scikit-learn-style surface: fit / predict / save / load
plus predict_proba_issr, so it slots into the training and serving code without
special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .features import FEATURES, feature_matrix

if TYPE_CHECKING:
    import pandas as pd

    from .config import MLConfig

# RHi is clamped to this range on output: 0 % (bone dry) up to a generous
# supersaturation ceiling. Real upper-tropospheric RHi rarely exceeds ~160 %.
_RHI_MIN, _RHI_MAX = 0.0, 180.0


@dataclass
class RHiCorrector:
    """Regime-split residual corrector with a calibrated-ready ISSR head.

    Hyperparameters default to small, CPU-friendly values; `from_config`
    pulls the production values from an `MLConfig`.
    """

    regime_split_rhi: float = 85.0
    issr_rhi_threshold: float = 100.0
    n_ensemble: int = 10
    xgb_max_depth: int = 6
    xgb_n_estimators: int = 300
    xgb_learning_rate: float = 0.05
    mlp_hidden: tuple[int, ...] = (64, 32)
    random_state: int = 42
    min_regime_samples: int = 50  # below this, fall back to one combined model

    # learned state (populated by fit)
    _ensemble: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _classifier: Any = field(default=None, repr=False)
    _fitted: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: MLConfig) -> RHiCorrector:
        return cls(
            regime_split_rhi=cfg.regime_split_rhi,
            issr_rhi_threshold=cfg.issr_rhi_threshold,
            n_ensemble=cfg.n_ensemble,
            xgb_max_depth=cfg.xgb_max_depth,
            xgb_n_estimators=cfg.xgb_n_estimators,
            xgb_learning_rate=cfg.xgb_learning_rate,
            mlp_hidden=cfg.mlp_hidden,
            random_state=cfg.random_state,
        )

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> RHiCorrector:
        """Fit the corrector.

        X : DataFrame containing the FEATURES columns (must include `rhi`).
        y : the residual target delta = rhi_iagos - rhi_nwp (per row).
        sample_weight : optional per-row weights.

        The ISSR classification label is derived internally as
        (rhi_nwp + delta >= issr_rhi_threshold), so the caller passes only the
        residual — one target, sklearn-style.
        """
        Xmat = feature_matrix(X).to_numpy(dtype=float)
        rhi_nwp = X["rhi"].to_numpy(dtype=float)
        delta = np.asarray(y, dtype=float)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

        # Derived classification label from the truth = nwp + residual.
        y_issr = (rhi_nwp + delta >= self.issr_rhi_threshold).astype(int)

        rng = np.random.default_rng(self.random_state)
        n = len(Xmat)

        # Bootstrap ensemble of the regime-split regressor.
        self._ensemble = []
        for b in range(self.n_ensemble):
            if b == 0:
                idx = np.arange(n)  # first member = full-data fit (stable mean)
            else:
                idx = rng.integers(0, n, size=n)
            member = self._fit_regime_regressor(
                Xmat[idx], rhi_nwp[idx], delta[idx],
                None if w is None else w[idx], seed=self.random_state + b,
            )
            self._ensemble.append(member)

        # Single classification head (calibrated later, externally).
        self._classifier = self._fit_classifier(Xmat, y_issr, w)
        self._fitted = True
        return self

    def _fit_regime_regressor(self, Xmat, rhi_nwp, delta, w, seed) -> dict[str, Any]:
        """Fit dry + humid residual sub-models, with a combined fallback when a
        regime is too sparse to fit on its own."""
        from sklearn.neural_network import MLPRegressor
        from xgboost import XGBRegressor

        humid = rhi_nwp >= self.regime_split_rhi
        dry = ~humid

        def _xgb() -> Any:
            return XGBRegressor(
                max_depth=self.xgb_max_depth,
                n_estimators=self.xgb_n_estimators,
                learning_rate=self.xgb_learning_rate,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            )

        def _mlp() -> Any:
            return MLPRegressor(
                hidden_layer_sizes=tuple(self.mlp_hidden),
                max_iter=500,
                random_state=seed,
                early_stopping=False,
            )

        if dry.sum() < self.min_regime_samples or humid.sum() < self.min_regime_samples:
            # Too few samples in some regime: one combined XGBoost model.
            combined = _xgb()
            _xgb_fit(combined, Xmat, delta, w)
            return {"mode": "combined", "model": combined}

        dry_model = _xgb()
        _xgb_fit(dry_model, Xmat[dry], delta[dry], None if w is None else w[dry])
        humid_model = _mlp()
        # MLPRegressor has no sample_weight; weights apply to XGB only.
        humid_model.fit(Xmat[humid], delta[humid])
        return {"mode": "split", "dry": dry_model, "humid": humid_model}

    def _fit_classifier(self, Xmat, y_issr, w) -> Any:
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            max_depth=self.xgb_max_depth,
            n_estimators=self.xgb_n_estimators,
            learning_rate=self.xgb_learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.random_state,
            n_jobs=1,
            verbosity=0,
            eval_metric="logloss",
        )
        # Degenerate label (all one class) — fall back to a constant predictor.
        if len(np.unique(y_issr)) < 2:
            return _ConstantClassifier(float(y_issr.mean()))
        if w is None:
            clf.fit(Xmat, y_issr)
        else:
            clf.fit(Xmat, y_issr, sample_weight=w)
        return clf

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    def _member_delta(self, member: dict[str, Any], Xmat, rhi_nwp) -> np.ndarray:
        if member["mode"] == "combined":
            return member["model"].predict(Xmat)
        humid = rhi_nwp >= self.regime_split_rhi
        out = np.empty(len(Xmat), dtype=float)
        if (~humid).any():
            out[~humid] = member["dry"].predict(Xmat[~humid])
        if humid.any():
            out[humid] = member["humid"].predict(Xmat[humid])
        return out

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (rhi_hat, rhi_std, p_issr).

        rhi_hat : corrected RHi (%), ensemble mean, clamped to plausibility.
        rhi_std : ensemble spread on corrected RHi (predictive uncertainty).
        p_issr  : raw P(ISSR) from the classifier head (calibrate externally).
        """
        self._check_fitted()
        Xmat = feature_matrix(X).to_numpy(dtype=float)
        rhi_nwp = X["rhi"].to_numpy(dtype=float)

        members = np.stack(
            [rhi_nwp + self._member_delta(m, Xmat, rhi_nwp) for m in self._ensemble]
        )
        members = np.clip(members, _RHI_MIN, _RHI_MAX)
        rhi_hat = members.mean(axis=0)
        rhi_std = members.std(axis=0)
        p_issr = self._classifier.predict_proba(Xmat)[:, 1]
        return rhi_hat, rhi_std, p_issr

    def predict_corrected_rhi(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X)[0]

    def predict_proba_issr(self, X: pd.DataFrame) -> np.ndarray:
        """Raw P(ISSR) in [0, 1] (one column)."""
        self._check_fitted()
        Xmat = feature_matrix(X).to_numpy(dtype=float)
        return self._classifier.predict_proba(Xmat)[:, 1]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> RHiCorrector:
        import joblib

        obj = joblib.load(path)
        if not isinstance(obj, RHiCorrector):
            raise TypeError(f"{path} did not contain an RHiCorrector")
        return obj

    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("RHiCorrector.predict called before fit().")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURES


def _xgb_fit(model, X, y, w) -> None:
    if w is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=w)


@dataclass
class _ConstantClassifier:
    """Fallback when the training labels are single-class. Mimics the slice of
    the scikit-learn API the corrector uses (`predict_proba[:, 1]`)."""

    p: float

    def predict_proba(self, X) -> np.ndarray:
        n = len(X)
        col1 = np.full(n, self.p)
        return np.column_stack([1.0 - col1, col1])
