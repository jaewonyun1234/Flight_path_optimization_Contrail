"""RHiCorrector: fits, predicts, routes regimes, and reduces the dry bias.

All hermetic — trains on the guarded synthetic fallback, never on real data.
"""

import warnings

import numpy as np
import pytest

# Needs the [ml] extra (pandas/scikit-learn/xgboost). The core lint-type-test CI
# job installs only [dev], so skip there; the dedicated `ml` job runs these fully.
pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")

from contrail_ml.data.schema import LABEL_DELTA
from contrail_ml.data.synthetic_fallback import make_synthetic_training_table
from contrail_ml.model import RHiCorrector


@pytest.fixture(scope="module")
def table():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_synthetic_training_table(n=4000, seed=7, allow_synthetic=True)


def _small_model():
    return RHiCorrector(n_ensemble=3, xgb_n_estimators=60, min_regime_samples=30,
                        regime_split_rhi=85.0, random_state=0)


def test_fit_predict_shapes_and_ranges(table):
    train, test = table.iloc[:3000], table.iloc[3000:]
    m = _small_model().fit(train, train[LABEL_DELTA].to_numpy())
    rhi_hat, rhi_std, p_issr = m.predict(test)

    n = len(test)
    assert rhi_hat.shape == (n,) and rhi_std.shape == (n,) and p_issr.shape == (n,)
    assert np.all(p_issr >= 0.0) and np.all(p_issr <= 1.0)
    assert np.all(rhi_std >= 0.0) and rhi_std.max() > 0.0  # ensemble has spread


def test_model_reduces_dry_bias(table):
    train, test = table.iloc[:3000], table.iloc[3000:]
    m = _small_model().fit(train, train[LABEL_DELTA].to_numpy())
    rhi_hat, _s, _p = m.predict(test)

    truth = test["rhi_iagos"].to_numpy()
    raw_mae = np.abs(test["rhi"].to_numpy() - truth).mean()
    model_mae = np.abs(rhi_hat - truth).mean()
    assert model_mae < raw_mae  # correcting helps


def test_predict_before_fit_raises(table):
    with pytest.raises(RuntimeError):
        _small_model().predict(table)


def test_predict_proba_issr_in_unit_interval(table):
    m = _small_model().fit(table, table[LABEL_DELTA].to_numpy())
    p = m.predict_proba_issr(table)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_combined_fallback_when_regime_sparse(table):
    # A very high min_regime_samples forces the combined-model branch.
    m = RHiCorrector(n_ensemble=2, xgb_n_estimators=40, min_regime_samples=10_000)
    m.fit(table, table[LABEL_DELTA].to_numpy())
    rhi_hat, _s, _p = m.predict(table)
    assert rhi_hat.shape[0] == len(table)
    assert all(member["mode"] == "combined" for member in m._ensemble)
