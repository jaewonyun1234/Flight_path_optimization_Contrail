"""Honesty guards (plan §0.1): synthetic data is opt-in and loud; real loaders
fail clearly when data is absent rather than fabricating it."""

import warnings

import pytest

from contrail_ml.config import MLConfig
from contrail_ml.data.synthetic_fallback import make_synthetic_training_table


def test_synthetic_fallback_refuses_without_flag():
    with pytest.raises(RuntimeError, match="allow_synthetic"):
        make_synthetic_training_table(n=100)


def test_synthetic_fallback_warns_loudly_when_allowed():
    with pytest.warns(UserWarning, match="SYNTHETIC"):
        df = make_synthetic_training_table(n=200, allow_synthetic=True)
    assert (df["source"] == "synthetic_fallback").all()


def test_synthetic_met_cube_refuses_without_flag():
    from contrail_ml import predict

    with pytest.raises(RuntimeError, match="allow_synthetic"):
        predict.synthetic_met_cube(MLConfig())


def test_synthetic_ml_issr_field_refuses_without_flag():
    from contrail_ml.issr_field import ml_issr_field

    with pytest.raises(RuntimeError, match="allow_synthetic"):
        ml_issr_field(MLConfig(), met_source="synthetic")


def test_iagos_loader_raises_when_dir_missing():
    from contrail_ml.data.iagos import load_iagos_waypoints

    cfg = MLConfig(iagos_dir=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(FileNotFoundError):
            load_iagos_waypoints(cfg)
