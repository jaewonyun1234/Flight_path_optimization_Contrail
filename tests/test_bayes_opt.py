"""Bayesian optimizer: minimizes a known 2D function, deterministically."""

import numpy as np

from contrail_env import gp_minimize


def quadratic(x):
    return float((x[0] - 0.3) ** 2 + (x[1] + 0.5) ** 2)


def test_minimizes_known_quadratic():
    rng = np.random.default_rng(0)
    result = gp_minimize(
        quadratic, [(-1.0, 1.0), (-1.0, 1.0)], n_calls=30, n_initial=8, rng=rng
    )
    assert result.y_best < 0.05
    assert abs(result.x_best[0] - 0.3) < 0.3
    assert abs(result.x_best[1] + 0.5) < 0.3


def test_deterministic_given_rng():
    r1 = gp_minimize(
        quadratic, [(-1.0, 1.0), (-1.0, 1.0)], n_calls=15,
        rng=np.random.default_rng(42),
    )
    r2 = gp_minimize(
        quadratic, [(-1.0, 1.0), (-1.0, 1.0)], n_calls=15,
        rng=np.random.default_rng(42),
    )
    assert r1.y_best == r2.y_best
    assert np.array_equal(r1.x_best, r2.x_best)
