"""Scenario generator: determinism + conflict structure."""

import numpy as np

from contrail_env import make_scenario


def test_same_seed_identical_scenario():
    a = make_scenario(4, 3, seed=7)
    b = make_scenario(4, 3, seed=7)
    assert np.array_equal(a.costs, b.costs)
    assert np.array_equal(a.flight_of, b.flight_of)
    assert a.conflicts == b.conflicts


def test_different_seeds_differ():
    a = make_scenario(4, 3, seed=0)
    b = make_scenario(4, 3, seed=1)
    assert not np.array_equal(a.costs, b.costs)


def test_shapes_and_groups():
    s = make_scenario(5, 3, seed=2)
    assert s.n_vars == 15
    assert s.costs.shape == (15,)
    assert s.flight_of.shape == (15,)
    groups = s.groups()
    assert len(groups) == 5
    assert groups[2] == [6, 7, 8]
    assert all(s.flight_of[v] == f for f, g in enumerate(groups) for v in g)


def test_conflicts_only_between_different_flights():
    for seed in range(10):
        s = make_scenario(6, 3, seed=seed)
        for i, j in s.conflicts:
            assert i < j
            assert s.flight_of[i] != s.flight_of[j]


def test_costs_positive():
    s = make_scenario(4, 3, seed=3)
    assert (s.costs > 0).all()
