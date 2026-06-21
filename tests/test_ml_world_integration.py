"""The seam: World(issr_source="ml") -> qubo -> CP-SAT, with no qubo/World edits.

Mirrors tests/test_solver_cpsat.py but with the ML ISSR field swapped in via the
duck-typed interface. The whole point of the project: the trained model drops
into the existing optimizer untouched.
"""

import warnings

import pytest

# Needs the [ml] extra (pandas/scipy/scikit-learn/xgboost). The core lint-type-test
# CI job installs only [dev], so skip there; the dedicated `ml` job runs these fully.
pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")

from contrail_env import (
    build_and_evaluate_flight,
    build_capacity_buckets,
    build_conflict_graph,
    build_random_flights,
    default_european_world,
    solve_cpsat,
)
from contrail_ml.config import MLConfig
from contrail_ml.issr_field import MLIssrField


def _ml_world():
    cfg = MLConfig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return default_european_world(
            seed=1,
            issr_source="ml",
            issr_kwargs=dict(config=cfg, met_source="synthetic",
                             allow_synthetic=True, grid_res_deg=3.0, seed=1),
        )


def test_ml_world_builds_with_ml_field():
    world = _ml_world()
    assert isinstance(world.issr, MLIssrField)


def test_ml_world_solves_feasibly():
    world = _ml_world()
    flights = build_random_flights(n_flights=4, world=world, seed=1,
                                   corridor_frac=0.05, snapshot_window_s=(0.0, 300.0))
    evals = []
    for f in flights:
        evals.extend(build_and_evaluate_flight(f, world))
    conflicts = build_conflict_graph(evals, world)
    buckets = build_capacity_buckets(evals, world)

    result = solve_cpsat(evals, conflicts, buckets, time_limit_s=10.0)
    selected = set(result.chosen_eval_indices)

    # One option per flight.
    assert len({evals[i].flight_name for i in selected}) == 4
    # No conflict edge fully chosen.
    for e in conflicts:
        assert not (e.i in selected and e.j in selected)
    # No capacity bucket exceeded.
    for b in buckets:
        assert sum(1 for m in b.members if m in selected) <= b.capacity
