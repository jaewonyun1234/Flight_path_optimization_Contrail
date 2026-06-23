"""contrail_env.analysis: energy landscape, sampler costs, hardness sweep.

Pure-function science behind the research tabs — checked here so the GUI only
has to draw correct numbers.
"""

import numpy as np

from contrail_env.analysis import (
    feasible_cost_landscape,
    gbs_sample_costs,
    hardness_sweep,
    random_sample_costs,
)
from contrail_env.scenario import ScenarioConfig, build_scenario_full
from contrail_env.solver_cpsat import solve_cpsat


def _instance(seed=1, n_flights=3):
    cfg = ScenarioConfig(seed=seed, n_flights=n_flights, beta_contrail=5.0, time_limit_s=5.0)
    _w, _f, evals, conflicts, buckets = build_scenario_full(cfg)
    return evals, conflicts, buckets


def test_landscape_optimum_matches_cpsat():
    evals, conflicts, buckets = _instance()
    costs, optimum = feasible_cost_landscape(evals, conflicts, buckets)
    assert costs.size > 0
    assert optimum == costs.min()
    # The landscape minimum must equal CP-SAT's chosen assignment cost (raw units).
    res = solve_cpsat(evals, conflicts, buckets, time_limit_s=5.0)
    raw_opt = sum(evals[i].cost_combined for i in res.chosen_eval_indices)
    assert abs(optimum - raw_opt) < 1e-6


def test_landscape_refuses_oversized():
    evals, conflicts, buckets = _instance()
    try:
        feasible_cost_landscape(evals, conflicts, buckets, max_combos=1)
    except ValueError as exc:
        assert "exceeds max_combos" in str(exc)
    else:
        raise AssertionError("expected ValueError for an oversized enumeration")


def test_sampler_costs_cannot_beat_the_optimum():
    evals, conflicts, buckets = _instance()
    _costs, optimum = feasible_cost_landscape(evals, conflicts, buckets)

    gbs = gbs_sample_costs(evals, conflicts, buckets, n_samples=120, seed=0)
    rnd = random_sample_costs(evals, conflicts, buckets, n_samples=120, seed=0)

    assert gbs.size == 120 and rnd.size == 120
    # Repaired samples are feasible, so none can be cheaper than the true optimum.
    assert gbs.min() >= optimum - 1e-9
    assert rnd.min() >= optimum - 1e-9


def test_hardness_sweep_shapes_and_effort():
    out = hardness_sweep([2, 3, 4], seed=42, time_limit_s=2.0)
    assert out["sizes"].tolist() == [2, 3, 4]
    for key in ("n_options", "cpsat_ms", "incumbents", "objective"):
        assert out[key].shape == (3,)
    assert np.all(out["cpsat_ms"] >= 0.0)
    assert np.all(out["incumbents"] >= 1)   # CP-SAT always finds at least one incumbent
