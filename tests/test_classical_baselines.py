"""Classical sampling baselines: simulated annealing + uniform random repair.

The fair classical *samplers* the benchmark compares the quantum pipelines
against at matched output budget. SA must nail small instances (the repair +
annealing should always reach the exact optimum on a 3-flight problem), and
both must be deterministic under a fixed seed.
"""

from contrail_env.analysis import feasible_cost_landscape
from contrail_env.classical_baselines import (
    solve_random_repair,
    solve_simulated_annealing,
)
from contrail_env.scenario import ScenarioConfig, build_scenario_full


def _instance(seed=1, n_flights=3):
    cfg = ScenarioConfig(seed=seed, n_flights=n_flights, beta_contrail=5.0, time_limit_s=5.0)
    _w, _f, evals, conflicts, buckets = build_scenario_full(cfg)
    return evals, conflicts, buckets


def test_sa_finds_optimum_small():
    evals, conflicts, buckets = _instance()
    _costs, optimum = feasible_cost_landscape(evals, conflicts, buckets)
    res = solve_simulated_annealing(evals, conflicts, buckets, n_samples=200, n_sweeps=64, seed=0)
    assert res.solver == "sa"
    assert res.best_cost <= optimum + 1e-6


def test_sa_deterministic():
    evals, conflicts, buckets = _instance()
    a = solve_simulated_annealing(evals, conflicts, buckets, n_samples=80, seed=3)
    b = solve_simulated_annealing(evals, conflicts, buckets, n_samples=80, seed=3)
    assert a.best_cost == b.best_cost
    assert a.feasibility_rate == b.feasibility_rate
    assert a.history == b.history


def test_random_repair_bounds():
    evals, conflicts, buckets = _instance()
    _costs, optimum = feasible_cost_landscape(evals, conflicts, buckets)
    res = solve_random_repair(evals, conflicts, buckets, n_samples=300, seed=0)
    assert res.solver == "random-repair"
    assert 0.0 <= res.feasibility_rate <= 1.0
    # Repaired samples are always feasible, so none can beat the true optimum.
    assert res.best_cost >= optimum - 1e-9
    assert res.n_samples == 300


def test_sa_delta_consistency():
    # The __debug__ assertion inside the SA loop checks the incremental ΔE
    # against a from-scratch penalized energy — a seeded run must pass it.
    assert __debug__, "run pytest without -O so the ΔE self-check is active"
    evals, conflicts, buckets = _instance()
    res = solve_simulated_annealing(evals, conflicts, buckets, n_samples=40, seed=7)
    assert res.meta["energy_evaluations"] == 40 * res.meta["n_sweeps"] * len(evals)
