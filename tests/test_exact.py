"""Brute force vs an exhaustive Python loop; repair always feasible."""

import numpy as np

from contrail_env import (
    approximation_ratio,
    assemble_qubo,
    brute_force_optimum,
    is_feasible,
    make_scenario,
    raw_metrics,
    repair,
    solve_greedy,
    solve_random,
)


def test_brute_force_matches_python_loop():
    scenario = make_scenario(2, 3, seed=4)  # 6 variables -> 64 states
    qubo = assemble_qubo(scenario)
    best_e, best_z = np.inf, None
    for k in range(2 ** scenario.n_vars):
        z = np.array([(k >> b) & 1 for b in range(scenario.n_vars)], dtype=int)
        e = qubo.energy(z)
        if e < best_e:
            best_e, best_z = e, z
    z_opt, e_min = brute_force_optimum(qubo)
    assert abs(e_min - best_e) < 1e-9
    assert np.array_equal(z_opt, best_z)


def test_repair_always_feasible():
    rng = np.random.default_rng(0)
    for seed in range(5):
        scenario = make_scenario(4, 3, seed=seed)
        qubo = assemble_qubo(scenario)
        for _ in range(30):
            bits = (rng.random(scenario.n_vars) < 0.5).astype(np.uint8)
            z = repair(bits, scenario)
            feasible, violations = is_feasible(z, qubo)
            assert feasible, violations


def test_solve_random_deterministic_and_bounded():
    scenario = make_scenario(3, 3, seed=1)
    qubo = assemble_qubo(scenario)
    _z_opt, e_min = brute_force_optimum(qubo)
    a = solve_random(scenario, qubo, n_samples=200, seed=0)
    b = solve_random(scenario, qubo, n_samples=200, seed=0)
    assert a.best_cost == b.best_cost
    assert a.best_cost >= e_min - 1e-9  # nothing beats the exact optimum
    assert 0.0 <= a.feasibility_rate <= 1.0


def test_approximation_ratio_edges():
    assert approximation_ratio(100.0, 100.0, 150.0) == 1.0   # optimal
    assert approximation_ratio(150.0, 100.0, 150.0) == 0.0   # random-level
    assert approximation_ratio(100.0, 100.0, 100.0) == 1.0   # zero gap, NaN-safe
    assert approximation_ratio(101.0, 100.0, 100.0) == 0.0


def test_solve_greedy_feasible_and_deterministic():
    for seed in range(5):
        scenario = make_scenario(4, 3, seed=seed)
        qubo = assemble_qubo(scenario)
        a = solve_greedy(scenario, qubo)
        b = solve_greedy(scenario, qubo)
        feasible, violations = is_feasible(a.best_z, qubo)
        assert feasible, violations
        assert a.best_cost == b.best_cost
        assert np.array_equal(a.best_z, b.best_z)


def test_raw_metrics_hand_computed():
    scenario = make_scenario(3, 3, seed=1)
    qubo = assemble_qubo(scenario)
    rng = np.random.default_rng(0)
    samples = (rng.random((8, scenario.n_vars)) < 0.5).astype(np.uint8)
    energies = [qubo.energy(s) for s in samples]
    feasible = [is_feasible(s, qubo)[0] for s in samples]
    best, mean, feas = raw_metrics(samples, qubo)
    assert abs(best - min(energies)) < 1e-9
    assert abs(mean - float(np.mean(energies))) < 1e-9
    assert abs(feas - float(np.mean(feasible))) < 1e-9


def test_random_raw_mean_exceeds_greedy_cost():
    # Sanity direction on a nontrivial instance: unrepaired noise carries
    # heavy penalty energy, far above the greedy heuristic's feasible cost.
    scenario = make_scenario(4, 3, seed=3)
    assert len(scenario.conflicts) > 0
    qubo = assemble_qubo(scenario)
    rand = solve_random(scenario, qubo, n_samples=300, seed=0)
    greedy = solve_greedy(scenario, qubo)
    assert rand.raw_mean_E is not None
    assert rand.raw_mean_E > greedy.best_cost
