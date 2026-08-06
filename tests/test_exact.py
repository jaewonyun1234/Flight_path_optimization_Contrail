"""Brute force vs an exhaustive Python loop; repair always feasible."""

import numpy as np

from contrail_env import (
    approximation_ratio,
    assemble_qubo,
    brute_force_optimum,
    is_feasible,
    make_scenario,
    repair,
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
