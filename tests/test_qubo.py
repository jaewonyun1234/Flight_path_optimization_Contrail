"""QUBO assembly: energy identity, penalty bound, zero-penalty optimum."""

import numpy as np

from contrail_env import (
    assemble_qubo,
    brute_force_optimum,
    cost_of_assignment,
    is_feasible,
    make_scenario,
)


def hand_energy(z, scenario, qubo):
    """cost + A*onehot + B*conflict, written out the slow, obvious way."""
    e = float(np.dot(scenario.costs, z))
    for group in scenario.groups():
        e += qubo.penalty_A * (sum(int(z[v]) for v in group) - 1) ** 2
    for i, j in scenario.conflicts:
        e += qubo.penalty_B * int(z[i]) * int(z[j])
    return e


def test_energy_matches_hand_computation():
    scenario = make_scenario(4, 3, seed=5)
    qubo = assemble_qubo(scenario)
    rng = np.random.default_rng(0)
    for _ in range(50):
        z = (rng.random(scenario.n_vars) < 0.5).astype(int)
        assert abs(qubo.energy(z) - hand_energy(z, scenario, qubo)) < 1e-9


def test_penalty_bound_holds():
    scenario = make_scenario(4, 3, seed=1)
    qubo = assemble_qubo(scenario)
    spread = float(scenario.costs.max() - scenario.costs.min())
    assert qubo.penalty_A > spread
    assert qubo.penalty_B > spread


def test_feasible_optimum_has_zero_penalty_energy():
    scenario = make_scenario(3, 3, seed=2)
    qubo = assemble_qubo(scenario)
    z_opt, e_min = brute_force_optimum(qubo)
    feasible, violations = is_feasible(z_opt, qubo)
    assert feasible, violations
    # At a feasible z every penalty term is zero, so energy == raw cost.
    assert abs(e_min - cost_of_assignment(z_opt, qubo)) < 1e-9
