"""Analog pipeline smoke tests: unitarity + beats-or-matches random."""

import numpy as np

from contrail_env import (
    AnnealSchedule,
    RydbergStatevector,
    assemble_qubo,
    independence_edges,
    make_scenario,
    node_weights,
    solve_pasqal_analog,
    solve_random,
)


def _tiny_sim(seed=0):
    scenario = make_scenario(2, 2, seed=seed)  # 4 qubits
    return scenario, RydbergStatevector(
        scenario.n_vars, node_weights(scenario.costs), independence_edges(scenario)
    )


def test_probabilities_normalized():
    _, sim = _tiny_sim()
    schedule = AnnealSchedule(T_ns=2000.0, omega_max=8.0, delta_init=-6.0, delta_final=6.0)
    probs = sim.run(schedule, n_steps=100)
    assert abs(float(probs.sum()) - 1.0) < 1e-9
    assert (probs >= 0).all()


def test_zero_drive_conserves_ground_state():
    # With Omega = 0 the Hamiltonian is diagonal: |0000> only picks up
    # phase, so all probability must stay exactly there (norm conservation).
    _, sim = _tiny_sim()
    schedule = AnnealSchedule(T_ns=2000.0, omega_max=0.0, delta_init=-6.0, delta_final=6.0)
    probs = sim.run(schedule, n_steps=100)
    assert abs(float(probs[0]) - 1.0) < 1e-9


def test_run_deterministic():
    _, sim = _tiny_sim()
    schedule = AnnealSchedule(T_ns=2500.0, omega_max=6.0, delta_init=-8.0, delta_final=8.0)
    p1 = sim.run(schedule, n_steps=120)
    p2 = sim.run(schedule, n_steps=120)
    assert np.array_equal(p1, p2)


def test_analog_beats_or_matches_random():
    scenario = make_scenario(2, 2, seed=3)
    qubo = assemble_qubo(scenario)
    rand = solve_random(scenario, qubo, n_samples=200, seed=0)
    pasqal = solve_pasqal_analog(
        scenario, qubo, n_shots=200, bo_iters=4, n_steps=100, seed=0,
        backend="statevector",
    )
    assert pasqal.best_cost <= rand.best_cost + 1e-9
    assert 0.0 <= pasqal.feasibility_rate <= 1.0
