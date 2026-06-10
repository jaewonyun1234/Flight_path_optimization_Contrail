"""Quantum pipelines tested against the independent brute-force oracle.

Same philosophy as test_solver_cpsat.py: `enumerate_optimum` (plain
exhaustive search) provides the ground truth, so a quantum solver matching
it is real evidence the embedding + sampling + repair chain is correct.

Everything here runs on the dependency-free backends (statevector for
Pasqal, exact MH hafnian sampler for Xanadu) with fixed seeds, so the
assertions are deterministic.
"""

import numpy as np
import pytest

from contrail_env import (
    build_and_evaluate_flight,
    build_capacity_buckets,
    build_conflict_graph,
    build_option_graph,
    build_random_flights,
    default_european_world,
    encode_option_graph,
    enumerate_optimum,
    hafnian,
    penalized_energy,
    repair_sample,
    sample_is_feasible,
    solve_pasqal_analog,
    solve_xanadu_gbs,
    takagi_symmetric,
)

# CP-SAT rounds costs to the cent, so "matches the optimum" means within
# one cent of slack per option, not bit-exact float equality.
_COST_TOL = 1e-2


@pytest.fixture(scope="module")
def instance():
    """Small instance with real conflicts (same recipe as the CP-SAT test)."""
    world = default_european_world(seed=1, n_issr_blobs=8)
    flights = build_random_flights(
        n_flights=3,
        world=world,
        seed=1,
        corridor_frac=0.04,
        snapshot_window_s=(0.0, 300.0),
    )
    evals = []
    for flight in flights:
        evals.extend(build_and_evaluate_flight(flight, world))
    conflicts = build_conflict_graph(evals, world)
    buckets = build_capacity_buckets(evals, world)
    assert len(conflicts) > 0, "instance must be non-trivial"
    return evals, conflicts, buckets


@pytest.fixture(scope="module")
def oracle_optimum(instance):
    evals, conflicts, buckets = instance
    _chosen, objective = enumerate_optimum(evals, conflicts, buckets)
    return objective


def _assert_assignment_feasible(indices, evals, conflicts, buckets):
    selected = set(indices)
    flights = {evals[i].flight_name for i in indices}
    assert len(indices) == len(flights), "one option per flight"
    for e in conflicts:
        assert not (e.i in selected and e.j in selected)
    for b in buckets:
        assert sum(1 for m in b.members if m in selected) <= b.capacity


# =============================================================================
# PASQAL ANALOG
# =============================================================================

def test_pasqal_reaches_oracle_optimum(instance, oracle_optimum):
    evals, conflicts, buckets = instance
    result = solve_pasqal_analog(
        evals, conflicts, buckets, n_shots=400, bo_iters=8, seed=0
    )
    assert result.backend == "statevector"
    _assert_assignment_feasible(result.chosen_eval_indices, evals, conflicts, buckets)
    assert result.best_cost <= oracle_optimum + _COST_TOL
    # The blockade dynamics should embed the constraints well enough that a
    # solid fraction of RAW samples is already feasible.
    assert result.feasibility_rate > 0.2
    assert len(result.history) >= 8
    # History is the running best -> monotone non-increasing.
    costs = [c for _step, c in result.history]
    assert all(b <= a + 1e-9 for a, b in zip(costs, costs[1:], strict=False))


def test_pasqal_budget_error():
    from contrail_env.pasqal_analog import MAX_STATEVECTOR_QUBITS, RydbergStatevector
    from contrail_env.quantum_common import BackendBudgetError, OptionGraph

    n = MAX_STATEVECTOR_QUBITS + 1
    graph = OptionGraph(
        n=n,
        costs=np.ones(n),
        weights=np.ones(n),
        groups={"F0": tuple(range(n))},
        conflict_edges=(),
        buckets=(),
    )
    with pytest.raises(BackendBudgetError):
        RydbergStatevector(graph)


# =============================================================================
# XANADU GBS
# =============================================================================

def test_xanadu_reaches_oracle_optimum(instance, oracle_optimum):
    evals, conflicts, buckets = instance
    result = solve_xanadu_gbs(
        evals, conflicts, buckets, n_samples=400, seed=0
    )
    assert result.backend == "mh-exact"
    _assert_assignment_feasible(result.chosen_eval_indices, evals, conflicts, buckets)
    assert result.best_cost <= oracle_optimum + _COST_TOL
    assert result.n_samples == 400


def test_hafnian_known_values():
    # Haf of the empty matrix is 1; odd dimension is 0.
    assert hafnian(np.zeros((0, 0))) == 1.0
    assert hafnian(np.ones((3, 3))) == 0.0
    # 2x2: the single pairing -> A[0, 1].
    a = np.array([[0.0, 0.7], [0.7, 0.0]])
    assert hafnian(a) == pytest.approx(0.7)
    # All-ones 4x4 / 6x6: number of perfect matchings of K_n = (n-1)!!.
    assert hafnian(np.ones((4, 4))) == pytest.approx(3.0)
    assert hafnian(np.ones((6, 6))) == pytest.approx(15.0)


def test_takagi_reconstructs_symmetric_matrix():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(8, 8))
    a = 0.5 * (a + a.T)
    lam, u = takagi_symmetric(a)
    assert np.all(lam >= 0)
    np.testing.assert_allclose(u @ np.conj(u).T, np.eye(8), atol=1e-10)  # unitary
    np.testing.assert_allclose(u @ np.diag(lam) @ u.T, a, atol=1e-10)


def test_gbs_encoding_is_physical(instance):
    evals, conflicts, buckets = instance
    graph = build_option_graph(evals, conflicts, buckets)
    enc = encode_option_graph(graph)
    # Spectral norm strictly < 1 (finite squeezing), and the complement
    # encoding must put ZERO coupling on independence edges.
    assert float(np.max(np.abs(np.linalg.eigvalsh(enc.B)))) < 1.0
    for i, j in graph.independence_edges():
        assert enc.B[i, j] == 0.0
    assert np.all(np.isfinite(enc.squeezing))
    assert enc.mean_photons > 0


# =============================================================================
# SHARED CORE — repair + feasibility + penalized energy
# =============================================================================

def test_repair_restores_one_hot_and_conflicts(instance):
    evals, conflicts, buckets = instance
    graph = build_option_graph(evals, conflicts, buckets)
    rng = np.random.default_rng(7)
    for _ in range(50):
        bits = (rng.random(graph.n) < 0.4).astype(np.uint8)
        indices = repair_sample(graph, bits)
        _assert_assignment_feasible(indices, evals, conflicts, buckets)


def test_feasibility_check_agrees_with_penalty(instance):
    evals, conflicts, buckets = instance
    graph = build_option_graph(evals, conflicts, buckets)
    rng = np.random.default_rng(11)
    for _ in range(50):
        bits = (rng.random(graph.n) < 0.3).astype(np.uint8)
        plain_cost = float(np.dot(graph.costs, bits))
        if sample_is_feasible(graph, bits):
            # Feasible -> penalty terms vanish, energy is the plain cost.
            assert penalized_energy(graph, bits) == pytest.approx(plain_cost)
        else:
            # Infeasible -> at least one penalty (> cost span) was added.
            assert penalized_energy(graph, bits) > plain_cost
