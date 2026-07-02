"""Instantaneous spectrum of the analog sweep (§S3).

Checked against closed forms (single qubit), against a dense eigendecomposition
(the matrix-free Lanczos path must match np.linalg.eigh), the Omega->0 diagonal
limit, and the qubit-budget guard.
"""

import numpy as np
import pytest

from contrail_env.pasqal_analog import AnnealSchedule, RydbergStatevector
from contrail_env.quantum_common import BackendBudgetError, OptionGraph
from contrail_env.spectral import hamiltonian_matvec, instantaneous_spectrum

_SCHED = AnnealSchedule(T_ns=4000.0, omega_max=10.0, delta_init=-8.0, delta_final=8.0)


def _graph(n, costs, groups, conflict_edges=(), buckets=()):
    costs = np.asarray(costs, dtype=float)
    return OptionGraph(
        n=n, costs=costs, weights=costs.max() - costs + 1.0,
        groups=groups, conflict_edges=conflict_edges, buckets=buckets,
    )


def test_single_qubit_closed_form():
    # One flight, one option -> w_span = 0 -> u_0 = 0.6 exactly; no edges.
    # Then H = (Omega/2) sigma_x - delta u_0 n and Delta = sqrt(Omega^2 + (u delta)^2).
    graph = _graph(1, [3.0], {"F": (0,)})
    s_grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    spec = instantaneous_spectrum(graph, _SCHED, s_grid=s_grid, k=2)
    u = 0.6
    for i, s in enumerate(s_grid):
        om = _SCHED.omega_at(float(s))
        dl = _SCHED.delta_at(float(s))
        expected = np.sqrt(om**2 + (u * dl) ** 2)
        assert spec.gap[i] == pytest.approx(expected, rel=1e-8)


def test_matches_dense_diag():
    # n = 7 (dim 128 > dense threshold) -> Lanczos path; compare lowest 6 to a
    # dense eigh of H built column-by-column via the public matvec. The instance
    # is deliberately asymmetric so the low spectrum is non-degenerate — a single
    # start-vector Lanczos cannot resolve eigenvalue multiplicities, so a
    # "matches dense" test must avoid degeneracies (not a bug: it is why the
    # exact-degeneracy count in S5 exists as an independent cross-check).
    rng = np.random.default_rng(1)
    n = 7
    costs = rng.uniform(1.0, 10.0, n)
    graph = _graph(n, costs, {"A": (0, 1, 2), "B": (3, 4, 5, 6)},
                   conflict_edges=((0, 3), (1, 4)))
    s_grid = np.array([0.15, 0.35, 0.5, 0.65, 0.85])
    spec = instantaneous_spectrum(graph, _SCHED, s_grid=s_grid, k=6)

    sim = RydbergStatevector(graph)
    dim = 1 << n
    eye = np.eye(dim)
    for i, s in enumerate(s_grid):
        om = _SCHED.omega_at(float(s))
        diag = sim.interaction_diag - _SCHED.delta_at(float(s)) * sim.detune_load
        ham = np.column_stack([hamiltonian_matvec(eye[:, j], diag, om, n) for j in range(dim)])
        np.testing.assert_allclose(ham, ham.T, atol=1e-10)   # real symmetric
        dense = np.linalg.eigvalsh(ham)[:6]
        np.testing.assert_allclose(spec.energies[i], dense, atol=1e-8)


def test_omega_zero_limit():
    graph = _graph(3, [2.0, 5.0, 1.0], {"A": (0, 1), "B": (2,)})
    spec = instantaneous_spectrum(graph, _SCHED, s_grid=np.array([0.0, 0.5]), k=4)
    # At s = 0 the drive Omega(0) = 0, so H is diagonal and E_0 = min_z V_0[z].
    assert _SCHED.omega_at(0.0) == 0.0
    sim = RydbergStatevector(graph)
    v0 = sim.interaction_diag - _SCHED.delta_at(0.0) * sim.detune_load
    assert spec.energies[0, 0] == pytest.approx(float(v0.min()), abs=1e-9)


def test_budget_guard():
    big = _graph(17, np.ones(17), {"F": tuple(range(17))})
    with pytest.raises(BackendBudgetError):
        instantaneous_spectrum(big, _SCHED, s_grid=np.array([0.5]), k=4)
    # allow_large only lifts the cap to 20, so 21 still refuses (cheaply).
    huge = _graph(21, np.ones(21), {"F": tuple(range(21))})
    with pytest.raises(BackendBudgetError):
        instantaneous_spectrum(huge, _SCHED, s_grid=np.array([0.5]), k=4, allow_large=True)


def test_gap_curve_single_interior_minimum():
    graph = _graph(6, [2.0, 5.0, 1.0, 4.0, 3.0, 6.0],
                   {"A": (0, 1, 2), "B": (3, 4, 5)}, conflict_edges=((0, 3),))
    spec = instantaneous_spectrum(graph, _SCHED, k=6)
    assert spec.gap.shape == (49,)
    assert 0.0 < spec.s_star < 1.0
    assert spec.delta_min == pytest.approx(spec.gap.min(), abs=1e-12)
    assert spec.end_degeneracy >= 1
