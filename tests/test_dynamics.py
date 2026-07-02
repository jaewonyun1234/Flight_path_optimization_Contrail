"""Time-resolved dynamics diagnostics (§S4): entropy, ground population, residual energy.

The entropy convention-lock test is the arbiter for the qubit-ordering
convention (psi.reshape((2,)*n) axis a <-> qubit n-1-a) — if it fails, fix the
reshape in bipartite_entropy, never the test.
"""

import math

import numpy as np
import pytest

from contrail_env.dynamics import (
    bipartite_entropy,
    residual_energy_vs_T,
    run_with_diagnostics,
)
from contrail_env.pasqal_analog import AnnealSchedule, RydbergStatevector
from contrail_env.quantum_common import OptionGraph

_SCHED = AnnealSchedule(T_ns=4000.0, omega_max=10.0, delta_init=-8.0, delta_final=8.0)


def _toy_graph():
    # One flight, two options -> a single blockaded (one-hot) pair, n=2.
    costs = np.array([2.0, 5.0])
    return OptionGraph(
        n=2, costs=costs, weights=costs.max() - costs + 1.0,
        groups={"F": (0, 1)}, conflict_edges=(), buckets=(),
    )


# =============================================================================
# ENTROPY CONVENTION LOCK
# =============================================================================

def test_entropy_convention_lock():
    n = 3
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1.0 / math.sqrt(2)   # |000>
    psi[5] = 1.0 / math.sqrt(2)   # |101> (bit0=1, bit1=0, bit2=1)
    assert bipartite_entropy(psi, n, (0,)) == pytest.approx(math.log(2), abs=1e-10)
    assert bipartite_entropy(psi, n, (2,)) == pytest.approx(math.log(2), abs=1e-10)
    assert bipartite_entropy(psi, n, (1,)) == pytest.approx(0.0, abs=1e-10)


def test_entropy_product_state_zero():
    n = 3
    psi = np.ones(1 << n, dtype=complex) / math.sqrt(1 << n)   # |+>^{⊗3}
    for cut in ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2)):
        assert bipartite_entropy(psi, n, cut) == pytest.approx(0.0, abs=1e-12)


def test_entropy_ghz():
    n = 3
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1.0 / math.sqrt(2)              # |000>
    psi[(1 << n) - 1] = 1.0 / math.sqrt(2)   # |111>
    for cut in ((0,), (1,), (2,)):
        assert bipartite_entropy(psi, n, cut) == pytest.approx(math.log(2), abs=1e-10)


# =============================================================================
# OBSERVER HOOK — backward compatibility + firing count
# =============================================================================

def test_observer_backward_compat():
    graph = _toy_graph()
    sim = RydbergStatevector(graph)
    probs_plain = sim.run(_SCHED, n_steps=100)

    calls: list[int] = []

    def observer(step, psi):
        calls.append(step)

    probs_observed = sim.run(_SCHED, n_steps=100, observer=observer, observe_stride=4)
    np.testing.assert_allclose(probs_plain, probs_observed, atol=1e-14)

    n_records = 25   # n_steps // stride = 100 // 4
    assert abs(len(calls) - (n_records + 1)) <= 1
    assert calls[0] == 0

    # observe_stride=0 (the default) skips the periodic in-loop captures but the
    # pre-loop observer(0, psi) call still always fires when an observer is set.
    calls2: list[int] = []
    sim.run(_SCHED, n_steps=50, observer=lambda step, psi: calls2.append(step))
    assert calls2 == [0]


# =============================================================================
# ADIABATIC LIMIT + RESIDUAL ENERGY SCALING
# =============================================================================

def test_adiabatic_limit():
    graph = _toy_graph()
    schedule = AnnealSchedule(T_ns=6000.0, omega_max=10.0, delta_init=-8.0, delta_final=8.0)
    record = run_with_diagnostics(graph, schedule, n_steps=400, n_records=20)
    assert record.p_ground[-1] >= 0.9


def test_residual_decreases():
    graph = _toy_graph()
    base = AnnealSchedule(T_ns=4000.0, omega_max=10.0, delta_init=-8.0, delta_final=8.0)
    result = residual_energy_vs_T(graph, base, np.array([600.0, 6000.0]), n_steps=200)
    assert result.residual[0] > result.residual[1]
