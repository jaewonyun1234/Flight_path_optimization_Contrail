"""
pasqal_analog.py — Pasqal neutral-atom analog pipeline.

Hand-coded adiabatic schedule Omega(t), delta(t) + Bayesian optimization of
its four parameters (T, Omega_max, delta_init, delta_final) following
Tibaldi et al. (arXiv:2501.16229), with weighted-MIS detunings in the
spirit of Saada Khelkhal & Barcikowsky (arXiv:2510.25473).

THE HAMILTONIAN
===============
    H(t)/hbar = sum_i Omega(t)/2 * sigma_x^i        [drive]
              - sum_i delta_i(t) * n_i              [detuning, n = |r><r|]
              + sum_{i<j in E} U * n_i n_j          [Rydberg blockade]

Independence edges E (one-hot cliques + ISSR conflicts) are realized as
blockade pairs: U = C6/hbar / r^6 at the 5-um hardware minimum spacing of
Pulser's AnalogDevice (~55.4 rad/us), so a double excitation across an edge
costs far more than any detuning can pay. Node weights enter as per-atom
detuning scale factors (heavier node = stronger drive toward |r>).

EXECUTION BACKENDS
==================
    "statevector"  Built-in split-operator (Strang) integrator of H(t).
                   Exact dynamics, dependency-free, <= 20 qubits.
    "pulser"       Real Pulser Sequence on AnalogDevice via the QuTiP
                   emulator. Requires `pip install pulser pulser-simulation`
                   AND a conflict graph that embeds as a unit-disk register
                   (see embedding_study.py — falls back cleanly when not).
    "auto"         pulser when available + embeddable + small enough,
                   else statevector.

All schedule parameters are kept inside AnalogDevice's published envelope:
Omega <= 12.57 rad/us, |delta| <= 125.66 rad/us, T <= 6000 ns.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .bayes_opt import gp_minimize
from .embedding_study import (
    MIN_ATOM_DISTANCE_UM,
    embed,
    independence_edges,
)
from .exact import SolveResult, evaluate_samples, raw_metrics
from .problem import Scenario
from .qubo import QUBOInstance

# --- Pulser AnalogDevice hardware envelope (Pulser docs) --------------------
OMEGA_MAX_HW = 12.57          # rad/us, max Rabi amplitude
DETUNING_MAX_HW = 125.66      # rad/us, max |detuning|
T_MAX_NS = 6000.0             # max sequence duration
C6_OVER_HBAR = 865_723.02     # rad us^-1 um^6, Rydberg level n=60

# Blockade strength for edge pairs placed at the minimum spacing.
U_BLOCKADE = C6_OVER_HBAR / MIN_ATOM_DISTANCE_UM**6   # ~55.4 rad/us

# Dense statevector holds one complex amplitude per basis state (2^n), so
# 20 qubits is the laptop ceiling; the QuTiP emulator tier is smaller.
MAX_STATEVECTOR_QUBITS = 20
MAX_PULSER_QUBITS = 12

# Bayesian-optimization search box, Tibaldi-style, clipped to hardware.
# delta_final stays well below U_BLOCKADE so the blockade always wins.
BO_BOUNDS: list[tuple[float, float]] = [
    (1500.0, T_MAX_NS),     # T_ns
    (4.0, OMEGA_MAX_HW),    # Omega_max  [rad/us]
    (-14.0, -2.0),          # delta_init [rad/us]
    (2.0, 14.0),            # delta_final[rad/us]
]


class BackendBudgetError(RuntimeError):
    """Instance exceeds the qubit budget of the requested backend."""


class EmbeddingError(RuntimeError):
    """The independence graph admits no valid unit-disk register layout."""


# =============================================================================
# ANALOG SCHEDULE
# =============================================================================

@dataclass(frozen=True)
class AnnealSchedule:
    """The four-parameter adiabatic schedule.

    Omega(t): 0 -> Omega_max (ramp), hold, -> 0 (ramp).
    delta(t): hold delta_init (< 0, favors |g>), linear sweep, hold
              delta_final (> 0, favors |r>).

    Both profiles are parameterized over s = t/T in [0, 1], mirroring the
    InterpolatedWaveform([0, om, om, 0]) / ([di, di, df, df]) shapes that
    the Pulser sequence uses verbatim.
    """

    T_ns: float
    omega_max: float
    delta_init: float
    delta_final: float
    ramp_frac: float = 0.2

    def omega_at(self, s: float) -> float:
        r = self.ramp_frac
        if s < r:
            return self.omega_max * s / r
        if s > 1.0 - r:
            return self.omega_max * (1.0 - s) / r
        return self.omega_max

    def delta_at(self, s: float) -> float:
        hold = 0.15
        if s < hold:
            return self.delta_init
        if s > 1.0 - hold:
            return self.delta_final
        frac = (s - hold) / (1.0 - 2.0 * hold)
        return self.delta_init + frac * (self.delta_final - self.delta_init)


# =============================================================================
# BUILT-IN STATE-VECTOR SIMULATOR (split-operator, exact dynamics)
# =============================================================================

def node_weights(costs: np.ndarray) -> np.ndarray:
    """Weighted-MIS node weights: w_i = c_max - c_i + margin.

    The cheapest option becomes the heaviest node and every weight stays
    strictly positive, so the detuning always pulls toward good options.
    """
    costs = np.asarray(costs, dtype=float)
    span = max(float(costs.max() - costs.min()), 1.0)
    return (costs.max() - costs) + 0.25 * span


class RydbergStatevector:
    """Dense state-vector integrator of the analog Rydberg Hamiltonian.

    Strang splitting per time step: the interaction + detuning part is
    diagonal in the computational basis (one phase multiply), the global
    drive factorizes into per-qubit X rotations (n cheap tensor ops). With
    U*dt <~ 0.4 rad per step the second-order splitting error is far below
    sampling noise.

    Memory is one complex vector of length 2^n, so n <= 20 on a laptop.
    """

    def __init__(
        self,
        n: int,
        weights: np.ndarray,
        edges: list[tuple[int, int]],
    ) -> None:
        if n > MAX_STATEVECTOR_QUBITS:
            raise BackendBudgetError(
                f"statevector backend: n = {n} qubits exceeds the "
                f"{MAX_STATEVECTOR_QUBITS}-qubit budget"
            )
        self.n = n
        states = np.arange(1 << n, dtype=np.int64)

        # Per-atom detuning scale u_i in [0.6, 1.0]: heavier (cheaper) nodes
        # are pushed harder toward |r>. This is the weighted-MIS detuning
        # choice, variant "local detuning" of arXiv:2510.25473.
        w = np.asarray(weights, dtype=float)
        w_span = float(w.max() - w.min())
        u = 0.6 + 0.4 * (w - w.min()) / (w_span if w_span > 0 else 1.0)
        self.detuning_scale = u

        # detune_load[z] = sum_i u_i * bit_i(z);  inter[z] = U * #edges-on(z)
        detune_load = np.zeros(1 << n)
        for q in range(n):
            detune_load += u[q] * ((states >> q) & 1)
        inter = np.zeros(1 << n)
        for i, j in edges:
            inter += U_BLOCKADE * (((states >> i) & 1) * ((states >> j) & 1))
        self._detune_load = detune_load
        self._inter = inter

    def run(self, schedule: AnnealSchedule, n_steps: int = 500) -> np.ndarray:
        """Evolve |00...0> under H(t); return the 2^n outcome probabilities."""
        n = self.n
        dt = (schedule.T_ns / 1000.0) / n_steps  # us; energies are rad/us
        psi = np.zeros(1 << n, dtype=np.complex128)
        psi[0] = 1.0

        for k in range(n_steps):
            s = (k + 0.5) / n_steps
            omega = schedule.omega_at(s)
            delta = schedule.delta_at(s)
            half_phase = np.exp(-0.5j * dt * (self._inter - delta * self._detune_load))

            psi *= half_phase
            theta = 0.5 * omega * dt  # exp(-i Omega dt/2 X) per qubit
            c, ms = math.cos(theta), -1j * math.sin(theta)
            for q in range(n):
                block = psi.reshape(1 << (n - 1 - q), 2, 1 << q)
                a = block[:, 0, :].copy()
                b = block[:, 1, :]
                block[:, 0, :] = c * a + ms * b
                block[:, 1, :] = ms * a + c * b
            psi *= half_phase

        probs = np.abs(psi) ** 2
        return probs / probs.sum()


def penalized_energy_vector(scenario: Scenario, qubo: QUBOInstance) -> np.ndarray:
    """Penalty-form energy of EVERY basis state, vectorized (length 2^n).

    Lets the Bayesian-optimization objective be the exact expectation
    <E> = sum_z p_z E(z) instead of a noisy shot average — deterministic,
    smooth, and free once the state vector exists.
    """
    n = scenario.n_vars
    states = np.arange(1 << n, dtype=np.int64)
    p = qubo.penalty_A

    energy = np.zeros(1 << n)
    for i in range(n):
        energy += scenario.costs[i] * ((states >> i) & 1)
    for members in scenario.groups():
        occ = np.zeros(1 << n)
        for m in members:
            occ += (states >> m) & 1
        energy += p * (occ - 1.0) ** 2
    for i, j in scenario.conflicts:
        energy += p * (((states >> i) & 1) * ((states >> j) & 1))
    return energy


def _sample_bits(probs: np.ndarray, n: int, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `shots` basis states from `probs`; return a (shots, n) bit array."""
    drawn = rng.choice(len(probs), size=shots, p=probs)
    return ((drawn[:, None] >> np.arange(n)[None, :]) & 1).astype(np.uint8)


# =============================================================================
# OPTIONAL: REAL PULSER BACKEND (hardware-grade Sequence on AnalogDevice)
# =============================================================================

def unit_disk_register(scenario: Scenario) -> np.ndarray:
    """2D register layout for the scenario's independence graph.

    Uses the shared multi-start embedder (embedding_study.embed, single
    source of truth: greedy init + force-directed refinement + seeded
    restarts) and raises EmbeddingError when no restart produces a valid
    unit-disk placement; the caller then falls back to the
    ideal-blockade statevector backend.
    """
    edges = independence_edges(scenario)
    coords, report = embed(scenario.n_vars, edges)
    if not report.valid:
        raise EmbeddingError(
            f"no valid unit-disk layout after {report.n_restarts_used} restarts: "
            f"{report.missing_edges} missing, {report.spurious_edges} spurious, "
            f"{report.crowded_pairs} crowded pair(s)"
        )
    return coords


def _pulser_sample(
    scenario: Scenario,
    schedule: AnnealSchedule,
    shots: int,
) -> np.ndarray:
    """Build + emulate a real Pulser Sequence. Requires pulser installed."""
    from pulser import Pulse, Register, Sequence  # type: ignore[import-not-found]
    from pulser.devices import AnalogDevice  # type: ignore[import-not-found]
    from pulser.waveforms import InterpolatedWaveform  # type: ignore[import-not-found]
    from pulser_simulation import QutipEmulator  # type: ignore[import-not-found]

    if scenario.n_vars > MAX_PULSER_QUBITS:
        raise BackendBudgetError(
            f"pulser/QuTiP tier: n = {scenario.n_vars} > {MAX_PULSER_QUBITS} qubits"
        )
    coords = unit_disk_register(scenario)
    register = Register.from_coordinates(coords, prefix="q")

    duration = int(schedule.T_ns)
    omega_wf = InterpolatedWaveform(duration, [0.0, schedule.omega_max, schedule.omega_max, 0.0])
    delta_wf = InterpolatedWaveform(
        duration,
        [schedule.delta_init, schedule.delta_init, schedule.delta_final, schedule.delta_final],
    )
    seq = Sequence(register, AnalogDevice)
    seq.declare_channel("rydberg", "rydberg_global")
    seq.add(Pulse(amplitude=omega_wf, detuning=delta_wf, phase=0.0), "rydberg")

    result = QutipEmulator.from_sequence(seq).run()
    counts = result.sample_final_state(N_samples=shots)
    rows: list[np.ndarray] = []
    for bitstring, count in counts.items():
        bits = np.array([int(ch) for ch in bitstring], dtype=np.uint8)
        rows.extend([bits] * int(count))
    return np.vstack(rows)


def pulser_available() -> bool:
    """True when pulser + pulser-simulation are importable."""
    try:
        import pulser  # type: ignore[import-not-found]  # noqa: F401
        import pulser_simulation  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


# =============================================================================
# THE SOLVER — BO loop + final sampling + repair
# =============================================================================

def solve_pasqal_analog(
    scenario: Scenario,
    qubo: QUBOInstance,
    *,
    n_shots: int = 1000,
    bo_iters: int = 16,
    n_steps: int = 500,
    seed: int = 0,
    backend: str = "auto",
    on_progress: Callable[[int, float], None] | None = None,
) -> SolveResult:
    """Analog-QAOA + Bayesian optimization for the flight-option QUBO.

    Pipeline: realize the independence graph as a blockade Hamiltonian;
    Bayes-optimize the schedule against the exact penalized energy
    expectation; sample `n_shots` bitstrings from the best schedule;
    repair; report best repaired cost, raw feasibility rate, wall clock.

    `on_progress(iteration, best_repaired_cost_so_far)` fires once per BO
    evaluation — this is the convergence curve.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    use_pulser = False
    if backend == "pulser":
        use_pulser = True
    elif backend == "auto" and pulser_available() and scenario.n_vars <= MAX_PULSER_QUBITS:
        try:
            unit_disk_register(scenario)
            use_pulser = True
        except EmbeddingError:
            use_pulser = False

    sim: RydbergStatevector | None = None
    if not use_pulser:
        sim = RydbergStatevector(
            scenario.n_vars,
            node_weights(scenario.costs),
            independence_edges(scenario),
        )
    energy_vec = penalized_energy_vector(scenario, qubo)

    # Track the best repaired solution seen anywhere (BO probes included).
    best_cost = float("inf")
    best_z = np.zeros(scenario.n_vars, dtype=int)
    feasibility = 0.0
    iteration = 0

    def consume(samples: np.ndarray) -> None:
        """Score one schedule's samples; update the incumbent."""
        nonlocal best_cost, best_z, feasibility, iteration
        z, cost, feas = evaluate_samples(scenario, qubo, samples)
        if cost < best_cost:
            best_cost = cost
            best_z = z
        feasibility = feas
        iteration += 1
        if on_progress is not None:
            on_progress(iteration, best_cost)

    def objective(params: np.ndarray) -> float:
        schedule = AnnealSchedule(
            T_ns=float(params[0]),
            omega_max=float(params[1]),
            delta_init=float(params[2]),
            delta_final=float(params[3]),
        )
        if use_pulser:
            samples = _pulser_sample(scenario, schedule, shots=200)
            consume(samples)
            idx = (samples.astype(np.int64) << np.arange(scenario.n_vars)).sum(axis=1)
            return float(energy_vec[idx].mean())
        assert sim is not None
        probs = sim.run(schedule, n_steps=n_steps)
        consume(_sample_bits(probs, scenario.n_vars, 256, rng))
        # Exact expectation <E_penalty>: smooth, deterministic BO signal.
        return float(probs @ energy_vec)

    bo = gp_minimize(
        objective,
        BO_BOUNDS,
        n_calls=bo_iters,
        n_initial=max(4, bo_iters // 3),
        rng=rng,
    )

    # Final high-statistics run at the optimized schedule.
    best_schedule = AnnealSchedule(
        T_ns=float(bo.x_best[0]),
        omega_max=float(bo.x_best[1]),
        delta_init=float(bo.x_best[2]),
        delta_final=float(bo.x_best[3]),
    )
    if use_pulser:
        final_samples = _pulser_sample(scenario, best_schedule, shots=n_shots)
    else:
        assert sim is not None
        probs = sim.run(best_schedule, n_steps=n_steps)
        final_samples = _sample_bits(probs, scenario.n_vars, n_shots, rng)

    z, cost, feas = evaluate_samples(scenario, qubo, final_samples)
    if cost < best_cost:
        best_cost = cost
        best_z = z
    raw_best, raw_mean, _ = raw_metrics(final_samples, qubo)

    return SolveResult(
        solver="pasqal-analog",
        best_z=best_z,
        best_cost=best_cost,
        feasibility_rate=feas,
        n_samples=n_shots,
        wall_clock_s=time.perf_counter() - t0,
        raw_best_E=raw_best,
        raw_mean_E=raw_mean,
    )
