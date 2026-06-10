"""
pasqal_analog.py — Pasqal neutral-atom analog pipeline (plan §7).

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
                   Exact dynamics, dependency-free, <= 20 qubits — the
                   role emu-sv plays in the plan's tier table (§9.1).
    "pulser"       Real Pulser Sequence on AnalogDevice via the QuTiP
                   emulator. Requires `pip install pulser pulser-simulation`
                   AND a conflict graph that embeds as a unit-disk register
                   (most synthetic instances do not — falls back cleanly).
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
from .flight import EvaluatedOption
from .quantum_common import (
    BackendBudgetError,
    OptionGraph,
    QuantumResult,
    build_option_graph,
    evaluate_samples,
    make_result,
)
from .qubo import CapacityBucket, ConflictEdge

# --- Pulser AnalogDevice hardware envelope (Pulser docs, plan §7.1) ---------
OMEGA_MAX_HW = 12.57          # rad/us, max Rabi amplitude
DETUNING_MAX_HW = 125.66      # rad/us, max |detuning|
T_MAX_NS = 6000.0             # max sequence duration
MIN_ATOM_DISTANCE_UM = 5.0    # min register spacing
C6_OVER_HBAR = 865_723.02     # rad us^-1 um^6, Rydberg level n=60

# Blockade strength for edge pairs placed at the minimum spacing.
U_BLOCKADE = C6_OVER_HBAR / MIN_ATOM_DISTANCE_UM**6   # ~55.4 rad/us

# Tier limits (plan §9.1): dense statevector is the emu-sv-class tier here;
# the QuTiP-backed Pulser emulator is the smallest tier.
MAX_STATEVECTOR_QUBITS = 20
MAX_PULSER_QUBITS = 12

# Bayesian-optimization search box, Tibaldi-style (plan §7.6), clipped to
# hardware. delta_final stays well below U_BLOCKADE so the blockade wins.
BO_BOUNDS: list[tuple[float, float]] = [
    (1500.0, T_MAX_NS),     # T_ns
    (4.0, OMEGA_MAX_HW),    # Omega_max  [rad/us]
    (-14.0, -2.0),          # delta_init [rad/us]
    (2.0, 14.0),            # delta_final[rad/us]
]


# =============================================================================
# ANALOG SCHEDULE
# =============================================================================

@dataclass(frozen=True)
class AnnealSchedule:
    """The four-parameter adiabatic schedule of plan §7.4.

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

class RydbergStatevector:
    """Dense state-vector integrator of the analog Rydberg Hamiltonian.

    Strang splitting per time step: the interaction + detuning part is
    diagonal in the computational basis (one phase multiply), the global
    drive factorizes into per-qubit X rotations (n cheap tensor ops). With
    U*dt <~ 0.4 rad per step the second-order splitting error is far below
    sampling noise.

    Memory is one complex vector of length 2^n, so n <= 20 on a laptop —
    the same niche emu-sv fills in the plan's emulator tiers.
    """

    def __init__(self, graph: OptionGraph) -> None:
        n = graph.n
        if n > MAX_STATEVECTOR_QUBITS:
            raise BackendBudgetError(
                f"statevector backend: n = {n} qubits exceeds the "
                f"{MAX_STATEVECTOR_QUBITS}-qubit budget (plan §9.1 tier table)"
            )
        self.n = n
        states = np.arange(1 << n, dtype=np.int64)

        # Per-atom detuning scale u_i in [0.6, 1.0]: heavier (cheaper) nodes
        # are pushed harder toward |r>. This is the weighted-MIS detuning
        # choice, variant "local detuning" of arXiv:2510.25473.
        w = graph.weights
        w_span = float(w.max() - w.min())
        u = 0.6 + 0.4 * (w - w.min()) / (w_span if w_span > 0 else 1.0)
        self.detuning_scale = u

        # detune_load[z] = sum_i u_i * bit_i(z);  inter[z] = U * #edges-on(z)
        detune_load = np.zeros(1 << n)
        for q in range(n):
            detune_load += u[q] * ((states >> q) & 1)
        inter = np.zeros(1 << n)
        for i, j in graph.independence_edges():
            inter += U_BLOCKADE * (((states >> i) & 1) * ((states >> j) & 1))
        self._detune_load = detune_load
        self._inter = inter

    def run(
        self,
        schedule: AnnealSchedule,
        n_steps: int = 500,
        on_step: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Evolve |00...0> under H(t) and return the 2^n outcome probabilities.

        `on_step(steps_done, n_steps)` fires every ~5% of the integration —
        a liveness heartbeat for callers (one evolution at 19-20 qubits takes
        minutes, and without this there is no sign of life in between).
        """
        n = self.n
        dt = (schedule.T_ns / 1000.0) / n_steps  # us; energies are rad/us
        psi = np.zeros(1 << n, dtype=np.complex128)
        psi[0] = 1.0
        stride = max(1, n_steps // 20)

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
            if on_step is not None and (k + 1) % stride == 0:
                on_step(k + 1, n_steps)

        probs = np.abs(psi) ** 2
        return probs / probs.sum()


def penalized_energy_vector(graph: OptionGraph) -> np.ndarray:
    """Penalty-form energy of EVERY basis state, vectorized (length 2^n).

    Lets the Bayesian-optimization objective be the exact expectation
    <E> = sum_z p_z E(z) instead of a noisy shot average — deterministic,
    smooth, and free once the state vector exists.
    """
    n = graph.n
    states = np.arange(1 << n, dtype=np.int64)
    p = graph.penalty()

    energy = np.zeros(1 << n)
    for i in range(n):
        energy += graph.costs[i] * ((states >> i) & 1)
    for members in graph.groups.values():
        occ = np.zeros(1 << n)
        for m in members:
            occ += (states >> m) & 1
        energy += p * (occ - 1.0) ** 2
    for i, j in graph.conflict_edges:
        energy += p * (((states >> i) & 1) * ((states >> j) & 1))
    for members, cap in graph.buckets:
        occ = np.zeros(1 << n)
        for m in members:
            occ += (states >> m) & 1
        energy += p * np.maximum(0.0, occ - cap) ** 2
    return energy


def _sample_bits(probs: np.ndarray, n: int, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `shots` basis states from `probs`; return a (shots, n) bit array."""
    drawn = rng.choice(len(probs), size=shots, p=probs)
    return ((drawn[:, None] >> np.arange(n)[None, :]) & 1).astype(np.uint8)


# =============================================================================
# OPTIONAL: REAL PULSER BACKEND (hardware-grade Sequence on AnalogDevice)
# =============================================================================

class EmbeddingError(RuntimeError):
    """The independence graph admits no valid unit-disk register layout."""


def unit_disk_embedding(graph: OptionGraph, r_blockade_um: float = 8.0) -> np.ndarray:
    """2D register layout: edges within the blockade radius, non-edges outside.

    Layout: each flight's option clique sits on a small circle (always
    mutually blockaded = one-hot), flight clusters sit on a large circle.
    Validates every pair against the disk-graph condition and raises
    EmbeddingError when the conflict topology is not unit-disk — which is
    the common case for arbitrary conflict graphs; the caller then falls
    back to the ideal-blockade statevector backend.
    """
    edges = set(graph.independence_edges())
    coords = np.zeros((graph.n, 2))
    n_groups = len(graph.groups)
    big_r = 1.6 * r_blockade_um * max(1.0, n_groups / math.pi)
    for g_idx, members in enumerate(graph.groups.values()):
        phi = 2.0 * math.pi * g_idx / max(1, n_groups)
        center = big_r * np.array([math.cos(phi), math.sin(phi)])
        k = len(members)
        for m_idx, m in enumerate(members):
            ang = 2.0 * math.pi * m_idx / max(1, k)
            local_r = 0.35 * r_blockade_um
            coords[m] = center + local_r * np.array([math.cos(ang), math.sin(ang)])

    for i in range(graph.n):
        for j in range(i + 1, graph.n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < MIN_ATOM_DISTANCE_UM:
                raise EmbeddingError(f"atoms {i},{j} closer than hardware minimum")
            is_edge = (i, j) in edges
            if is_edge and d > 0.95 * r_blockade_um:
                raise EmbeddingError(f"edge {i}-{j} outside blockade radius")
            if not is_edge and d < 1.05 * r_blockade_um:
                raise EmbeddingError(f"non-edge {i}-{j} inside blockade radius")
    return coords


def _pulser_sample(
    graph: OptionGraph,
    schedule: AnnealSchedule,
    shots: int,
) -> np.ndarray:
    """Build + emulate a real Pulser Sequence. Requires pulser installed."""
    from pulser import Pulse, Register, Sequence  # type: ignore[import-not-found]
    from pulser.devices import AnalogDevice  # type: ignore[import-not-found]
    from pulser.waveforms import InterpolatedWaveform  # type: ignore[import-not-found]
    from pulser_simulation import QutipEmulator  # type: ignore[import-not-found]

    if graph.n > MAX_PULSER_QUBITS:
        raise BackendBudgetError(
            f"pulser/QuTiP tier: n = {graph.n} > {MAX_PULSER_QUBITS} qubits"
        )
    coords = unit_disk_embedding(graph)
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
    evals: list[EvaluatedOption],
    conflicts: list[ConflictEdge],
    buckets: list[CapacityBucket],
    *,
    n_shots: int = 1000,
    bo_iters: int = 16,
    n_steps: int = 500,
    seed: int = 0,
    backend: str = "auto",
    on_progress: Callable[[int, float], None] | None = None,
    on_phase: Callable[[str, float], None] | None = None,
) -> QuantumResult:
    """Analog-QAOA + Bayesian optimization for the flight-option QUBO.

    Pipeline (plan §7, §10.1 step 3): embed the OptionGraph as a blockade
    Hamiltonian; Bayes-optimize the schedule against the exact penalized
    energy expectation; sample `n_shots` bitstrings from the best schedule;
    repair; report best repaired cost, raw feasibility rate, wall clock.

    `on_progress(iteration, best_repaired_cost_so_far)` fires once per BO
    evaluation — this is the convergence curve of plan §10.3.

    `on_phase(message, fraction_done)` is a fine-grained liveness heartbeat:
    it fires ~20 times per Schrödinger integration, so callers (the GUI) can
    show progress even when a single BO evaluation takes minutes.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    graph = build_option_graph(evals, conflicts, buckets)

    use_pulser = False
    if backend == "pulser":
        use_pulser = True
    elif backend == "auto" and pulser_available() and graph.n <= MAX_PULSER_QUBITS:
        try:
            unit_disk_embedding(graph)
            use_pulser = True
        except EmbeddingError:
            use_pulser = False

    sim: RydbergStatevector | None = None
    energy_vec: np.ndarray | None = None
    if not use_pulser:
        sim = RydbergStatevector(graph)
        energy_vec = penalized_energy_vector(graph)

    # Track the best repaired solution seen anywhere (BO probes included).
    best_cost = float("inf")
    best_indices: list[int] = []
    history: list[tuple[int, float]] = []
    iteration = 0

    def consume(samples: np.ndarray) -> float:
        """Score one schedule's samples; update incumbent + history."""
        nonlocal best_cost, best_indices, iteration
        evaluation = evaluate_samples(graph, samples)
        if evaluation.best_cost < best_cost:
            best_cost = evaluation.best_cost
            best_indices = evaluation.best_indices
        iteration += 1
        history.append((iteration, best_cost))
        if on_progress is not None:
            on_progress(iteration, best_cost)
        return evaluation.best_cost

    # One Schrödinger evolution = 1/(bo_iters + 1) of the work (the +1 is
    # the final high-statistics run). `evals_done` positions the heartbeat.
    def evolve(schedule: AnnealSchedule, label: str, evals_done: int) -> np.ndarray:
        assert sim is not None
        if on_phase is None:
            return sim.run(schedule, n_steps=n_steps)
        emit_phase = on_phase

        def heartbeat(step: int, total: int) -> None:
            frac = (evals_done + step / total) / (bo_iters + 1)
            emit_phase(
                f"{label}: integrating {graph.n}-qubit dynamics, step {step}/{total}",
                frac,
            )

        return sim.run(schedule, n_steps=n_steps, on_step=heartbeat)

    def objective(params: np.ndarray) -> float:
        schedule = AnnealSchedule(
            T_ns=float(params[0]),
            omega_max=float(params[1]),
            delta_init=float(params[2]),
            delta_final=float(params[3]),
        )
        if use_pulser:
            if on_phase is not None:
                on_phase(
                    f"BO eval {iteration + 1}/{bo_iters}: Pulser/QuTiP emulation",
                    iteration / (bo_iters + 1),
                )
            samples = _pulser_sample(graph, schedule, shots=200)
            consume(samples)
            from .quantum_common import penalized_energy
            return float(np.mean([penalized_energy(graph, s) for s in samples]))
        assert sim is not None and energy_vec is not None
        probs = evolve(schedule, f"BO eval {iteration + 1}/{bo_iters}", iteration)
        consume(_sample_bits(probs, graph.n, 256, rng))
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
        final_samples = _pulser_sample(graph, best_schedule, shots=n_shots)
    else:
        assert sim is not None
        probs = evolve(best_schedule, "final sampling at the best schedule", bo_iters)
        final_samples = _sample_bits(probs, graph.n, n_shots, rng)

    evaluation = evaluate_samples(graph, final_samples)
    if evaluation.best_cost > best_cost:
        # Keep the best incumbent found during BO probing.
        evaluation.best_cost = best_cost
        evaluation.best_indices = best_indices
    history.append((iteration + 1, evaluation.best_cost))

    return make_result(
        solver="pasqal-analog",
        backend="pulser-qutip" if use_pulser else "statevector",
        graph=graph,
        evals=evals,
        evaluation=evaluation,
        wall_clock_s=time.perf_counter() - t0,
        history=history,
        meta={
            "T_ns": round(best_schedule.T_ns, 1),
            "omega_max_rad_us": round(best_schedule.omega_max, 3),
            "delta_init_rad_us": round(best_schedule.delta_init, 3),
            "delta_final_rad_us": round(best_schedule.delta_final, 3),
            "U_blockade_rad_us": round(U_BLOCKADE, 2),
            "n_qubits": graph.n,
            "bo_iters": bo_iters,
            "mean_penalized_energy": round(bo.y_best, 2),
        },
    )
