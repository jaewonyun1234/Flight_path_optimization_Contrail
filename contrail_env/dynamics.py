"""
dynamics.py — Time-resolved diagnostics of the analog sweep (§S4).

One DynamicsRecord per evolution: bipartite von Neumann entropy S_A(t),
instantaneous ground-space population P_0(t), the penalized-energy expectation
<E_pen>(t), and the spectral gap Delta(s) at the recorded points — plus a
residual-energy-vs-T scaling sweep.

THESE ARE DYNAMICS DIAGNOSTICS that locate the critical window of the analog
sweep; they are not evidence of computational usefulness.

Reuses the S3 spectral machinery (lowest_eigenpairs, the real-symmetric
diagonal built from RydbergStatevector.interaction_diag / detune_load) for
P_0(t) and the gap, and RydbergStatevector.run's observer hook for the live
state vector — nothing here re-derives the Hamiltonian.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from .pasqal_analog import T_MAX_NS, AnnealSchedule, RydbergStatevector, penalized_energy_vector
from .quantum_common import BackendBudgetError, OptionGraph
from .spectral import DEFAULT_MAX_QUBITS, lowest_eigenpairs

# =============================================================================
# BIPARTITE ENTROPY
# =============================================================================

def bipartite_entropy(psi: np.ndarray, n: int, cut: tuple[int, ...]) -> float:
    """Von Neumann entropy (nats) of the reduced state on qubits `cut`.

    Convention: bit q of the basis index is qubit q, so psi.reshape((2,)*n)
    puts qubit n-1-a on axis a. We moveaxis the cut's axes to the front,
    reshape to (2**|cut|, -1), and take the Schmidt spectrum via SVD — S =
    -sum(p ln p) over p = s**2, clipped at 1e-15 to avoid log(0).
    """
    psi_r = np.asarray(psi, dtype=np.complex128).reshape((2,) * n)
    axes_a = sorted(n - 1 - q for q in cut)
    other_axes = [a for a in range(n) if a not in axes_a]
    moved = np.moveaxis(psi_r, axes_a + other_axes, list(range(n)))
    dim_a = 1 << len(axes_a)
    matrix = moved.reshape(dim_a, -1)
    s = np.linalg.svd(matrix, compute_uv=False)
    p = np.clip(s**2, 1e-15, None)
    return float(-np.sum(p * np.log(p)))


def default_cuts(graph: OptionGraph) -> dict[str, tuple[int, ...]]:
    """Physically meaningful entropy cuts — never an arbitrary half-chain alone.

    'flights_half': the option qubits of the first ceil(F/2) flights, in the
    insertion order of graph.groups (a physical bipartition of the register).
    'index_half': tuple(range(n // 2)) — the plain half-chain, for comparison.
    """
    names = list(graph.groups.keys())
    half = math.ceil(len(names) / 2)
    flights_half = tuple(sorted(idx for name in names[:half] for idx in graph.groups[name]))
    index_half = tuple(range(graph.n // 2))
    return {"flights_half": flights_half, "index_half": index_half}


# =============================================================================
# DIAGNOSTICS RUN — entropy, ground-space population, energy vs time
# =============================================================================

@dataclass(frozen=True)
class DynamicsRecord:
    """The arrays behind the Dynamics tab, one row per recorded time point."""

    s: np.ndarray
    t_ns: np.ndarray
    entropies: dict[str, np.ndarray]     # cut label -> (m,), nats
    p_ground: np.ndarray                 # instantaneous ground-SPACE population
    gap: np.ndarray                      # E_1 - E_0 at recorded s
    energy_pen: np.ndarray               # <E_pen> = sum_z |psi_z|^2 E_pen[z]
    schedule: dict[str, float]
    cuts: dict[str, tuple[int, ...]]


def run_with_diagnostics(
    graph: OptionGraph,
    schedule: AnnealSchedule,
    *,
    n_steps: int = 500,
    n_records: int = 25,
    cuts: dict[str, tuple[int, ...]] | None = None,
    k_eigs: int = 6,
    max_qubits: int = DEFAULT_MAX_QUBITS,
    on_phase: Callable[[str, float], None] | None = None,
) -> DynamicsRecord:
    """Evolve under `schedule` once, recording entropy / P_0 / <E_pen> / gap.

    `observe_stride = max(1, n_steps // n_records)`. At each of the ~n_records+1
    recorded points, P_0 and the gap come from the S3 Lanczos eigenpairs of
    H(s) at that instant (real float64 eigenvectors; overlaps with the complex
    state via np.vdot). Entropies are computed on a COPY of the live buffer —
    the observer contract of RydbergStatevector.run forbids holding onto it.
    """
    n = graph.n
    if n > max_qubits:
        raise BackendBudgetError(
            f"run_with_diagnostics: n = {n} qubits exceeds the {max_qubits}-qubit budget"
        )
    if cuts is None:
        cuts = default_cuts(graph)

    sim = RydbergStatevector(graph)
    energy_vec = penalized_energy_vector(graph)
    observe_stride = max(1, n_steps // n_records)

    records: list[tuple[int, np.ndarray]] = []

    def observer(step: int, psi: np.ndarray) -> None:
        records.append((step, psi.copy()))

    sim.run(schedule, n_steps=n_steps, observer=observer, observe_stride=observe_stride)

    m = len(records)
    s_arr = np.empty(m)
    t_ns = np.empty(m)
    entropies = {name: np.empty(m) for name in cuts}
    p_ground = np.empty(m)
    gap = np.empty(m)
    energy_pen = np.empty(m)

    for i, (step, psi) in enumerate(records):
        s = step / n_steps
        s_arr[i] = s
        t_ns[i] = s * schedule.T_ns
        for name, cut in cuts.items():
            entropies[name][i] = bipartite_entropy(psi, n, cut)

        probs = np.abs(psi) ** 2
        energy_pen[i] = float(probs @ energy_vec)

        omega = schedule.omega_at(s)
        delta = schedule.delta_at(s)
        diag = sim.interaction_diag - delta * sim.detune_load
        k_eff = min(k_eigs, 1 << n)
        w, v = lowest_eigenpairs(diag, omega, n, k_eff, return_vectors=True)
        assert v is not None
        e0 = float(w[0])
        tol = 1e-8 * max(1.0, abs(e0))
        overlaps = np.abs(np.array([np.vdot(v[:, j], psi) for j in range(v.shape[1])])) ** 2
        p_ground[i] = float(overlaps[(w - e0) < tol].sum())
        gap[i] = float(w[1] - w[0]) if k_eff >= 2 else 0.0

        if on_phase is not None:
            on_phase(f"dynamics: s={s:.2f}", (i + 1) / m)

    return DynamicsRecord(
        s=s_arr, t_ns=t_ns, entropies=entropies, p_ground=p_ground,
        gap=gap, energy_pen=energy_pen,
        schedule={
            "T_ns": schedule.T_ns, "omega_max": schedule.omega_max,
            "delta_init": schedule.delta_init, "delta_final": schedule.delta_final,
        },
        cuts=cuts,
    )


# =============================================================================
# RESIDUAL-ENERGY SCALING — one evolution per T
# =============================================================================

@dataclass(frozen=True)
class ResidualEnergyResult:
    """eps(T) = <E_pen>_final - min_z E_pen[z], plus a log-log power-law fit."""

    T_ns: np.ndarray
    residual: np.ndarray
    mu: float    # log-log least-squares exponent (eps ~ T^mu); nan if unfittable
    r2: float    # fit quality; nan if unfittable


def residual_energy_vs_T(
    graph: OptionGraph,
    base: AnnealSchedule,
    T_grid: np.ndarray,
    *,
    n_steps: int = 500,
    fit_window: tuple[float, float] | None = None,
    on_phase: Callable[[str, float], None] | None = None,
) -> ResidualEnergyResult:
    """Residual energy at sweep duration T, for every T in T_grid (clipped to T_MAX_NS).

    An empirical Kibble-Zurek-style scaling exponent on ONE instance family —
    not a universality claim. The fit masks eps < 1e-12 (numerically zero) and,
    if given, restricts to `fit_window` (T_lo, T_hi).
    """
    t_grid = np.clip(np.asarray(T_grid, dtype=float), None, T_MAX_NS)
    sim = RydbergStatevector(graph)
    energy_vec = penalized_energy_vector(graph)
    min_e = float(energy_vec.min())

    residual = np.empty(len(t_grid))
    for i, t_ns in enumerate(t_grid):
        sched = replace(base, T_ns=float(t_ns))
        probs = sim.run(sched, n_steps=n_steps)
        residual[i] = float(probs @ energy_vec) - min_e
        if on_phase is not None:
            on_phase(f"residual: T={t_ns:.0f}ns", (i + 1) / len(t_grid))

    mask = residual > 1e-12
    if fit_window is not None:
        lo, hi = fit_window
        mask &= (t_grid >= lo) & (t_grid <= hi)

    if int(mask.sum()) >= 2:
        x = np.log(t_grid[mask])
        y = np.log(residual[mask])
        coef = np.polyfit(x, y, 1)
        mu, intercept = float(coef[0]), float(coef[1])
        y_pred = mu * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    else:
        mu, r2 = math.nan, math.nan

    return ResidualEnergyResult(T_ns=t_grid, residual=residual, mu=mu, r2=r2)
