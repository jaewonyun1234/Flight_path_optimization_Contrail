"""
spectral.py — Instantaneous spectrum of the analog Rydberg sweep (§S3).

The canonical adiabaticity diagnostic. As the schedule sweeps s = t/T from 0
to 1, the instantaneous Hamiltonian H(s) has eigenvalues E_k(s); the ground gap

    Delta(s) = E_1(s) - E_0(s)

and its minimum Delta_min at s* control how slowly the sweep must run: the
adiabatic condition is T >~ O(Omega_max / Delta_min^2). This module returns the
spectrum so the Dynamics tab can plot it and check the BO-chosen T against
Delta_min. It is a DYNAMICS DIAGNOSTIC, not a performance claim.

WHY float64
===========
H(s) is REAL SYMMETRIC in the computational basis: the only off-diagonal term is
the transverse drive (Omega/2) sum_i sigma^x_i, which is real; the detuning and
Rydberg blockade are diagonal. So the whole spectral computation runs in float64
via a matrix-free scipy.sparse.linalg.eigsh on a LinearOperator — only the
time-dependent state vector (dynamics.py) needs to be complex.

The Hamiltonian and every constant are reused verbatim from pasqal_analog
(RydbergStatevector.interaction_diag / detune_load, AnnealSchedule); nothing is
re-derived here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

from .pasqal_analog import AnnealSchedule, RydbergStatevector
from .quantum_common import BackendBudgetError, OptionGraph

# Above this many qubits the spectrum needs allow_large; the dense fallback for
# stalled Lanczos runs is only affordable up to 12 qubits (a 4096^2 float64
# matrix is ~134 MB).
DEFAULT_MAX_QUBITS = 16
DENSE_FALLBACK_MAX_QUBITS = 12
_DENSE_DIM = 64  # small systems: dense eigh is cheaper (and eigsh needs k < dim-1)


@dataclass(frozen=True)
class AnalogSpectrum:
    """Instantaneous eigenvalues of H(s) along the sweep.

    Attributes:
        s:              (m,) grid of s = t/T values
        energies:       (m, k) lowest-k eigenvalues, ascending, rad/us
        gap:            (m,) E_1(s) - E_0(s)
        delta_min:      minimum gap over the interior grid
        s_star:         s at which delta_min occurs
        end_degeneracy: #{k : E_k(s_last) - E_0(s_last) < deg_tol} at s = last
        n:              number of qubits
        k:              number of eigenvalues actually returned
        s_star_at_boundary: True when s* is the first/last scanned point, so the
                        true minimum may lie outside the window and delta_min /
                        s* are lower-bound estimates, not interior minima
    """

    s: np.ndarray
    energies: np.ndarray
    gap: np.ndarray
    delta_min: float
    s_star: float
    end_degeneracy: int
    n: int
    k: int
    s_star_at_boundary: bool = False


def hamiltonian_matvec(psi: np.ndarray, diag: np.ndarray, omega: float, n: int) -> np.ndarray:
    """y = diag * psi + (omega/2) * sum_q X_q psi, matrix-free, float64.

    X_q swaps the two halves of psi along qubit-q's axis — the same block-swap
    reshape RydbergStatevector.run uses for its drive step (bit q of the basis
    index is qubit q, so axis a of psi.reshape((2,)*n) is qubit n-1-a).
    """
    psi = np.asarray(psi, dtype=np.float64).reshape(-1)
    y = diag * psi
    half = 0.5 * omega
    if half != 0.0:
        drive = np.zeros_like(psi)
        for q in range(n):
            block = psi.reshape(1 << (n - 1 - q), 2, 1 << q)
            drive += block[:, ::-1, :].reshape(-1)
        y += half * drive
    return y


def _dense_hamiltonian(diag: np.ndarray, omega: float, n: int) -> np.ndarray:
    """Full real-symmetric H(s) as a dense float64 matrix (small n only)."""
    dim = diag.shape[0]
    ham = np.diag(diag).astype(np.float64)
    half = 0.5 * omega
    if half != 0.0:
        idx = np.arange(dim)
        for q in range(n):
            ham[idx, idx ^ (1 << q)] += half
    return ham


def _eigsh_lowest(
    diag: np.ndarray, omega: float, n: int, k_eff: int, return_vectors: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Lowest-k_eff eigenpairs via Lanczos; retry with a larger Krylov space.

    Near s in {0, 1} the drive vanishes and the spectrum clusters, which stalls
    Lanczos — so on ArpackNoConvergence we retry once with a bigger ncv before
    letting the caller fall back to dense.
    """
    dim = diag.shape[0]
    op = LinearOperator(
        (dim, dim), matvec=lambda p: hamiltonian_matvec(p, diag, omega, n), dtype=np.float64
    )
    last: ArpackNoConvergence | None = None
    for ncv in (None, min(dim, max(4 * k_eff, 40))):
        kw: dict[str, object] = {"k": k_eff, "which": "SA", "tol": 1e-9}
        if ncv is not None:
            kw["ncv"] = ncv
        try:
            if return_vectors:
                w, v = eigsh(op, return_eigenvectors=True, **kw)
                order = np.argsort(w)
                return w[order], v[:, order]
            w = eigsh(op, return_eigenvectors=False, **kw)
            return np.sort(w), None
        except ArpackNoConvergence as exc:
            last = exc
    assert last is not None
    raise last


def lowest_eigenpairs(
    diag: np.ndarray, omega: float, n: int, k: int, *, return_vectors: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Lowest-k eigenvalues (ascending), optionally with eigenvectors.

    Uses matrix-free Lanczos for large systems and dense eigh for small ones or
    as the convergence fallback (n <= 12). Raises BackendBudgetError only when
    Lanczos stalls on a system too large for the dense fallback.
    """
    dim = diag.shape[0]
    k_eff = min(k, dim)
    if dim > _DENSE_DIM and k_eff < dim - 1:
        try:
            return _eigsh_lowest(diag, omega, n, k_eff, return_vectors)
        except ArpackNoConvergence as exc:
            if n > DENSE_FALLBACK_MAX_QUBITS:
                raise BackendBudgetError(
                    f"eigsh stalled near Omega~0 at n={n} qubits, above the "
                    f"{DENSE_FALLBACK_MAX_QUBITS}-qubit dense-fallback limit; "
                    "shrink the s-grid away from the endpoints or reduce k"
                ) from exc
    ham = _dense_hamiltonian(diag, omega, n)
    w, v = np.linalg.eigh(ham)
    return (w[:k_eff], v[:, :k_eff] if return_vectors else None)


def _argmin_at_boundary(gap: np.ndarray) -> bool:
    """True if the gap's minimum sits at the first or last scanned point.

    A boundary argmin means the true delta_min may lie outside the scan window
    (e.g. when delta_final leaves the classical gap shrinking monotonically into
    the endpoint), so the reported delta_min / s* are then lower-bound estimates.
    """
    i = int(np.argmin(gap))
    return i == 0 or i == gap.shape[0] - 1


def instantaneous_spectrum(
    graph: OptionGraph,
    schedule: AnnealSchedule,
    *,
    s_grid: np.ndarray | None = None,
    k: int = 6,
    max_qubits: int = DEFAULT_MAX_QUBITS,
    allow_large: bool = False,
    deg_tol_rel: float = 1e-8,
    on_phase: Callable[[str, float], None] | None = None,
) -> AnalogSpectrum:
    """Eigenvalues E_k(s), the gap Delta(s), Delta_min at s*, and end degeneracy.

    `schedule` is duck-typed: it only needs omega_at(s) / delta_at(s). The
    default s-grid excludes the endpoints (where degenerate optima legitimately
    drive Delta -> 0), and Delta_min / s* are taken over the interior only.
    """
    n = graph.n
    limit = 20 if allow_large else max_qubits
    if n > limit:
        raise BackendBudgetError(
            f"instantaneous_spectrum: n = {n} qubits exceeds the {limit}-qubit "
            f"budget (pass allow_large=True to raise it to 20)"
        )
    if s_grid is None:
        s_grid = np.linspace(0.02, 0.98, 49)
    s_grid = np.asarray(s_grid, dtype=np.float64)

    sim = RydbergStatevector(graph)
    inter = sim.interaction_diag
    detune_load = sim.detune_load

    m = len(s_grid)
    k_eff = min(k, 1 << n)
    energies = np.empty((m, k_eff), dtype=np.float64)
    for i, s in enumerate(s_grid):
        omega = schedule.omega_at(float(s))
        delta = schedule.delta_at(float(s))
        diag = inter - delta * detune_load
        w, _ = lowest_eigenpairs(diag, omega, n, k_eff)
        energies[i, :] = w
        if on_phase is not None:
            on_phase(f"spectrum: s={float(s):.2f}", (i + 1) / m)

    gap = (energies[:, 1] - energies[:, 0]) if k_eff >= 2 else np.zeros(m)
    # Delta_min / s* over the interior grid only.
    interior = (s_grid > 1e-9) & (s_grid < 1.0 - 1e-9)
    search = np.where(interior, gap, np.inf)
    i_min = int(np.argmin(search))
    delta_min = float(gap[i_min])
    s_star = float(s_grid[i_min])

    e_last = energies[-1]
    deg_tol = deg_tol_rel * max(1.0, abs(float(e_last[0])))
    end_degeneracy = int(np.sum(e_last - e_last[0] < deg_tol))

    return AnalogSpectrum(
        s=s_grid,
        energies=energies,
        gap=gap,
        delta_min=delta_min,
        s_star=s_star,
        end_degeneracy=end_degeneracy,
        n=n,
        k=k_eff,
        # Flag on `search` (the interior-masked array s* is drawn from), so the
        # boundary check stays consistent with the reported s_star.
        s_star_at_boundary=_argmin_at_boundary(search),
    )
