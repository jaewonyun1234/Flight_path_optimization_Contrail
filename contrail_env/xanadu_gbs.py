"""
xanadu_gbs.py — Xanadu photonic Gaussian Boson Sampling pipeline (plan §8).

ENCODING (plan §8.2)
====================
Weighted MIS on the independence graph G (one-hot cliques + ISSR conflicts)
is weighted max-clique on the complement G-bar. Following Arrazola & Bromley
(PRL 121, 030503, 2018) and the WAW weighting of Banchi et al. (Sci. Adv. 6,
eaax1950, 2020), the encoded matrix is

    B = c * W A_bar W,   W = diag(w_i / w_max),   ||B||_spec = c' < 1,

and an ideal GBS device with squeezing r_i = arctanh(lambda_i) from the
Autonne–Takagi decomposition B = U diag(lambda) U^T samples node subsets S
with probability proportional to |Haf(B_S)|^2 — concentrating on dense,
heavy subgraphs of G-bar, i.e. cheap partial assignments.

EXECUTION BACKENDS
==================
    "mh-exact"        Built-in Metropolis–Hastings sampler of the EXACT
                      GBS subset distribution P(S) ∝ |Haf(B_S)|^2 in the
                      collision-free regime. Dependency-free; hafnians are
                      computed exactly (subsets stay small because mean
                      photon number is tuned to ~ the number of flights).
    "strawberryfields" Hand-coded SF Program — Sgate(r_i), Interferometer(U),
                      MeasureFock on the gaussian backend (no sf.apps
                      wrapping), per plan §8.3. Requires
                      `pip install strawberryfields`.
    "auto"            strawberryfields when importable, else mh-exact.

Post-processing is the shrink-grow + capacity repair of plan §8.4, shared
with the Pasqal pipeline (quantum_common.repair_sample).

Physics note: pure squeezed light emits photons in pairs, so raw GBS
samples always have EVEN size — odd-size assignments are reached only
through the grow step. That is faithful to the hardware and is exactly why
the plan separates raw feasibility rate from repaired cost.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .fingerprint import fingerprint_to_flat_dict
from .flight import EvaluatedOption
from .quantum_common import (
    OptionGraph,
    QuantumResult,
    build_option_graph,
    evaluate_samples,
    make_result,
)
from .qubo import CapacityBucket, ConflictEdge

# Spectral norm of the encoded matrix B; sets the squeezing / mean photon
# number (higher = larger subsets sampled). 0.65 keeps mean photon number
# near the number of flights for the instance sizes in the plan.
DEFAULT_TARGET_NORM = 0.65

# MH chain settings: enough mixing for the ~20-node graphs of this project.
_BURN_IN = 400
_THIN = 5
_MOVE_PROBS = (0.4, 0.4, 0.2)  # add-pair, remove-pair, swap-one


# =============================================================================
# HAFNIAN — exact, for the small subsets GBS produces here
# =============================================================================

def hafnian(matrix: np.ndarray) -> float:
    """Exact hafnian by recursive pairing (sum over perfect matchings).

    Haf of an empty matrix is 1, of odd dimension 0. Complexity (m-1)!!,
    fine for the m <= ~12 subsets the sampler visits; memoization on the
    remaining-index set caps repeated work.
    """
    m = matrix.shape[0]
    if m % 2 == 1:
        return 0.0
    memo: dict[tuple[int, ...], float] = {(): 1.0}

    def rec(idx: tuple[int, ...]) -> float:
        if idx in memo:
            return memo[idx]
        first, rest = idx[0], idx[1:]
        total = 0.0
        for pos, j in enumerate(rest):
            coupling = matrix[first, j]
            if coupling != 0.0:
                total += coupling * rec(rest[:pos] + rest[pos + 1:])
        memo[idx] = total
        return total

    return rec(tuple(range(m)))


def takagi_symmetric(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Autonne–Takagi decomposition A = U diag(lam) U^T for real symmetric A.

    From the eigendecomposition A = V diag(e) V^T: lam = |e|, and U = V D
    with D_jj = 1 for e_j >= 0 and i for e_j < 0 (the phase absorbs the
    sign). Returns (lam, U) with lam >= 0 and U unitary.
    """
    eigvals, eigvecs = np.linalg.eigh(matrix)
    phases = np.where(eigvals >= 0, 1.0 + 0.0j, 1.0j)
    u = eigvecs.astype(complex) * phases[None, :]
    return np.abs(eigvals), u


# =============================================================================
# GRAPH -> GBS ENCODING
# =============================================================================

@dataclass(frozen=True)
class GBSEncoding:
    """Everything a GBS device (or its emulator) needs for this instance.

    Attributes:
        B:            encoded symmetric matrix, ||B||_spec < 1
        lam:          Takagi singular values (= tanh of the squeezing)
        U:            interferometer unitary
        squeezing:    per-mode squeezing parameters r_i = arctanh(lam_i)
        mean_photons: expected photon number sum lam^2 / (1 - lam^2)
        max_subset:   post-selection cap on subset size for the sampler
    """

    B: np.ndarray
    lam: np.ndarray
    U: np.ndarray
    squeezing: np.ndarray
    mean_photons: float
    max_subset: int


def encode_option_graph(
    graph: OptionGraph, target_norm: float = DEFAULT_TARGET_NORM
) -> GBSEncoding:
    """WAW-weighted complement-graph encoding (plan §8.2)."""
    n = graph.n
    independence = set(graph.independence_edges())
    a_bar = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in independence:
                a_bar[i, j] = a_bar[j, i] = 1.0

    w_norm = graph.weights / graph.weights.max()
    waw = a_bar * np.outer(w_norm, w_norm)

    spec = float(np.max(np.abs(np.linalg.eigvalsh(waw)))) if n else 0.0
    scale = target_norm / spec if spec > 0 else 1.0
    b = scale * waw

    lam, u = takagi_symmetric(b)
    lam = np.clip(lam, 0.0, 1.0 - 1e-9)
    squeezing = np.arctanh(lam)
    mean_photons = float(np.sum(lam**2 / (1.0 - lam**2)))

    # One option per flight is the largest useful clique; cap a bit above
    # (rounded even — pure GBS emits photon pairs) to bound hafnian cost.
    cap = graph.n_flights + 2
    max_subset = min(cap + (cap % 2), max(4, n))
    return GBSEncoding(
        B=b, lam=lam, U=u, squeezing=squeezing,
        mean_photons=mean_photons, max_subset=max_subset,
    )


# =============================================================================
# EXACT GBS SUBSET SAMPLER (Metropolis–Hastings on P(S) ∝ |Haf(B_S)|^2)
# =============================================================================

class GBSSubsetSampler:
    """MH chain over node subsets with the exact GBS target distribution.

    States are even-size subsets (pure squeezed states emit photon pairs);
    moves are add-pair / remove-pair / swap-one, each with the proper
    Hastings correction. Subset probabilities are cached, so revisited
    states cost a dict lookup instead of a hafnian.
    """

    def __init__(self, encoding: GBSEncoding, rng: np.random.Generator) -> None:
        self._b = encoding.B
        self._n = encoding.B.shape[0]
        self._max = encoding.max_subset
        self._rng = rng
        self._prob_cache: dict[tuple[int, ...], float] = {(): 1.0}
        self._state: tuple[int, ...] = ()

    def _prob(self, subset: tuple[int, ...]) -> float:
        if subset not in self._prob_cache:
            sub = self._b[np.ix_(subset, subset)]
            h = hafnian(sub)
            self._prob_cache[subset] = h * h
        return self._prob_cache[subset]

    def _propose(self) -> tuple[tuple[int, ...], float] | None:
        """One proposal: (new_subset, hastings_ratio) or None if invalid."""
        s = self._state
        size = len(s)
        outside = self._n - size
        kind = self._rng.choice(3, p=_MOVE_PROBS)
        p_add, p_rem, _ = _MOVE_PROBS

        if kind == 0:  # add a pair
            if size + 2 > self._max or outside < 2:
                return None
            pool = [v for v in range(self._n) if v not in s]
            i, j = self._rng.choice(len(pool), size=2, replace=False)
            new = tuple(sorted(s + (pool[i], pool[j])))
            q_fwd = p_add / math.comb(outside, 2)
            q_rev = p_rem / math.comb(size + 2, 2)
            return new, q_rev / q_fwd

        if kind == 1:  # remove a pair
            if size < 2:
                return None
            i, j = self._rng.choice(size, size=2, replace=False)
            keep = tuple(v for pos, v in enumerate(s) if pos not in (i, j))
            q_fwd = p_rem / math.comb(size, 2)
            q_rev = p_add / math.comb(outside + 2, 2)
            return keep, q_rev / q_fwd

        # swap one member for one outsider (parity preserved, symmetric)
        if size == 0 or outside == 0:
            return None
        out_pos = int(self._rng.integers(size))
        pool = [v for v in range(self._n) if v not in s]
        in_node = pool[int(self._rng.integers(len(pool)))]
        new = tuple(sorted(v for pos, v in enumerate(s) if pos != out_pos) + [in_node])
        return tuple(sorted(new)), 1.0

    def _step(self) -> None:
        proposal = self._propose()
        if proposal is None:
            return
        new, hastings = proposal
        p_old = self._prob(self._state)
        p_new = self._prob(new)
        if p_new <= 0.0:
            return
        accept = 1.0 if p_old <= 0.0 else min(1.0, (p_new / p_old) * hastings)
        if self._rng.random() < accept:
            self._state = new

    def sample(
        self,
        n_samples: int,
        on_batch: Callable[[int], None] | None = None,
        batch_size: int = 50,
    ) -> list[tuple[int, ...]]:
        """Draw `n_samples` subsets (after burn-in, thinned)."""
        for _ in range(_BURN_IN):
            self._step()
        out: list[tuple[int, ...]] = []
        while len(out) < n_samples:
            for _ in range(_THIN):
                self._step()
            out.append(self._state)
            if on_batch is not None and len(out) % batch_size == 0:
                on_batch(len(out))
        return out


# =============================================================================
# OPTIONAL: REAL STRAWBERRY FIELDS BACKEND (hand-coded Program, plan §8.3)
# =============================================================================

def _strawberryfields_sample(
    encoding: GBSEncoding, n_samples: int
) -> list[tuple[int, ...]]:
    """Sgate + Interferometer + MeasureFock on the gaussian backend."""
    import strawberryfields as sf  # type: ignore[import-not-found]
    from strawberryfields.ops import (  # type: ignore[import-not-found]
        Interferometer,
        MeasureFock,
        Sgate,
    )

    n = encoding.B.shape[0]
    prog = sf.Program(n)
    with prog.context as q:
        for i in range(n):
            Sgate(float(encoding.squeezing[i])) | q[i]
        Interferometer(encoding.U) | q
        MeasureFock() | q

    engine = sf.Engine("gaussian")
    samples = engine.run(prog, shots=n_samples).samples
    return [tuple(np.where(s >= 1)[0]) for s in samples]


def strawberryfields_available() -> bool:
    """True when strawberryfields is importable."""
    try:
        import strawberryfields  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


# =============================================================================
# THE SOLVER
# =============================================================================

def solve_xanadu_gbs(
    evals: list[EvaluatedOption],
    conflicts: list[ConflictEdge],
    buckets: list[CapacityBucket],
    *,
    n_samples: int = 1000,
    seed: int = 0,
    backend: str = "auto",
    target_norm: float = DEFAULT_TARGET_NORM,
    on_progress: Callable[[int, float], None] | None = None,
) -> QuantumResult:
    """GBS sampling + shrink-grow repair for the flight-option QUBO.

    Pipeline (plan §8, §10.1 step 4): encode the weighted complement graph,
    draw `n_samples` photon-click subsets, repair each into a full
    assignment, report best repaired cost / raw feasibility / wall clock.

    `on_progress(samples_drawn, best_repaired_cost_so_far)` fires every 50
    samples — the energy-vs-samples convergence curve of the plan.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    graph = build_option_graph(evals, conflicts, buckets)
    encoding = encode_option_graph(graph, target_norm=target_norm)

    use_sf = backend == "strawberryfields" or (
        backend == "auto" and strawberryfields_available()
    )

    def subset_to_bits(subset: tuple[int, ...]) -> np.ndarray:
        bits = np.zeros(graph.n, dtype=np.uint8)
        for v in subset:
            bits[v] = 1
        return bits

    history: list[tuple[int, float]] = []

    # Time ONLY the draw of the samples (the per-shot cost t_shot the TTS
    # metric needs); for GBS the sampling IS the pipeline, so there is no
    # separate tuning phase to subtract.
    t_sample0 = time.perf_counter()
    if use_sf:
        subsets = _strawberryfields_sample(encoding, n_samples)
        backend_name = "sf-gaussian"
    else:
        sampler = GBSSubsetSampler(encoding, rng)
        collected: list[tuple[int, ...]] = []

        def on_batch(count: int) -> None:
            # Score the batch incrementally so the GUI sees a live curve; skip
            # the §S5 fingerprint here (only best_cost is read, and it would
            # re-repair every unique row on every batch).
            partial = evaluate_samples(
                graph, (subset_to_bits(s) for s in collected[:count]),
                collect_fingerprint=False,
            )
            history.append((count, partial.best_cost))
            if on_progress is not None:
                on_progress(count, partial.best_cost)

        collected = sampler.sample(n_samples, on_batch=on_batch)
        subsets = collected
        backend_name = "mh-exact"
    final_sampling_wall_clock_s = time.perf_counter() - t_sample0

    evaluation = evaluate_samples(graph, (subset_to_bits(s) for s in subsets))
    history.append((evaluation.n_samples, evaluation.best_cost))
    if on_progress is not None:
        on_progress(evaluation.n_samples, evaluation.best_cost)

    return make_result(
        solver="xanadu-gbs",
        backend=backend_name,
        graph=graph,
        evals=evals,
        evaluation=evaluation,
        wall_clock_s=time.perf_counter() - t0,
        history=history,
        meta={
            "n_modes": graph.n,
            "target_spectral_norm": target_norm,
            "mean_photons": round(encoding.mean_photons, 2),
            "max_squeezing": round(float(encoding.squeezing.max()), 3),
            "max_subset": encoding.max_subset,
            "final_sampling_wall_clock_s": final_sampling_wall_clock_s,
            "fingerprint": (
                fingerprint_to_flat_dict(evaluation.fingerprint) if evaluation.fingerprint else {}
            ),
        },
    )
