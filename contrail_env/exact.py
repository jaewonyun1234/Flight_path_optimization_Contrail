"""
exact.py — Ground truth, baselines, and metrics.

Four jobs, all classical and all small:

    1. brute_force_optimum : enumerate every bitstring, return the true
                             ground state of the QUBO. This is the exact
                             answer every sampler is judged against.
    2. solve_random        : uniform random bitstrings fed through the
                             same repair as the quantum sampler — the
                             null control. If a sampler cannot beat this,
                             its structure is buying nothing.
    3. solve_greedy        : the repair pass run ONCE on the all-zeros
                             bitstring — the repair heuristic named as
                             an honest competitor (see below).
    4. metrics             : approximation ratio, raw-energy metrics,
                             feasibility rate — small pure functions.

REPAIR PIPELINE
===============
Samplers return bitstrings that may violate constraints. Repair is one
deterministic greedy pass, seeded by the sample:

    1. one-hot  : per flight, prefer the cheapest *sampled* option
                  (fall back to the cheapest option overall);
    2. conflict : flights are finalized in cost order and never pick an
                  option that conflicts with an already-finalized one.

RAW vs REPAIRED — which number to trust
=======================================
The repair pass IS a competent greedy heuristic: on easy instances it
reaches the exact optimum from pure noise, so every solver ties at
approx_ratio = 1.0 on repaired cost. Repaired cost therefore measures
the PIPELINE, not the sampler. The sampler's own quality is its RAW
QUBO energy E(z) over the unrepaired bitstrings (raw_metrics below) —
that is the quantity it was asked to minimize and repair cannot game
it. solve_greedy makes the hidden heuristic an explicitly named solver
so "beats greedy" is a visible bar, not an accident of plumbing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .problem import Scenario
from .qubo import QUBOInstance, is_feasible

# Above ~22 variables the 2^n enumeration stops being a "seconds on a
# laptop" operation; refuse loudly instead of hanging.
MAX_BRUTE_FORCE_VARS = 22
_CHUNK_BITS = 16  # enumerate in chunks of 2^16 rows to bound memory


@dataclass
class SolveResult:
    """One solver's outcome on one scenario, in run.py's CSV shape.

    raw_best_E / raw_mean_E are QUBO energies of the UNREPAIRED samples
    (the primary quality evidence — see module docstring); None for
    solvers that draw no raw samples (brute force, greedy).
    """

    solver: str
    best_z: np.ndarray
    best_cost: float
    feasibility_rate: float
    n_samples: int
    wall_clock_s: float
    raw_best_E: float | None = None
    raw_mean_E: float | None = None


# =============================================================================
# 1. BRUTE FORCE — the exact optimum
# =============================================================================

def brute_force_optimum(qubo: QUBOInstance) -> tuple[np.ndarray, float]:
    """Return (z_opt, E_min): the global minimum of z^T Q z + constant.

    Vectorized enumeration of all 2^n bitstrings, in chunks so memory
    stays bounded. With penalties above the c_max - c_min bound the
    ground state is guaranteed feasible, so E_min is also the optimal
    COST (penalties contribute zero at a feasible z).
    """
    n = qubo.n
    assert n <= MAX_BRUTE_FORCE_VARS, (
        f"brute force refuses n = {n} > {MAX_BRUTE_FORCE_VARS} variables "
        f"(2^{n} states); shrink the scenario or use a sampler"
    )
    bit_cols = np.arange(n, dtype=np.int64)
    best_e = np.inf
    best_z = np.zeros(n, dtype=int)
    for start in range(0, 1 << n, 1 << _CHUNK_BITS):
        stop = min(start + (1 << _CHUNK_BITS), 1 << n)
        states = np.arange(start, stop, dtype=np.int64)
        Z = ((states[:, None] >> bit_cols[None, :]) & 1).astype(float)
        # z^T Q z for every row at once: (Z Q * Z) summed over columns.
        energies = np.einsum("ij,ij->i", Z @ qubo.Q, Z)
        k = int(np.argmin(energies))
        if energies[k] < best_e:
            best_e = float(energies[k])
            best_z = Z[k].astype(int)
    return best_z, best_e + qubo.constant


# =============================================================================
# 2. REPAIR + RANDOM BASELINE
# =============================================================================

def repair(bits: np.ndarray, scenario: Scenario) -> np.ndarray:
    """Turn a raw sample into a feasible one-hot assignment (z vector).

    Deterministic. Always restores one-hot. Conflicts are restored
    greedily, with a bounded retry: when a flight ends up blocked (every
    option conflicts with an earlier pick), the whole pass reruns with
    that flight promoted to the front of the order so it picks first.
    A conflict can survive only if every retry fails — on dense graphs a
    greedy order may miss an assignment that exists, so this is strong,
    not complete.
    """
    costs = scenario.costs
    adjacency: dict[int, set[int]] = {i: set() for i in range(scenario.n_vars)}
    for i, j in scenario.conflicts:
        adjacency[i].add(j)
        adjacency[j].add(i)

    # Per flight: sampled options first (cheapest sampled leads), then the
    # rest by cost. Flights whose sample already chose something go first.
    queue: list[tuple[float, int, list[int]]] = []
    for f, members in enumerate(scenario.groups()):
        sampled = sorted((m for m in members if bits[m]), key=lambda m: costs[m])
        rest = sorted((m for m in members if not bits[m]), key=lambda m: costs[m])
        lead = float(costs[sampled[0]]) if sampled else float(costs[rest[0]]) + 1e9
        queue.append((lead, f, sampled + rest))
    queue.sort(key=lambda item: (item[0], item[1]))

    def one_pass(order: list[tuple[float, int, list[int]]]) -> tuple[set[int], int | None]:
        """Greedy pass; returns (chosen, first blocked flight or None)."""
        chosen: set[int] = set()
        blocked: int | None = None
        for _lead, f, candidates in order:
            pick = next((c for c in candidates if not (adjacency[c] & chosen)), None)
            if pick is None:  # no conflict-free option left; keep one-hot anyway
                pick = candidates[0]
                if blocked is None:
                    blocked = f
            chosen.add(pick)
        return chosen, blocked

    promoted: list[int] = []
    chosen, blocked = one_pass(queue)
    while blocked is not None and blocked not in promoted:
        promoted.insert(0, blocked)
        order = [item for f in promoted for item in queue if item[1] == f]
        order += [item for item in queue if item[1] not in promoted]
        attempt, attempt_blocked = one_pass(order)
        if attempt_blocked is None:
            chosen, blocked = attempt, None
        else:
            chosen, blocked = attempt, attempt_blocked

    z = np.zeros(scenario.n_vars, dtype=int)
    z[sorted(chosen)] = 1
    return z


def solve_random(
    scenario: Scenario,
    qubo: QUBOInstance,
    n_samples: int = 1000,
    seed: int = 0,
) -> SolveResult:
    """Uniform Bernoulli(1/2) bitstrings + repair: the null control."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    samples = (rng.random((n_samples, scenario.n_vars)) < 0.5).astype(np.uint8)
    best_z, best_cost, feas = evaluate_samples(scenario, qubo, samples)
    raw_best, raw_mean, _ = raw_metrics(samples, qubo)
    return SolveResult(
        solver="random",
        best_z=best_z,
        best_cost=best_cost,
        feasibility_rate=feas,
        n_samples=n_samples,
        wall_clock_s=time.perf_counter() - t0,
        raw_best_E=raw_best,
        raw_mean_E=raw_mean,
    )


def solve_greedy(scenario: Scenario, qubo: QUBOInstance) -> SolveResult:
    """The repair heuristic itself, run once from a blank slate.

    Repairing the all-zeros bitstring means "no option sampled anywhere":
    the pass simply assigns every flight its cheapest conflict-free
    option in cost order. Deterministic, no samples, no randomness —
    the honest classical bar every sampler must clear.
    """
    t0 = time.perf_counter()
    z = repair(np.zeros(scenario.n_vars, dtype=np.uint8), scenario)
    return SolveResult(
        solver="greedy",
        best_z=z,
        best_cost=float(np.dot(scenario.costs, z)),
        feasibility_rate=1.0,
        n_samples=1,
        wall_clock_s=time.perf_counter() - t0,
    )


def evaluate_samples(
    scenario: Scenario,
    qubo: QUBOInstance,
    samples: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Repair every unique sample; return (best_z, best_cost, raw feasibility).

    Deduplicating first keeps the cost linear in the number of DISTINCT
    bitstrings, which is tiny when a sampler concentrates on good states.
    """
    unique, counts = np.unique(np.asarray(samples, dtype=np.uint8), axis=0, return_counts=True)
    best_cost = np.inf
    best_z = np.zeros(scenario.n_vars, dtype=int)
    n_feasible = 0
    for row, count in zip(unique, counts, strict=True):
        if is_feasible(row, qubo)[0]:
            n_feasible += int(count)
        z = repair(row, scenario)
        cost = float(np.dot(scenario.costs, z))
        if cost < best_cost:
            best_cost = cost
            best_z = z
    n_total = int(counts.sum())
    return best_z, best_cost, (n_feasible / n_total if n_total else 0.0)


# =============================================================================
# 3. METRICS — small pure functions
# =============================================================================

def raw_metrics(
    samples: np.ndarray, qubo: QUBOInstance
) -> tuple[float, float, float]:
    """(best_raw_E, mean_raw_E, raw_feasibility) of UNREPAIRED samples.

    Energies are full QUBO energies E(z) = z^T Q z + constant, penalties
    included — exactly what the sampler is asked to minimize, and the one
    number the repair step cannot game. This is the primary evidence of
    sampler quality (see module docstring).
    """
    Z = np.asarray(samples, dtype=float)
    energies = np.einsum("ij,ij->i", Z @ qubo.Q, Z) + qubo.constant
    n_feasible = sum(1 for row in np.asarray(samples) if is_feasible(row, qubo)[0])
    return (
        float(energies.min()),
        float(energies.mean()),
        n_feasible / len(Z) if len(Z) else 0.0,
    )


def approximation_ratio(best_cost: float, e_min: float, e_rand_mean: float) -> float:
    """How much of the random-to-optimal gap did the solver close?

        ratio = (E_rand_mean - best_cost) / (E_rand_mean - E_min)

    1.0 = found the optimum, 0.0 = no better than the mean repaired
    random draw. NaN-safe: when random already sits at the optimum
    (zero gap), any solver that matches E_min gets 1.0.
    """
    gap = e_rand_mean - e_min
    if gap <= 0.0:
        return 1.0 if best_cost <= e_min + 1e-9 else 0.0
    return (e_rand_mean - best_cost) / gap


def mean_random_cost(
    scenario: Scenario, n_samples: int = 256, seed: int = 12345
) -> float:
    """Mean repaired cost of uniform draws — the scale bar for the ratio."""
    rng = np.random.default_rng(seed)
    samples = (rng.random((n_samples, scenario.n_vars)) < 0.5).astype(np.uint8)
    costs = [float(np.dot(scenario.costs, repair(s, scenario))) for s in samples]
    return float(np.mean(costs))
