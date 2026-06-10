"""
quantum_common.py — Shared structures for the quantum solver pipelines.

Both quantum backends (Pasqal analog Rydberg, Xanadu photonic GBS) consume
the SAME reduced view of the problem:

    OptionGraph = option variables only (no capacity slack bits)
                  + independence edges (one-hot pairs + ISSR conflicts)
                  + node weights derived from costs (cheap option = heavy node)

WHY NO SLACK BITS?
==================
The QUBO in qubo.py carries binary slack bits so the capacity inequality
becomes an exact quadratic penalty. The quantum samplers do not need them:
capacity is restored classically by the repair step below — the same choice
the research plan makes for the laptop emulator tier ("slack bits handled
classically", plan §9.2). This keeps the register/mode count at sum_f K_f.

REPAIR PIPELINE (plan §7.7 / §8.4)
==================================
Quantum samplers return bitstrings that may violate constraints. Repair is
one deterministic greedy pass, seeded by the sample:

    1. one-hot   : per flight, prefer the cheapest *sampled* option;
    2. conflict  : flights are finalized in cost order and never pick an
                   option that conflicts with an already-finalized one;
    3. capacity  : a candidate that would overflow a (sector, time-bucket)
                   is skipped while a compatible alternative exists.

Both pipelines report the RAW feasibility rate (before repair) — metric 3
of the benchmarking protocol (§10.2) — alongside the best repaired cost.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .flight import EvaluatedOption
from .qubo import CapacityBucket, ConflictEdge


class BackendBudgetError(RuntimeError):
    """Instance exceeds the qubit/mode budget of the requested backend."""


# =============================================================================
# OPTION GRAPH — the single object both quantum pipelines embed
# =============================================================================

@dataclass(frozen=True)
class OptionGraph:
    """Weighted independence structure over option variables (no slack bits).

    Attributes:
        n:              number of option variables (= len(evals))
        costs:          combined cost per option, shape (n,)
        weights:        MIS node weights, shape (n,); higher = more desirable
        groups:         flight_name -> tuple of option variable ids
        conflict_edges: cross-flight ISSR conflicts, (i, j) with i < j
        buckets:        ((member ids...), capacity) per binding bucket
    """

    n: int
    costs: np.ndarray
    weights: np.ndarray
    groups: dict[str, tuple[int, ...]]
    conflict_edges: tuple[tuple[int, int], ...]
    buckets: tuple[tuple[tuple[int, ...], int], ...]

    @property
    def n_flights(self) -> int:
        return len(self.groups)

    def independence_edges(self) -> list[tuple[int, int]]:
        """Conflict edges plus the one-hot cliques (same-flight pairs).

        An independent set of this graph is a partial assignment: at most one
        option per flight, no two conflicting options. This is the edge set
        the Rydberg blockade enforces and the GBS complement graph encodes.
        """
        edges = list(self.conflict_edges)
        for members in self.groups.values():
            for a_pos, a in enumerate(members):
                for b in members[a_pos + 1:]:
                    edges.append((a, b))
        return edges

    def penalty(self) -> float:
        """Penalty constant > c_max − c_min (Lucas 2014 bound, 2x safety)."""
        span = float(self.costs.max() - self.costs.min())
        return 2.0 * (1.0 + span)


def build_option_graph(
    evals: list[EvaluatedOption],
    conflicts: list[ConflictEdge],
    buckets: list[CapacityBucket],
) -> OptionGraph:
    """Reduce the evaluated scenario to the OptionGraph the samplers embed.

    Node weights follow the GBS weighted-MIS convention (plan §8.2):
    w_i = c_max − c_i + margin, so the cheapest option is the heaviest node
    and every weight is strictly positive.
    """
    costs = np.array([ev.cost_combined for ev in evals], dtype=float)
    span = max(float(costs.max() - costs.min()), 1.0)
    weights = (costs.max() - costs) + 0.25 * span

    groups: dict[str, list[int]] = {}
    for i, ev in enumerate(evals):
        groups.setdefault(ev.flight_name, []).append(i)

    return OptionGraph(
        n=len(evals),
        costs=costs,
        weights=weights,
        groups={name: tuple(members) for name, members in groups.items()},
        conflict_edges=tuple((e.i, e.j) for e in conflicts),
        buckets=tuple((tuple(b.members), b.capacity) for b in buckets),
    )


# =============================================================================
# FEASIBILITY + ENERGY ON RAW SAMPLES
# =============================================================================

def sample_is_feasible(graph: OptionGraph, bits: np.ndarray) -> bool:
    """True iff `bits` satisfies one-hot, conflict, and capacity exactly."""
    for members in graph.groups.values():
        if sum(int(bits[m]) for m in members) != 1:
            return False
    for i, j in graph.conflict_edges:
        if bits[i] and bits[j]:
            return False
    for members, cap in graph.buckets:
        if sum(int(bits[m]) for m in members) > cap:
            return False
    return True


def penalized_energy(graph: OptionGraph, bits: np.ndarray) -> float:
    """Penalty-form energy of one bitstring over the option variables.

    Mirrors the QUBO Hamiltonian (plan §3.1) with capacity expressed as the
    inequality violation squared (slack bits resolved analytically):

        E = sum c_i x_i + A sum_f (sum_k x − 1)^2
          + B sum_conflicts x_i x_j + C sum_b max(0, occ_b − cap_b)^2
    """
    p = graph.penalty()
    e = float(np.dot(graph.costs, bits))
    for members in graph.groups.values():
        occ = sum(int(bits[m]) for m in members)
        e += p * (occ - 1) ** 2
    for i, j in graph.conflict_edges:
        e += p * int(bits[i]) * int(bits[j])
    for members, cap in graph.buckets:
        over = sum(int(bits[m]) for m in members) - cap
        if over > 0:
            e += p * over * over
    return e


# =============================================================================
# REPAIR — greedy feasibility restoration, seeded by the sample
# =============================================================================

def repair_sample(graph: OptionGraph, bits: np.ndarray) -> list[int]:
    """Turn a raw sample into a full assignment (one eval index per flight).

    Deterministic. Always restores one-hot; restores conflicts and capacity
    greedily (a violation can only survive if a flight has no compatible
    option left, which CP-SAT would report as infeasible too).
    """
    adjacency: dict[int, set[int]] = {i: set() for i in range(graph.n)}
    for i, j in graph.conflict_edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    option_buckets: dict[int, list[int]] = {i: [] for i in range(graph.n)}
    for b_idx, (members, _cap) in enumerate(graph.buckets):
        for m in members:
            option_buckets[m].append(b_idx)

    # Per flight: sampled options first (cheapest sampled leads), then the
    # rest by cost. Flights whose sample already chose something go first.
    queue: list[tuple[float, str, list[int]]] = []
    for name, members in graph.groups.items():
        sampled = sorted((m for m in members if bits[m]), key=lambda m: graph.costs[m])
        rest = sorted((m for m in members if not bits[m]), key=lambda m: graph.costs[m])
        lead_cost = graph.costs[sampled[0]] if sampled else graph.costs[rest[0]] + 1e9
        queue.append((float(lead_cost), name, sampled + rest))
    queue.sort(key=lambda item: (item[0], item[1]))

    chosen: list[int] = []
    occupancy = [0] * len(graph.buckets)
    for _lead, _name, candidates in queue:
        pick = None
        for cand in candidates:
            if adjacency[cand] & set(chosen):
                continue
            if any(occupancy[b] + 1 > graph.buckets[b][1] for b in option_buckets[cand]):
                continue
            pick = cand
            break
        if pick is None:  # relax capacity, then conflicts (rare, kept feasible-ish)
            for cand in candidates:
                if not (adjacency[cand] & set(chosen)):
                    pick = cand
                    break
        if pick is None:
            pick = candidates[0]
        chosen.append(pick)
        for b in option_buckets[pick]:
            occupancy[b] += 1
    return sorted(chosen)


# =============================================================================
# SAMPLE BATCH EVALUATION
# =============================================================================

@dataclass
class SampleEvaluation:
    """Aggregate metrics of a batch of raw samples after repair."""

    best_indices: list[int]
    best_cost: float
    feasibility_rate: float
    best_raw_cost: float | None
    n_samples: int


def evaluate_samples(
    graph: OptionGraph,
    samples: Iterable[np.ndarray],
) -> SampleEvaluation:
    """Repair every (unique) sample and aggregate the protocol metrics."""
    unique: dict[bytes, tuple[np.ndarray, int]] = {}
    n_total = 0
    for bits in samples:
        n_total += 1
        key = np.asarray(bits, dtype=np.uint8).tobytes()
        if key in unique:
            unique[key] = (unique[key][0], unique[key][1] + 1)
        else:
            unique[key] = (np.asarray(bits, dtype=np.uint8), 1)

    best_cost = float("inf")
    best_indices: list[int] = []
    best_raw: float | None = None
    n_feasible_raw = 0

    for bits, count in unique.values():
        if sample_is_feasible(graph, bits):
            n_feasible_raw += count
            raw_cost = float(np.dot(graph.costs, bits))
            if best_raw is None or raw_cost < best_raw:
                best_raw = raw_cost
        indices = repair_sample(graph, bits)
        cost = float(sum(graph.costs[i] for i in indices))
        if cost < best_cost:
            best_cost = cost
            best_indices = indices

    return SampleEvaluation(
        best_indices=best_indices,
        best_cost=best_cost,
        feasibility_rate=n_feasible_raw / n_total if n_total else 0.0,
        best_raw_cost=best_raw,
        n_samples=n_total,
    )


# =============================================================================
# RESULT TYPE — shared by both quantum solvers
# =============================================================================

@dataclass
class QuantumResult:
    """Outcome of one quantum solve, in the same shape as CPSATResult.

    Attributes:
        solver:              "pasqal-analog" | "xanadu-gbs"
        backend:             which execution path produced the samples
        chosen:              flight_name -> chosen option_index
        chosen_eval_indices: indices into `evals` (one per flight)
        best_cost:           best repaired combined cost (compare to E*)
        feasibility_rate:    fraction of RAW samples satisfying everything
        best_raw_cost:       best feasible raw-sample cost (None if no raw
                             sample was feasible before repair)
        n_samples:           raw samples drawn
        wall_clock_s:        end-to-end solve time
        history:             convergence curve [(step, best cost so far)]
        meta:                solver-specific parameters for the paper tables
    """

    solver: str
    backend: str
    chosen: dict[str, int]
    chosen_eval_indices: list[int]
    best_cost: float
    feasibility_rate: float
    best_raw_cost: float | None
    n_samples: int
    wall_clock_s: float
    history: list[tuple[int, float]]
    meta: dict[str, object] = field(default_factory=dict)


def make_result(
    *,
    solver: str,
    backend: str,
    graph: OptionGraph,
    evals: list[EvaluatedOption],
    evaluation: SampleEvaluation,
    wall_clock_s: float,
    history: list[tuple[int, float]],
    meta: dict[str, object],
) -> QuantumResult:
    """Assemble a QuantumResult from a SampleEvaluation."""
    chosen = {
        evals[i].flight_name: evals[i].option_index for i in evaluation.best_indices
    }
    return QuantumResult(
        solver=solver,
        backend=backend,
        chosen=chosen,
        chosen_eval_indices=list(evaluation.best_indices),
        best_cost=evaluation.best_cost,
        feasibility_rate=evaluation.feasibility_rate,
        best_raw_cost=evaluation.best_raw_cost,
        n_samples=evaluation.n_samples,
        wall_clock_s=wall_clock_s,
        history=history,
        meta=meta,
    )
