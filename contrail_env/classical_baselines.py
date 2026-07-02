"""
classical_baselines.py — Fair classical *samplers* at matched output budget.

WHY THIS MODULE EXISTS
======================
A sampling benchmark needs a classical *sampler* competitor, not only the
exact CP-SAT verifier. Approximation ratio alone cannot tell "the quantum
sampler concentrates on good states" apart from "the repair step is so strong
that even noise repairs to something decent". So we add two classical samplers
that draw the SAME number of raw samples as the quantum pipelines and go
through the SAME repair (quantum_common.repair_sample):

    solve_random_repair       uniform Bernoulli(1/2) bitstrings — the null
                              control. If a quantum sampler cannot beat this,
                              its structure is buying nothing.
    solve_simulated_annealing single-flip Metropolis SA on the penalized QUBO
                              energy — a strong classical heuristic baseline.

Both return a QuantumResult, so they slot into benchmark.py and the Benchmark
tab exactly like the quantum solvers, with no special-casing downstream.

HONEST COMPUTE ACCOUNTING
=========================
The benchmark matches *output samples* (n_samples), but SA spends far more
*energy evaluations* per sample than a quantum shot does. SA therefore records
`meta["energy_evaluations"]` so a reader can see the real classical work behind
its samples — hiding that would flatter SA dishonestly.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

from .flight import EvaluatedOption
from .quantum_common import (
    OptionGraph,
    QuantumResult,
    build_option_graph,
    evaluate_samples,
    make_result,
    penalized_energy,
    repair_sample,
)
from .qubo import CapacityBucket, ConflictEdge


def _repaired_cost(graph: OptionGraph, bits: np.ndarray) -> float:
    """Repaired combined cost of one raw bitstring (the benchmark's unit)."""
    idx = repair_sample(graph, bits)
    return float(sum(graph.costs[i] for i in idx))


def _record_history(
    r: int,
    n_samples: int,
    best: float,
    stride: int,
    history: list[tuple[int, float]],
    on_progress: Callable[[int, float], None] | None,
) -> None:
    """Append (restart, best-so-far) every `stride` restarts and at the end."""
    if r % stride == 0 or r == n_samples - 1:
        history.append((r, best))
        if on_progress is not None:
            on_progress(r, best)


# =============================================================================
# UNIFORM RANDOM + REPAIR — the null control
# =============================================================================

def solve_random_repair(
    evals: list[EvaluatedOption],
    conflicts: list[ConflictEdge],
    buckets: list[CapacityBucket],
    *,
    n_samples: int = 1000,
    seed: int = 0,
    on_progress: Callable[[int, float], None] | None = None,
) -> QuantumResult:
    """Draw `n_samples` uniform bitstrings, repair each, report the best.

    This is the honest floor of the benchmark: uniform noise fed through the
    same greedy repair as the quantum samplers. Any sampler that does not beat
    it is not exploiting problem structure.
    """
    t0 = time.perf_counter()
    graph = build_option_graph(evals, conflicts, buckets)
    rng = np.random.default_rng(seed)
    samples = (rng.random((n_samples, graph.n)) < 0.5).astype(np.uint8)

    stride = max(1, n_samples // 50)
    history: list[tuple[int, float]] = []
    best = math.inf
    # We repair each sample here to stream a live best-so-far curve, and let
    # evaluate_samples repair the (deduplicated) batch for the authoritative
    # aggregate. The repeated repair is cheap at these sizes and keeps the
    # convergence curve honest (best-so-far in draw order).
    for r in range(n_samples):
        cost = _repaired_cost(graph, samples[r])
        if cost < best:
            best = cost
        _record_history(r, n_samples, best, stride, history, on_progress)

    evaluation = evaluate_samples(graph, samples)
    wall = time.perf_counter() - t0
    return make_result(
        solver="random-repair",
        backend="numpy",
        graph=graph,
        evals=evals,
        evaluation=evaluation,
        wall_clock_s=wall,
        history=history,
        meta={
            "n_samples": n_samples,
            "energy_evaluations": n_samples,
            "final_sampling_wall_clock_s": wall,
        },
    )


# =============================================================================
# SIMULATED ANNEALING — single-flip Metropolis on the penalized QUBO energy
# =============================================================================

def _incremental_structure(
    graph: OptionGraph,
) -> tuple[np.ndarray, list[tuple[int, ...]], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Precompute the maps that make one flip's ΔE an O(degree) update.

    Returns (option_group, group_members, conflict_adj, option_buckets, caps).
    """
    n = graph.n
    group_members = list(graph.groups.values())
    option_group = np.empty(n, dtype=np.int64)
    for gid, members in enumerate(group_members):
        for m in members:
            option_group[m] = gid

    adj_lists: list[list[int]] = [[] for _ in range(n)]
    for i, j in graph.conflict_edges:
        adj_lists[i].append(j)
        adj_lists[j].append(i)
    conflict_adj = [np.array(a, dtype=np.int64) for a in adj_lists]

    bucket_lists: list[list[int]] = [[] for _ in range(n)]
    for b_idx, (members, _cap) in enumerate(graph.buckets):
        for m in members:
            bucket_lists[m].append(b_idx)
    option_buckets = [np.array(b, dtype=np.int64) for b in bucket_lists]
    caps = np.array([cap for _members, cap in graph.buckets], dtype=np.int64)
    return option_group, group_members, conflict_adj, option_buckets, caps


def solve_simulated_annealing(
    evals: list[EvaluatedOption],
    conflicts: list[ConflictEdge],
    buckets: list[CapacityBucket],
    *,
    n_samples: int = 1000,
    n_sweeps: int = 64,
    t_hot: float | None = None,
    t_cold: float | None = None,
    seed: int = 0,
    on_progress: Callable[[int, float], None] | None = None,
) -> QuantumResult:
    """Metropolis single-flip SA on E_pen, one returned sample per restart.

    `n_samples` independent restarts; each cools geometrically from `t_hot` to
    `t_cold` over `n_sweeps` sweeps (one sweep = n proposed single flips), and
    contributes its final state as one sample. The batch then goes through the
    shared repair, so SA competes on the same footing as the quantum samplers.

    Energy bookkeeping is incremental: the penalty-form energy change of a
    single flip is an O(degree) formula (see below), so a full restart costs
    O(n_sweeps · n · degree) instead of O(n_sweeps · n · |E|).
    """
    t0 = time.perf_counter()
    graph = build_option_graph(evals, conflicts, buckets)
    n = graph.n
    penalty = graph.penalty()
    costs = graph.costs
    option_group, group_members, conflict_adj, option_buckets, caps = _incremental_structure(graph)
    n_groups = len(group_members)

    # Temperature scale: default t_hot is the spread of E_pen over uniform
    # probes (floored at penalty/10 so it never collapses on a flat landscape).
    if t_hot is None:
        probe_rng = np.random.default_rng([seed, 20240517])
        probes = (probe_rng.random((256, n)) < 0.5).astype(np.uint8)
        energies = np.array([penalized_energy(graph, p) for p in probes])
        t_hot = max(float(energies.std()), penalty / 10.0)
    if t_cold is None:
        t_cold = 1e-3 * t_hot

    if n_sweeps > 1:
        ks = np.arange(n_sweeps)
        temps = t_hot * (t_cold / t_hot) ** (ks / (n_sweeps - 1))
    else:
        temps = np.array([t_hot], dtype=float)

    finals = np.empty((n_samples, n), dtype=np.uint8)
    stride = max(1, n_samples // 50)
    history: list[tuple[int, float]] = []
    best = math.inf

    for r in range(n_samples):
        rng_r = np.random.default_rng([seed, r])
        dbg_rng = np.random.default_rng([seed, r, 7])  # separate so __debug__ never shifts rng_r
        x = (rng_r.random(n) < 0.5).astype(np.uint8)

        occ_g = np.zeros(n_groups, dtype=np.int64)
        for gid, members in enumerate(group_members):
            occ_g[gid] = int(x[list(members)].sum())
        occ_b = np.array(
            [int(x[list(members)].sum()) for members, _cap in graph.buckets], dtype=np.int64
        )
        energy = penalized_energy(graph, x)  # running energy, reset each restart

        for temp in temps:
            for _ in range(n):
                m = int(rng_r.integers(n))
                gid = int(option_group[m])
                nb = conflict_adj[m]
                n_on = int(x[nb].sum()) if nb.size else 0
                bs = option_buckets[m]

                if x[m] == 0:  # 0 -> 1
                    dcap = 0.0
                    for b in bs:
                        ob = int(occ_b[b])
                        cap = int(caps[b])
                        dcap += max(0, ob + 1 - cap) ** 2 - max(0, ob - cap) ** 2
                    d_e = float(costs[m]) + penalty * (2 * int(occ_g[gid]) - 1) \
                        + penalty * n_on + penalty * dcap
                else:  # 1 -> 0
                    dcap = 0.0
                    for b in bs:
                        ob = int(occ_b[b])
                        cap = int(caps[b])
                        dcap += max(0, ob - 1 - cap) ** 2 - max(0, ob - cap) ** 2
                    d_e = -float(costs[m]) + penalty * (3 - 2 * int(occ_g[gid])) \
                        - penalty * n_on + penalty * dcap

                if d_e <= 0.0 or rng_r.random() < math.exp(-d_e / temp):
                    if x[m] == 0:
                        x[m] = 1
                        occ_g[gid] += 1
                        for b in bs:
                            occ_b[b] += 1
                    else:
                        x[m] = 0
                        occ_g[gid] -= 1
                        for b in bs:
                            occ_b[b] -= 1
                    energy += d_e
                    if __debug__ and dbg_rng.random() < 0.01:
                        # Guard the incremental ΔE against a from-scratch energy.
                        # Tolerance is RELATIVE: fuel-scale costs push E_pen to
                        # ~1e6 here, where 1e-9 absolute is below the float64
                        # rounding floor. A real ΔE bug misses by >= one penalty
                        # (~1e4), i.e. ~1e-2 relative — caught easily.
                        full = penalized_energy(graph, x)
                        assert abs(energy - full) <= 1e-9 * max(1.0, abs(full))

        finals[r] = x
        cost = _repaired_cost(graph, x)
        if cost < best:
            best = cost
        _record_history(r, n_samples, best, stride, history, on_progress)

    evaluation = evaluate_samples(graph, finals)
    wall = time.perf_counter() - t0
    return make_result(
        solver="sa",
        backend="numpy",
        graph=graph,
        evals=evals,
        evaluation=evaluation,
        wall_clock_s=wall,
        history=history,
        meta={
            "n_sweeps": n_sweeps,
            "t_hot": round(float(t_hot), 4),
            "t_cold": round(float(t_cold), 6),
            "energy_evaluations": n_samples * n_sweeps * n,
            "final_sampling_wall_clock_s": wall,
        },
    )
