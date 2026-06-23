"""
analysis.py — Research-grade diagnostics for the contrail QUBO.

Pure functions (no Qt, no plotting) so the science is unit-testable on its own;
the GUI tabs just call these and draw the result. Three analyses:

1. ENERGY LANDSCAPE (`feasible_cost_landscape`) — the combined cost of every
   feasible one-hot assignment. The distribution + where the optimum sits is the
   thing a sampler is trying to navigate.
2. SAMPLER QUALITY (`gbs_sample_costs`, `random_sample_costs`) — the repaired
   costs a GBS sampler vs a uniform-random baseline actually achieve, overlaid on
   the landscape: does the quantum sampler concentrate near the optimum?
3. HARDNESS / SCALING (`hardness_sweep`) — CP-SAT solve time + incumbents vs
   problem size: where does the instance stop being trivial?

Costs are in the SAME units everywhere — `EvaluatedOption.cost_combined` (which
equals `OptionGraph.costs[i]`) — so the landscape and the sample overlays line up.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .options import EvaluatedOption
    from .qubo import CapacityBucket, ConflictEdge


# ===========================================================================
# 1. ENERGY LANDSCAPE — exact, for small instances
# ===========================================================================


def feasible_cost_landscape(
    evals: list[EvaluatedOption],
    conflicts: list[ConflictEdge],
    buckets: list[CapacityBucket],
    *,
    max_combos: int = 500_000,
) -> tuple[np.ndarray, float]:
    """Combined cost of every FEASIBLE one-hot assignment, plus the optimum.

    Enumerates one option per flight (K^F combos), drops any that violate a
    conflict or a sector capacity, and returns the array of surviving costs and
    their minimum. Raises if the instance is too big to enumerate (use sampling
    instead).
    """
    groups: dict[str, list[int]] = {}
    for i, ev in enumerate(evals):
        groups.setdefault(ev.flight_name, []).append(i)
    choice_lists = list(groups.values())

    n_combos = 1
    for cl in choice_lists:
        n_combos *= max(1, len(cl))
    if n_combos > max_combos:
        raise ValueError(
            f"{n_combos} assignments exceeds max_combos={max_combos}; "
            "this exact landscape is a small-instance tool."
        )

    costs = np.array([ev.cost_combined for ev in evals], dtype=float)
    conflict_pairs = [(e.i, e.j) for e in conflicts]
    bucket_members = [(list(b.members), b.capacity) for b in buckets]

    feasible: list[float] = []
    for combo in product(*choice_lists):
        sel = set(combo)
        if any(i in sel and j in sel for i, j in conflict_pairs):
            continue
        if any(sum(1 for m in members if m in sel) > cap for members, cap in bucket_members):
            continue
        feasible.append(float(costs[list(combo)].sum()))

    arr = np.array(feasible, dtype=float)
    optimum = float(arr.min()) if arr.size else float("inf")
    return arr, optimum


# ===========================================================================
# 2. SAMPLER QUALITY — repaired-cost distributions
# ===========================================================================


def _repaired_costs(graph, bit_samples) -> np.ndarray:
    """Repair each raw bitstring into a feasible assignment and return its cost."""
    from .quantum_common import repair_sample

    out: list[float] = []
    for bits in bit_samples:
        idx = repair_sample(graph, bits)
        out.append(float(sum(graph.costs[i] for i in idx)))
    return np.array(out, dtype=float)


def gbs_sample_costs(
    evals: list[EvaluatedOption],
    conflicts: list[ConflictEdge],
    buckets: list[CapacityBucket],
    *,
    n_samples: int = 400,
    seed: int = 0,
) -> np.ndarray:
    """Repaired combined costs from `n_samples` GBS draws (built-in sampler)."""
    from .quantum_common import build_option_graph
    from .xanadu_gbs import DEFAULT_TARGET_NORM, GBSSubsetSampler, encode_option_graph

    graph = build_option_graph(evals, conflicts, buckets)
    encoding = encode_option_graph(graph, target_norm=DEFAULT_TARGET_NORM)
    sampler = GBSSubsetSampler(encoding, np.random.default_rng(seed))
    subsets = sampler.sample(n_samples)

    def to_bits(subset) -> np.ndarray:
        bits = np.zeros(graph.n, dtype=np.uint8)
        for v in subset:
            bits[v] = 1
        return bits

    return _repaired_costs(graph, (to_bits(s) for s in subsets))


def random_sample_costs(
    evals: list[EvaluatedOption],
    conflicts: list[ConflictEdge],
    buckets: list[CapacityBucket],
    *,
    n_samples: int = 400,
    seed: int = 0,
) -> np.ndarray:
    """Baseline: repaired costs from `n_samples` uniform-random bitstrings.

    Same repair pipeline as GBS, so the only difference is the proposal — a fair
    'is the quantum sampler actually doing better than chance?' control.
    """
    from .quantum_common import build_option_graph

    graph = build_option_graph(evals, conflicts, buckets)
    rng = np.random.default_rng(seed)
    bits = (rng.integers(0, 2, size=graph.n).astype(np.uint8) for _ in range(n_samples))
    return _repaired_costs(graph, bits)


# ===========================================================================
# 3. HARDNESS / SCALING — CP-SAT effort vs problem size
# ===========================================================================


def hardness_sweep(
    sizes: list[int],
    *,
    seed: int = 42,
    time_limit_s: float = 5.0,
    beta_contrail: float = 5.0,
) -> dict[str, np.ndarray]:
    """Solve a fresh instance at each `n_flights` in `sizes`; record CP-SAT effort.

    Returns arrays keyed by: sizes, n_options, cpsat_ms (wall clock), incumbents
    (improved solutions CP-SAT found), objective. A flat incumbents=1 line is the
    honest signal that the instance is trivially easy for the exact solver.
    """
    from .scenario import ScenarioConfig, build_scenario_full
    from .solver_cpsat import solve_cpsat

    rows: dict[str, list[float]] = {
        "sizes": [], "n_options": [], "cpsat_ms": [], "incumbents": [], "objective": [],
    }
    for nf in sizes:
        cfg = ScenarioConfig(seed=seed, n_flights=int(nf),
                             beta_contrail=beta_contrail, time_limit_s=time_limit_s)
        _w, _f, evals, conflicts, buckets = build_scenario_full(cfg)
        res = solve_cpsat(evals, conflicts, buckets, time_limit_s=time_limit_s)
        rows["sizes"].append(int(nf))
        rows["n_options"].append(len(evals))
        rows["cpsat_ms"].append(res.wall_clock_s * 1000.0)
        rows["incumbents"].append(res.n_improvements)
        rows["objective"].append(res.objective)
    return {k: np.array(v) for k, v in rows.items()}
