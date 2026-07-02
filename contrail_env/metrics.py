"""
metrics.py — Per-shot benchmark statistics: success probability + TTS.

WHY
===
Approximation ratio measures "how good was the best sample". It cannot tell a
slow-but-reliable sampler apart from a fast-but-lucky one, because it ignores
the *rate* at which a sampler produces optimal shots. The standard fix (Rønnow
et al., *Science* 345, 420, 2014) is the success probability and the derived
time-to-solution:

    p_s        = Pr[one repaired shot hits the exact optimum E*_exact]
    TTS_0.99   = t_shot · ln(1 - 0.99) / ln(1 - p_s)

TTS is the wall-clock time to reach the optimum with 99% confidence, given the
per-shot cost t_shot and the observed p_s. It is heavy-tailed across seeds
(one p_s = 0 seed sends TTS to infinity), so it must be aggregated with the
median + IQR, never a mean — see benchmark.aggregate.

E*_exact IS THE THRESHOLD, NOT THE CP-SAT objective
===================================================
The CP-SAT objective is cent-quantized (solver_cpsat._COST_SCALE = 100), while
repaired sample costs are exact floats on the same arithmetic path. Comparing
against the quantized objective would spuriously miss or admit ties, so the
success threshold uses optimum_exact = sum_i c_i over the CP-SAT choice.
"""

from __future__ import annotations

import math

from .quantum_common import SampleEvaluation


def success_probability(
    evaluation: SampleEvaluation,
    optimum_exact: float,
    *,
    rel_tol: float = 1e-9,
) -> float:
    """Multiplicity-weighted fraction of shots whose repaired cost hits E*.

    A shot "hits" when its repaired cost <= optimum_exact·(1 + rel_tol) + 1e-12
    (a small absolute pad so exact-equal floats always count). Weighted by the
    per-shot multiplicities in evaluation.repaired_unique, so it is a true shot
    fraction, not a unique-assignment fraction.
    """
    threshold = optimum_exact * (1.0 + rel_tol) + 1e-12
    total = sum(mult for _idx, _cost, mult in evaluation.repaired_unique)
    if total == 0:
        return 0.0
    hits = sum(mult for _idx, cost, mult in evaluation.repaired_unique if cost <= threshold)
    return hits / total


def time_to_solution(
    t_shot_s: float,
    p_success: float,
    *,
    quantile: float = 0.99,
) -> float:
    """TTS to reach the optimum with probability `quantile` (Rønnow et al.).

    p_success = 0  -> math.inf  (the optimum is never sampled)
    p_success >= 1 -> t_shot_s  (one shot already suffices)
    """
    if p_success <= 0.0:
        return math.inf
    if p_success >= 1.0:
        return t_shot_s
    return t_shot_s * math.log(1.0 - quantile) / math.log(1.0 - p_success)
