"""
fingerprint.py — Constraint-violation fingerprint of raw pre-repair samples (§S5).

WHY
===
Both quantum platforms embed the SAME QUBO but fail its constraints
differently. This module MEASURES that; the physical hypotheses live only in
docstrings and the G3 tab, never in test assertions:

    Pasqal (blockade):  the Rydberg blockade makes "<= 1 per one-hot clique"
                        nearly hard (U ~ 55.4 rad/us), while "= 1" must be paid
                        for by delta(t) -> predicted failure mode is
                        UNDER-selection (empty flight groups = one-hot deficit).
    Xanadu (GBS):       |Haf(A_S)|^2 weighting favors dense subsets ->
                        predicted failure mode is OVER-selection (one-hot
                        excess, capacity overflow).

DEFINITIONS (per raw pre-repair sample b in {0,1}^n)
=====================================================
    one-hot deficit    = #{f : sum_{k in f} b_k = 0}
    one-hot excess     = sum_f max(0, sum_{k in f} b_k - 1)
                          (on Pasqal this IS the blockade-break count)
    conflict violations = sum_{(i,j) in graph.conflict_edges} b_i b_j
                          (conflict edges ONLY — same-flight pairs are already
                          counted as excess; no double counting)
    capacity overflow  = sum_b max(0, occ_b - cap_b)
    repair distance    = Hamming distance between b and the FULL repaired
                          bitstring x_rep(b) (repair_sample(graph, b))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quantum_common import OptionGraph, repair_sample


@dataclass(frozen=True)
class ViolationStats:
    """Multiplicity-weighted summary of one violation family over a batch."""

    mean: float
    std: float
    max: float
    rate_nonzero: float


@dataclass(frozen=True)
class Fingerprint:
    """The five-family constraint-violation decomposition of one sample batch."""

    n_samples: int
    onehot_deficit: ViolationStats
    onehot_excess: ViolationStats
    conflict: ViolationStats
    capacity_overflow: ViolationStats
    repair_distance: ViolationStats
    shares: dict[str, float]   # family mean / sum of the four violation means; {} if all zero


def _onehot_deficit_excess(graph: OptionGraph, bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized deficit/excess per (unique) row, one pass per flight group."""
    s = bits.shape[0]
    deficit = np.zeros(s, dtype=np.int64)
    excess = np.zeros(s, dtype=np.int64)
    for members in graph.groups.values():
        occ = bits[:, list(members)].astype(np.int64).sum(axis=1)
        deficit += (occ == 0).astype(np.int64)
        excess += np.maximum(0, occ - 1)
    return deficit, excess


def _conflict_violations(graph: OptionGraph, bits: np.ndarray) -> np.ndarray:
    s = bits.shape[0]
    viol = np.zeros(s, dtype=np.int64)
    for i, j in graph.conflict_edges:
        viol += bits[:, i].astype(np.int64) * bits[:, j].astype(np.int64)
    return viol


def _capacity_overflow(graph: OptionGraph, bits: np.ndarray) -> np.ndarray:
    s = bits.shape[0]
    overflow = np.zeros(s, dtype=np.int64)
    for members, cap in graph.buckets:
        occ = bits[:, list(members)].astype(np.int64).sum(axis=1)
        overflow += np.maximum(0, occ - cap)
    return overflow


def _repair_distance(graph: OptionGraph, bits: np.ndarray) -> np.ndarray:
    """Hamming distance to the full repaired bitstring, one repair per row."""
    n = graph.n
    out = np.empty(bits.shape[0], dtype=np.int64)
    for i, row in enumerate(bits):
        indices = repair_sample(graph, row)
        rep = np.zeros(n, dtype=np.uint8)
        rep[indices] = 1
        out[i] = int(np.sum(row != rep))
    return out


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> ViolationStats:
    total = float(weights.sum())
    if total == 0.0:
        return ViolationStats(mean=0.0, std=0.0, max=0.0, rate_nonzero=0.0)
    values = values.astype(np.float64)
    mean = float(np.average(values, weights=weights))
    var = float(np.average((values - mean) ** 2, weights=weights))
    rate_nonzero = float(weights[values > 0].sum() / total)
    return ViolationStats(mean=mean, std=var**0.5, max=float(values.max()), rate_nonzero=rate_nonzero)


def fingerprint_bitstrings(graph: OptionGraph, bits: np.ndarray) -> Fingerprint:
    """Decompose a batch of RAW (pre-repair) samples into the five families.

    Prices repair once per unique row (dedup, same pattern as
    quantum_common.evaluate_samples), then expands every statistic by
    multiplicity — so duplicate shots don't pay for duplicate repairs.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n_samples = bits.shape[0]

    unique: dict[bytes, tuple[np.ndarray, int]] = {}
    for row in bits:
        key = row.tobytes()
        if key in unique:
            u_row, count = unique[key]
            unique[key] = (u_row, count + 1)
        else:
            unique[key] = (row, 1)

    if unique:
        u_bits = np.stack([row for row, _c in unique.values()])
        weights = np.array([c for _r, c in unique.values()], dtype=np.int64)
    else:
        u_bits = np.zeros((0, graph.n), dtype=np.uint8)
        weights = np.zeros(0, dtype=np.int64)

    deficit, excess = _onehot_deficit_excess(graph, u_bits)
    conflict = _conflict_violations(graph, u_bits)
    overflow = _capacity_overflow(graph, u_bits)
    distance = _repair_distance(graph, u_bits)

    onehot_deficit = _weighted_stats(deficit, weights)
    onehot_excess = _weighted_stats(excess, weights)
    conflict_stats = _weighted_stats(conflict, weights)
    capacity_stats = _weighted_stats(overflow, weights)
    repair_stats = _weighted_stats(distance, weights)

    means = {
        "onehot_deficit": onehot_deficit.mean,
        "onehot_excess": onehot_excess.mean,
        "conflict": conflict_stats.mean,
        "capacity_overflow": capacity_stats.mean,
    }
    total_mean = sum(means.values())
    shares = {k: v / total_mean for k, v in means.items()} if total_mean > 0.0 else {}

    return Fingerprint(
        n_samples=n_samples,
        onehot_deficit=onehot_deficit,
        onehot_excess=onehot_excess,
        conflict=conflict_stats,
        capacity_overflow=capacity_stats,
        repair_distance=repair_stats,
        shares=shares,
    )


def fingerprint_to_flat_dict(fp: Fingerprint) -> dict[str, float]:
    """Flatten a Fingerprint into a meta-dict-friendly {name: float} mapping."""
    out: dict[str, float] = {"n_samples": float(fp.n_samples)}
    for name, stats in (
        ("onehot_deficit", fp.onehot_deficit),
        ("onehot_excess", fp.onehot_excess),
        ("conflict", fp.conflict),
        ("capacity_overflow", fp.capacity_overflow),
        ("repair_distance", fp.repair_distance),
    ):
        out[f"{name}_mean"] = stats.mean
        out[f"{name}_std"] = stats.std
        out[f"{name}_max"] = stats.max
        out[f"{name}_rate_nonzero"] = stats.rate_nonzero
    for name, share in fp.shares.items():
        out[f"{name}_share"] = share
    return out
