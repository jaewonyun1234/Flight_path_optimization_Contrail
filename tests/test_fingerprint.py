"""Constraint-violation fingerprint of raw pre-repair samples (§S5).

The arithmetic tests use the exact toy graph from the spec (two flights of two
options each, one conflict edge, one over-constrained capacity bucket) so the
deficit/excess/conflict/overflow numbers can be checked by hand. That bucket
(cap=1 spanning all 4 options, while both flights must pick one each) makes
every one-hot-satisfying assignment violate capacity by construction — so the
"feasible batch" / "repair distance zero" tests use a second, adequately-
capacitated graph instead (deficit/excess/conflict/overflow cannot all be zero
on the first graph while still satisfying one-hot).
"""

import numpy as np
import pytest

from contrail_env.analysis import ground_state_degeneracy
from contrail_env.fingerprint import fingerprint_bitstrings
from contrail_env.quantum_common import OptionGraph


def _graph():
    costs = np.array([1.0, 2.0, 3.0, 4.0])
    return OptionGraph(
        n=4, costs=costs, weights=costs.max() - costs + 1.0,
        groups={"A": (0, 1), "B": (2, 3)},
        conflict_edges=((0, 2),),
        buckets=(((0, 1, 2, 3), 1),),
    )


def _feasible_graph():
    # Same groups/conflict edge, but capacity 2 (enough for one pick per
    # flight) so a genuinely feasible one-hot assignment exists.
    costs = np.array([1.0, 2.0, 3.0, 4.0])
    return OptionGraph(
        n=4, costs=costs, weights=costs.max() - costs + 1.0,
        groups={"A": (0, 1), "B": (2, 3)},
        conflict_edges=(),
        buckets=(((0, 1, 2, 3), 2),),
    )


def test_all_zeros():
    fp = fingerprint_bitstrings(_graph(), np.array([[0, 0, 0, 0]], dtype=np.uint8))
    assert fp.onehot_deficit.mean == 2
    assert fp.onehot_excess.mean == 0
    assert fp.conflict.mean == 0
    assert fp.capacity_overflow.mean == 0


def test_all_ones():
    fp = fingerprint_bitstrings(_graph(), np.array([[1, 1, 1, 1]], dtype=np.uint8))
    assert fp.onehot_deficit.mean == 0
    assert fp.onehot_excess.mean == 2
    assert fp.conflict.mean == 1
    assert fp.capacity_overflow.mean == 3


def test_mixed_assignment():
    fp = fingerprint_bitstrings(_graph(), np.array([[1, 0, 1, 0]], dtype=np.uint8))
    assert fp.onehot_deficit.mean == 0
    assert fp.onehot_excess.mean == 0
    assert fp.conflict.mean == 1
    assert fp.capacity_overflow.mean == 1


def test_shares_sum_to_one_when_violations_exist():
    fp = fingerprint_bitstrings(_graph(), np.array([[1, 1, 1, 1]], dtype=np.uint8))
    assert fp.shares
    assert sum(fp.shares.values()) == pytest.approx(1.0)


def test_shares_empty_on_feasible_batch():
    bits = np.array([[1, 0, 1, 0]], dtype=np.uint8)  # one pick per flight, within capacity
    fp = fingerprint_bitstrings(_feasible_graph(), bits)
    assert fp.onehot_deficit.mean == 0
    assert fp.onehot_excess.mean == 0
    assert fp.conflict.mean == 0
    assert fp.capacity_overflow.mean == 0
    assert fp.shares == {}


def test_repair_distance_zero_for_feasible():
    bits = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    fp = fingerprint_bitstrings(_feasible_graph(), bits)
    assert fp.repair_distance.mean == 0


def test_multiplicity_weighting():
    # Two copies of an all-zeros row and one all-ones row -> weighted mean.
    bits = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.uint8)
    fp = fingerprint_bitstrings(_graph(), bits)
    assert fp.n_samples == 3
    # deficit: (2*2 + 1*0) / 3
    assert fp.onehot_deficit.mean == pytest.approx(4 / 3)


def test_degeneracy_counts_ties():
    costs = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 3.0])
    assert ground_state_degeneracy(costs, optimum=1.0) == 4
