"""Unit-disk embedder: triangle works, 7-leaf star fails, spurious counted."""

import numpy as np

from contrail_env import check_embedding, greedy_embedding
from contrail_env.embedding_study import NONEDGE_MIN, R_BLOCKADE_UM


def test_triangle_embeds_validly():
    edges = [(0, 1), (0, 2), (1, 2)]
    coords = greedy_embedding(3, edges)
    report = check_embedding(coords, edges)
    assert report.valid, report


def test_star_with_seven_leaves_fails():
    # Hub 0 conflicts with 7 leaves that do not conflict with each other:
    # all leaves must sit within the blockade disk of the hub yet pairwise
    # OUTSIDE each other's disks — geometrically impossible in 2D for 7.
    edges = [(0, leaf) for leaf in range(1, 8)]
    coords = greedy_embedding(8, edges)
    report = check_embedding(coords, edges)
    assert not report.valid


def test_check_embedding_counts_one_spurious_edge():
    # Two non-adjacent nodes deliberately placed inside each other's
    # blockade radius (but above hardware min spacing) = 1 spurious edge.
    d = 0.9 * NONEDGE_MIN * R_BLOCKADE_UM
    coords = np.array([[0.0, 0.0], [d, 0.0], [100.0, 0.0]])
    report = check_embedding(coords, edges=[])
    assert report.spurious_edges == 1
    assert report.missing_edges == 0
    assert not report.valid


def test_far_apart_non_edges_are_valid():
    coords = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
    report = check_embedding(coords, edges=[])
    assert report.valid
    assert report.distortion == 1.0
