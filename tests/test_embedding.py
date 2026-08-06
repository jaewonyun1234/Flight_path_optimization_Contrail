"""Embedder pipeline: small graphs embed, 7-leaf star fails, refine repairs."""

import numpy as np

from contrail_env import check_embedding, embed, refine_embedding
from contrail_env.embedding_study import NONEDGE_MIN, R_BLOCKADE_UM


def test_triangle_embeds_validly():
    edges = [(0, 1), (0, 2), (1, 2)]
    _coords, report = embed(3, edges)
    assert report.valid, report


def test_five_leaf_star_embeds_validly():
    # 5 leaves within the hub's blockade disk, pairwise outside each
    # other's — a regular pentagon fits, so a strong embedder must find it.
    edges = [(0, leaf) for leaf in range(1, 6)]
    _coords, report = embed(6, edges)
    assert report.valid, report


def test_star_with_seven_leaves_fails():
    # 7 leaves pairwise far apart cannot pack inside one blockade disk in
    # 2D. With refinement + restarts behind embed(), a failure here means
    # GEOMETRY is the obstruction, not embedder weakness.
    edges = [(0, leaf) for leaf in range(1, 8)]
    _coords, report = embed(8, edges)
    assert not report.valid


def test_embed_deterministic_given_seed():
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
    coords_a, report_a = embed(5, edges, seed=7)
    coords_b, report_b = embed(5, edges, seed=7)
    assert np.array_equal(coords_a, coords_b)
    assert report_a == report_b


def test_refine_strictly_reduces_violations():
    # Valid triangle, then one node dragged far away: 2 missing edges.
    edges = [(0, 1), (0, 2), (1, 2)]
    coords = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    assert check_embedding(coords, edges).valid
    coords[2] = [3.0, 40.0]
    before = check_embedding(coords, edges)
    assert before.missing_edges + before.spurious_edges == 2
    refined = refine_embedding(coords, 3, edges)
    after = check_embedding(refined, edges)
    assert (after.missing_edges + after.spurious_edges
            < before.missing_edges + before.spurious_edges)


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
