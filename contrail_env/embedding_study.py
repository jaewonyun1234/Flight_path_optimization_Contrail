"""
embedding_study.py — Classical unit-disk embeddability study.

THE QUESTION
============
Pasqal's neutral-atom hardware realizes a graph by PLACING atoms in the
plane: two atoms interact (blockade) iff they sit within the blockade
radius R_b. So a problem graph is only directly runnable if it is a
UNIT-DISK GRAPH: coordinates must exist where every edge pair is closer
than R_b and every non-edge pair is farther. This module asks, with no
quantum simulation at all: as the flight count grows, what fraction of
our contrail conflict graphs still admit such a placement?

WHAT IS EMBEDDED
================
The full independence graph of a scenario: the cross-flight ISSR
conflicts PLUS each flight's one-hot clique (its K options are mutually
exclusive, which the blockade must also enforce).

HONEST LIMITATION (read this)
=============================
The embedder below is GREEDY: it places one node at a time and never
backtracks. When it fails, that does NOT prove the graph is not
unit-disk-embeddable — a cleverer placement might exist. The reported
quantity is "fraction embeddable BY THIS EMBEDDER", which is the
practically relevant number for the hardware pipeline (it is the same
routine the solver uses to build registers). The known geometric
obstruction in 2D is the high-degree star: one option conflicting with
many mutually-non-conflicting options cannot pack around a single atom,
so `max_degree` is logged for every instance.

Purely classical — numpy only, nothing exponential. Runnable as:

    python -m contrail_env.embedding_study --csv embedding.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .problem import Scenario, make_scenario

# Hardware-flavored geometry defaults (Pulser AnalogDevice scale, in um).
R_BLOCKADE_UM = 8.0
MIN_ATOM_DISTANCE_UM = 5.0
# Safety margins: edges must be clearly inside the radius, non-edges
# clearly outside, so blockade strength differences stay decisive.
EDGE_MAX = 0.95      # edge pairs must satisfy  d <= EDGE_MAX * R_b
NONEDGE_MIN = 1.05   # non-edge pairs must satisfy  d >= NONEDGE_MIN * R_b


def independence_edges(scenario: Scenario) -> list[tuple[int, int]]:
    """Conflict edges plus the one-hot cliques (same-flight pairs).

    An independent set of this graph is a partial assignment: at most one
    option per flight, no two conflicting options. This is the edge set
    the Rydberg blockade must enforce.
    """
    edges = list(scenario.conflicts)
    for members in scenario.groups():
        for a_pos, a in enumerate(members):
            for b in members[a_pos + 1:]:
                edges.append((a, b))
    return edges


# =============================================================================
# GREEDY EMBEDDER — single source of truth (pasqal_analog imports this)
# =============================================================================

def greedy_embedding(
    n: int,
    edges: Iterable[tuple[int, int]],
    r_blockade: float = R_BLOCKADE_UM,
) -> np.ndarray:
    """Attempt a 2D placement of an n-node graph as a unit-disk register.

    Greedy, deterministic, never backtracks: nodes are placed in
    descending-degree order (hardest first); each node tries a fixed fan
    of candidate positions around the centroid of its already-placed
    neighbors and keeps the candidate violating the fewest constraints.
    A node with no placed neighbors starts a new cluster far from
    everything. Always returns coordinates — call `check_embedding` to
    find out whether they are actually valid.
    """
    edge_set = {(min(i, j), max(i, j)) for i, j in edges}
    neighbors: list[set[int]] = [set() for _ in range(n)]
    for i, j in edge_set:
        neighbors[i].add(j)
        neighbors[j].add(i)

    # Fixed candidate fan: radii x angles around the neighbor centroid.
    radii = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80]) * r_blockade
    angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    fan = np.array(
        [[r * math.cos(a), r * math.sin(a)] for r in radii for a in angles]
    )

    order = sorted(range(n), key=lambda v: (-len(neighbors[v]), v))
    coords = np.zeros((n, 2))
    placed: list[int] = []
    cluster_x = 0.0  # next free x position for a fresh cluster

    for v in order:
        placed_nbrs = [u for u in placed if u in neighbors[v]]
        if not placed_nbrs:
            # New cluster: far enough that it cannot touch anything placed.
            coords[v] = (cluster_x, 0.0)
            cluster_x += 3.0 * r_blockade
            placed.append(v)
            continue

        centroid = coords[placed_nbrs].mean(axis=0)
        candidates = centroid[None, :] + fan
        placed_xy = coords[placed]
        is_nbr = np.array([u in neighbors[v] for u in placed])
        # Distances: candidates x placed nodes, all at once.
        d = np.linalg.norm(candidates[:, None, :] - placed_xy[None, :, :], axis=2)
        too_far = is_nbr[None, :] & (d > EDGE_MAX * r_blockade)
        too_close = (~is_nbr[None, :]) & (d < NONEDGE_MIN * r_blockade)
        crowded = d < MIN_ATOM_DISTANCE_UM
        violations = (too_far | too_close | crowded).sum(axis=1)
        coords[v] = candidates[int(np.argmin(violations))]
        placed.append(v)
        cluster_x = max(cluster_x, coords[v, 0] + 3.0 * r_blockade)

    return coords


# =============================================================================
# VALIDATION
# =============================================================================

@dataclass(frozen=True)
class EmbeddingReport:
    """Pairwise verification of a placement against the disk-graph rules.

    Attributes:
        valid:          every constraint satisfied
        missing_edges:  edge pairs placed too far apart (blockade lost)
        spurious_edges: non-edge pairs placed too close (fake blockade)
        crowded_pairs:  pairs closer than the hardware minimum spacing
        distortion:     worst violation ratio (1.0 when valid); e.g. a
                        missing edge at distance d scores d / (EDGE_MAX * R_b)
    """

    valid: bool
    missing_edges: int
    spurious_edges: int
    crowded_pairs: int
    distortion: float


def check_embedding(
    coords: np.ndarray,
    edges: Iterable[tuple[int, int]],
    r_blockade: float = R_BLOCKADE_UM,
) -> EmbeddingReport:
    """Verify both disk-graph conditions for every pair of nodes."""
    n = len(coords)
    edge_set = {(min(i, j), max(i, j)) for i, j in edges}
    missing = 0
    spurious = 0
    crowded = 0
    distortion = 1.0
    for i in range(n):
        d_row = np.linalg.norm(coords[i + 1:] - coords[i], axis=1)
        for off, d in enumerate(d_row):
            j = i + 1 + off
            dist = float(d)
            if (i, j) in edge_set:
                if dist > EDGE_MAX * r_blockade:
                    missing += 1
                    distortion = max(distortion, dist / (EDGE_MAX * r_blockade))
            elif dist < NONEDGE_MIN * r_blockade:
                spurious += 1
                distortion = max(
                    distortion, (NONEDGE_MIN * r_blockade) / max(dist, 1e-9)
                )
            if dist < MIN_ATOM_DISTANCE_UM:
                crowded += 1
                distortion = max(distortion, MIN_ATOM_DISTANCE_UM / max(dist, 1e-9))
    return EmbeddingReport(
        valid=(missing == 0 and spurious == 0 and crowded == 0),
        missing_edges=missing,
        spurious_edges=spurious,
        crowded_pairs=crowded,
        distortion=distortion,
    )


# =============================================================================
# THE STUDY
# =============================================================================

DEFAULT_FLIGHT_COUNTS = (4, 6, 8, 12, 16, 24, 32, 48, 64, 96)


def run_embedding_study(
    flight_counts: Sequence[int] = DEFAULT_FLIGHT_COUNTS,
    n_options: int = 3,
    seeds: Sequence[int] = tuple(range(5)),
    csv_path: str = "embedding.csv",
) -> list[dict[str, object]]:
    """Embed every (flight_count, seed) scenario and record the outcome."""
    rows: list[dict[str, object]] = []
    for n_flights in flight_counts:
        for seed in seeds:
            scenario = make_scenario(n_flights, n_options, seed)
            n = scenario.n_vars
            conflicts = scenario.conflicts
            degree = np.zeros(n, dtype=int)
            for i, j in conflicts:
                degree[i] += 1
                degree[j] += 1
            edges = independence_edges(scenario)
            report = check_embedding(greedy_embedding(n, edges), edges)
            rows.append({
                "n_flights": n_flights,
                "n_vars": n,
                "n_conflict_edges": len(conflicts),
                "edge_density": (
                    len(conflicts) / (n * (n - 1) / 2) if n > 1 else 0.0
                ),
                "max_degree": int(degree.max()) if n else 0,
                "embed_success": report.valid,
                "missing_edges": report.missing_edges,
                "spurious_edges": report.spurious_edges,
                "distortion": round(report.distortion, 4),
            })
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--csv", default="embedding.csv")
    parser.add_argument("--options", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--flights",
        default=",".join(str(f) for f in DEFAULT_FLIGHT_COUNTS),
        help="comma-separated flight counts",
    )
    args = parser.parse_args()
    counts = [int(tok) for tok in args.flights.split(",")]
    rows = run_embedding_study(
        counts, args.options, tuple(range(args.seeds)), args.csv
    )
    for n_flights in counts:
        sub = [r for r in rows if r["n_flights"] == n_flights]
        ok = sum(1 for r in sub if r["embed_success"])
        print(f"F = {n_flights:3d}: {ok}/{len(sub)} embedded")
    print(f"wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
