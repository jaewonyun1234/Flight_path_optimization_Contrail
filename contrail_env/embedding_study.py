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
The embedder is a PIPELINE: a greedy one-pass initializer, a
force-directed refinement stage (springs pull violated edges together,
push spurious pairs apart), and seeded multi-start restarts with
Gaussian jitter (`embed` below). When all restarts fail, that still
does NOT prove the graph is not unit-disk-embeddable — a cleverer
placement might exist. The reported quantity is "fraction embeddable BY
THIS EMBEDDER", which is the practically relevant number for the
hardware pipeline (it is the same routine the solver uses to build
registers). The known geometric obstruction in 2D is the high-degree
star: one option conflicting with many mutually-non-conflicting options
cannot pack around a single atom, so the study logs `max_degree` (on
the full independence graph — the graph actually embedded) and
`n_nodes_deg_ge_6`, the count of nodes past the packing limit.

A SECOND, subtler obstruction shows up at LOW degree: when two flights
conflict at ALL K levels, their two one-hot triangles plus the K
matching edges form a triangular prism (K=3). With the safety margins
(edges <= 0.95 R_b, non-edges >= 1.05 R_b, 5 um floor) the prism has no
valid 2D placement: the aligned-triangles case is provably impossible
(it would need a triangle side > the blockade radius), and 500-restart
searches never find any other. Shared-level conflict triangles between
three flights behave the same way. The study therefore also logs
`n_all_level_pairs` (flight pairs conflicting at every level) so a
failure can be attributed: high degree, prism-like structure, or
genuinely unexplained.

Purely classical — numpy only, nothing exponential. Runnable as:

    python -m contrail_env.embedding_study --csv embedding.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from .problem import Scenario, make_scenario

# Hardware-flavored geometry defaults (Pulser AnalogDevice scale, in um).
R_BLOCKADE_UM = 8.0
MIN_ATOM_DISTANCE_UM = 5.0
# Safety margins: edges must be clearly inside the radius, non-edges
# clearly outside, so blockade strength differences stay decisive.
EDGE_MAX = 0.95      # edge pairs must satisfy  d <= EDGE_MAX * R_b
NONEDGE_MIN = 1.05   # non-edge pairs must satisfy  d >= NONEDGE_MIN * R_b


def all_level_conflict_pairs(scenario: Scenario) -> int:
    """Flight pairs that conflict at EVERY level (prism obstruction).

    Two such flights' one-hot triangles plus the level-matching edges
    form a triangular prism, which does not embed under the margin
    rules — a local geometric obstruction at max degree 3.
    """
    conf = set(scenario.conflicts)
    k = scenario.n_options
    return sum(
        1
        for f in range(scenario.n_flights)
        for g in range(f + 1, scenario.n_flights)
        if all((f * k + level, g * k + level) in conf for level in range(k))
    )


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
        valid:           every constraint satisfied
        missing_edges:   edge pairs placed too far apart (blockade lost)
        spurious_edges:  non-edge pairs placed too close (fake blockade)
        crowded_pairs:   pairs closer than the hardware minimum spacing
        distortion:      worst violation ratio (1.0 when valid); e.g. a
                         missing edge at distance d scores d / (EDGE_MAX * R_b)
        n_restarts_used: multi-start attempts consumed by embed() (0 when
                         the report came from a bare check_embedding call)
        refine_iters:    total refinement iterations across those attempts
    """

    valid: bool
    missing_edges: int
    spurious_edges: int
    crowded_pairs: int
    distortion: float
    n_restarts_used: int = 0
    refine_iters: int = 0


def _pair_masks(
    coords: np.ndarray, edges: Iterable[tuple[int, int]], r_blockade: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(dist, E, missing, spurious, crowded) as n x n matrices.

    `dist` has +inf on the diagonal; the three violation masks are
    symmetric, so every violated PAIR appears twice (divide sums by 2).
    """
    n = len(coords)
    E = np.zeros((n, n), dtype=bool)
    for i, j in edges:
        E[i, j] = E[j, i] = True
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    missing = E & (dist > EDGE_MAX * r_blockade)
    spurious = (~E) & (dist < NONEDGE_MIN * r_blockade)
    crowded = dist < MIN_ATOM_DISTANCE_UM
    return dist, E, missing, spurious, crowded


def check_embedding(
    coords: np.ndarray,
    edges: Iterable[tuple[int, int]],
    r_blockade: float = R_BLOCKADE_UM,
) -> EmbeddingReport:
    """Verify both disk-graph conditions for every pair of nodes."""
    coords = np.asarray(coords, dtype=float)
    edges = list(edges)
    if not np.isfinite(coords).all():
        # NaN/inf coordinates would make every comparison False and thus
        # look "valid" — report a diverged placement as maximally broken.
        n_edges = len({(min(i, j), max(i, j)) for i, j in edges})
        return EmbeddingReport(
            valid=False,
            missing_edges=n_edges,
            spurious_edges=0,
            crowded_pairs=0,
            distortion=float("inf"),
        )
    dist, _e, missing, spurious, crowded = _pair_masks(coords, edges, r_blockade)
    distortion = 1.0
    if missing.any():
        distortion = max(distortion, float(dist[missing].max()) / (EDGE_MAX * r_blockade))
    if spurious.any():
        distortion = max(
            distortion, (NONEDGE_MIN * r_blockade) / max(float(dist[spurious].min()), 1e-9)
        )
    if crowded.any():
        distortion = max(
            distortion, MIN_ATOM_DISTANCE_UM / max(float(dist[crowded].min()), 1e-9)
        )
    n_missing = int(missing.sum()) // 2
    n_spurious = int(spurious.sum()) // 2
    n_crowded = int(crowded.sum()) // 2
    return EmbeddingReport(
        valid=(n_missing == 0 and n_spurious == 0 and n_crowded == 0),
        missing_edges=n_missing,
        spurious_edges=n_spurious,
        crowded_pairs=n_crowded,
        distortion=distortion,
    )


# =============================================================================
# REFINEMENT + MULTI-START — the embedder pipeline entry point
# =============================================================================

def _refine_counted(
    coords: np.ndarray,
    n: int,
    edges: list[tuple[int, int]],
    r_blockade: float,
    n_iters: int,
    step: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    """Force-directed repair; returns (coords, iterations actually run)."""
    coords = coords.copy()
    target_edge = 0.8 * EDGE_MAX * r_blockade
    target_non = 1.1 * NONEDGE_MIN * r_blockade
    best_viol = np.inf
    best_it = 0
    for it in range(n_iters):
        dist, _e, missing, spurious, crowded = _pair_masks(coords, edges, r_blockade)
        n_viol = int(missing.sum() + spurious.sum() + crowded.sum())
        if n_viol == 0:
            return coords, it
        if n_viol < best_viol:
            best_viol, best_it = n_viol, it
        elif it - best_it > 100:
            return coords, it  # stagnated; let the caller restart instead
        # Unit vectors i -> away from j; coincident pairs get a fixed
        # deterministic direction. It must be ANTISYMMETRIC (i pushed one
        # way, j the other) or coincident nodes move in lockstep forever.
        diff = coords[:, None, :] - coords[None, :, :]
        safe = np.where(np.isfinite(dist) & (dist > 1e-9), dist, 1.0)
        unit = diff / safe[..., None]
        zero_pair = np.isfinite(dist) & (dist <= 1e-9)
        if zero_pair.any():
            sign = np.sign(np.arange(n)[:, None] - np.arange(n)[None, :])
            unit[zero_pair] = np.stack(
                [sign[zero_pair], np.zeros(int(zero_pair.sum()))], axis=1
            )

        mag = np.zeros_like(dist)
        # Edge too far: pull i toward j (negative = toward).
        mag[missing] = -(dist[missing] - target_edge)
        # Non-edge too close: push i away from j.
        mag[spurious] = target_non - dist[spurious]
        # Below the hardware floor: strong extra repulsion.
        mag[crowded] += 2.0 * (1.2 * MIN_ATOM_DISTANCE_UM - np.where(
            np.isfinite(dist), dist, 0.0
        )[crowded])
        disp = step * (mag[..., None] * unit).sum(axis=1)
        # Cap each node's move per iteration: uncapped spring sums between
        # far-apart clusters overshoot, oscillate, and diverge to inf.
        norms = np.linalg.norm(disp, axis=1, keepdims=True)
        cap = 0.5 * r_blockade
        disp *= np.where(norms > cap, cap / np.maximum(norms, 1e-12), 1.0)
        # Annealed kicks (seeded, decaying to zero): pure spring dynamics
        # stalls in tug-of-war equilibria where attraction and repulsion
        # balance while violations remain; noise early, convergence late.
        if rng is not None:
            sigma = 0.25 * r_blockade * (1.0 - it / n_iters) ** 2
            disp += rng.normal(0.0, sigma, size=coords.shape)
        coords = coords + disp
    return coords, n_iters


def refine_embedding(
    coords: np.ndarray,
    n: int,
    edges: list[tuple[int, int]],
    r_blockade: float = R_BLOCKADE_UM,
    n_iters: int = 300,
    step: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Deterministic force-directed repair of a placement.

    Per iteration, every violated pair contributes a spring force:
    missing edges attract toward 0.8 * EDGE_MAX * R_b, spurious pairs
    repel to 1.1 * NONEDGE_MIN * R_b, pairs under the hardware floor get
    a strong extra push. Early-stops as soon as the placement is valid.
    With `rng` set, small seeded annealing kicks (decaying to zero) are
    added while violations persist — deterministic given the rng, and
    necessary to escape equilibria where the spring forces balance.
    """
    refined, _iters = _refine_counted(coords, n, edges, r_blockade, n_iters, step, rng)
    return refined


def embed(
    n: int,
    edges: list[tuple[int, int]],
    r_blockade: float = R_BLOCKADE_UM,
    n_restarts: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, EmbeddingReport]:
    """Multi-start embedder: greedy init + refinement + seeded jitter.

    Restart 0 refines the greedy layout as-is; restarts 1..k add small
    seeded Gaussian jitter to the greedy layout first. Returns the first
    VALID result, else the best attempt (fewest missing + spurious +
    crowded, tie-break lower distortion). Fully deterministic given
    `seed`. This is the single entry point used by both the solver
    register builder and the scaling study.
    """
    base = greedy_embedding(n, edges, r_blockade)
    best_coords = base
    best_report: EmbeddingReport | None = None
    best_key: tuple[int, float] | None = None
    total_iters = 0
    used = 0
    for restart in range(n_restarts):
        rng = np.random.default_rng([seed, restart])
        start = base
        if restart > 0:
            start = base + rng.normal(0.0, 0.35 * r_blockade, size=base.shape)
        coords, iters = _refine_counted(start, n, edges, r_blockade, 300, 0.15, rng)
        total_iters += iters
        used = restart + 1
        report = check_embedding(coords, edges, r_blockade)
        key = (
            report.missing_edges + report.spurious_edges + report.crowded_pairs,
            report.distortion,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_coords = coords
            best_report = report
        if report.valid:
            break
    assert best_report is not None
    return best_coords, replace(
        best_report, n_restarts_used=used, refine_iters=total_iters
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
    """Embed every (flight_count, seed) scenario and record the outcome.

    Degree diagnostics are computed on the FULL independence graph
    (conflicts + one-hot cliques) — the graph actually embedded — not on
    conflicts alone. `n_nodes_deg_ge_6` counts nodes past the 2D packing
    limit: >= 6 mutually-far neighbors cannot surround one atom, so a
    failure with such nodes present has a local, provable cause.
    """
    rows: list[dict[str, object]] = []
    for n_flights in flight_counts:
        for seed in seeds:
            scenario = make_scenario(n_flights, n_options, seed)
            n = scenario.n_vars
            edges = independence_edges(scenario)
            degree = np.zeros(n, dtype=int)
            for i, j in edges:
                degree[i] += 1
                degree[j] += 1
            _coords, report = embed(n, edges, seed=seed)
            rows.append({
                "n_flights": n_flights,
                "n_vars": n,
                "n_conflict_edges": len(scenario.conflicts),
                "edge_density": (
                    len(scenario.conflicts) / (n * (n - 1) / 2) if n > 1 else 0.0
                ),
                "max_degree": int(degree.max()) if n else 0,
                "n_nodes_deg_ge_6": int((degree >= 6).sum()),
                "n_all_level_pairs": all_level_conflict_pairs(scenario),
                "embed_success": report.valid,
                "missing_edges": report.missing_edges,
                "spurious_edges": report.spurious_edges,
                "distortion": round(report.distortion, 4),
                "n_restarts_used": report.n_restarts_used,
                "refine_iters": report.refine_iters,
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
        failures = [r for r in sub if not r["embed_success"]]
        crowded = sum(1 for r in failures if int(str(r["n_nodes_deg_ge_6"])) > 0)
        prism = sum(
            1 for r in failures
            if int(str(r["n_nodes_deg_ge_6"])) == 0
            and int(str(r["n_all_level_pairs"])) > 0
        )
        other = len(failures) - crowded - prism
        print(f"F = {n_flights:3d}: {ok}/{len(sub)} embedded | failures: "
              f"{crowded} deg>=6, {prism} prism-only, {other} unexplained")
    print(f"wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
