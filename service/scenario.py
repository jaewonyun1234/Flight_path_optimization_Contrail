"""
scenario.py — Shared scenario construction (server + GUI use the same logic).

A ScenarioConfig is fully seeded, so building from it is deterministic: the
server solves exactly the problem the GUI draws. Keeping this in one place
guarantees the two never drift apart.

Per-flight initial flight level
===============================
`build_random_flights` puts every flight at one initial FL. Here we give each
flight its OWN baseline cruise level, drawn (seeded) from the grid's FL bands,
so the scenario spans several altitudes instead of one flat layer — more
realistic, and it makes the 3D trajectory view actually show altitude.
"""

from __future__ import annotations

import numpy as np

from contrail_env import (
    CapacityBucket,
    ConflictEdge,
    EvaluatedOption,
    Flight,
    World,
    build_and_evaluate_flight,
    build_baseline_profile,
    build_capacity_buckets,
    build_conflict_graph,
    build_random_flights,
    default_european_world,
)

from .generated import solver_pb2

# Used when a client leaves all three cost weights at 0 (proto3 cannot tell
# "unset" from "0"): fall back to the env defaults instead of all-costs-zero.
_DEFAULT_WEIGHTS = (1.0, 5.0, 0.5)


def cost_weights(cfg: solver_pb2.ScenarioConfig) -> tuple[float, float, float]:
    """(alpha_fuel, beta_contrail, gamma_disruption), with an all-zero fallback."""
    weights = (cfg.alpha_fuel, cfg.beta_contrail, cfg.gamma_disruption)
    if weights == (0.0, 0.0, 0.0):
        return _DEFAULT_WEIGHTS
    return weights


def build_world_and_flights(cfg: solver_pb2.ScenarioConfig) -> tuple[World, list[Flight]]:
    """Build the world and flights for a config, each flight at its own FL."""
    world = default_european_world(seed=cfg.seed, n_issr_blobs=cfg.n_issr_blobs)
    flights = build_random_flights(
        n_flights=cfg.n_flights,
        world=world,
        seed=cfg.seed,
        corridor_frac=cfg.corridor_frac or 0.05,
        snapshot_window_s=(0.0, cfg.snapshot_window_s or 300.0),
    )

    # Give each flight a randomized (but seed-deterministic) baseline FL.
    rng = np.random.default_rng(int(cfg.seed) * 1000 + 7)
    fl_bands = world.grid.fl_bands
    for flight in flights:
        fl = int(rng.choice(fl_bands))
        flight.baseline = build_baseline_profile(
            flight.aircraft,
            total_duration_s=flight.baseline.total_duration_s,
            initial_fl=fl,
        )
    return world, flights


def build_scenario_full(
    cfg: solver_pb2.ScenarioConfig,
) -> tuple[World, list[Flight], list[EvaluatedOption], list[ConflictEdge], list[CapacityBucket]]:
    """Full scenario: world, flights, evaluated options, conflicts, capacity buckets."""
    world, flights = build_world_and_flights(cfg)
    weights = cost_weights(cfg)

    evals: list[EvaluatedOption] = []
    for flight in flights:
        evals.extend(build_and_evaluate_flight(flight, world, cost_weights=weights))

    conflicts = build_conflict_graph(evals, world)
    buckets = build_capacity_buckets(evals, world)
    return world, flights, evals, conflicts, buckets
