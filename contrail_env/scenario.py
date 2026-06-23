"""
scenario.py — Build and solve a scenario in one process (no network layer).

A `ScenarioConfig` is fully seeded, so building from it is deterministic: the
same config always yields the same problem. This module owns the config object,
the scenario builders, and an in-process CP-SAT solve, so the desktop app runs
as a single process — no gRPC service, no ZMQ broker.

Per-flight initial flight level
===============================
`build_random_flights` puts every flight at one initial FL. Here we give each
flight its OWN baseline cruise level, drawn (seeded) from the grid's FL bands,
so the scenario spans several altitudes instead of one flat layer — more
realistic, and it makes the 3D trajectory view actually show altitude.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

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
    fl_to_m,
    mach_to_ms,
    solve_cpsat,
)

# Used when the caller leaves all three cost weights at 0: fall back to sensible
# defaults instead of an all-costs-zero (degenerate) objective.
_DEFAULT_WEIGHTS = (1.0, 5.0, 0.5)


@dataclass
class ScenarioConfig:
    """Everything needed to build + solve one deterministic scenario."""

    seed: int = 42
    n_flights: int = 4
    n_issr_blobs: int = 6
    alpha_fuel: float = 0.0
    beta_contrail: float = 0.0
    gamma_disruption: float = 0.0
    corridor_frac: float = 0.0
    snapshot_window_s: float = 0.0
    time_limit_s: float = 10.0
    issr_threshold: float = 0.0


@dataclass
class FlightChoice:
    """The chosen option for one flight, with its cost breakdown."""

    flight_name: str
    chosen_option: int
    fuel_kg: float
    contrail_cells: int
    disruption_flmin: float


@dataclass
class SolveResult:
    """The outcome of a CP-SAT solve (what the dashboard displays)."""

    objective: float
    status: str
    wall_clock_s: float
    n_conflicts: int
    n_options_total: int
    choices: list[FlightChoice] = field(default_factory=list)


def cost_weights(cfg: ScenarioConfig) -> tuple[float, float, float]:
    """(alpha_fuel, beta_contrail, gamma_disruption), with an all-zero fallback."""
    weights = (cfg.alpha_fuel, cfg.beta_contrail, cfg.gamma_disruption)
    if weights == (0.0, 0.0, 0.0):
        return _DEFAULT_WEIGHTS
    return weights


def build_world_and_flights(cfg: ScenarioConfig) -> tuple[World, list[Flight]]:
    """Build the world and flights for a config.

    Each flight gets (seed-deterministically):
      * a random heading — a chord crossing the central region, so flights
        come from any direction yet still overlap and conflict, and
      * its own baseline cruise flight level.
    """
    world = default_european_world(seed=cfg.seed, n_issr_blobs=cfg.n_issr_blobs)
    # Threshold drives what counts as a contrail (cell RHi-excess > threshold),
    # so it affects the solve, not just the picture. 0 means "use the default".
    if cfg.issr_threshold > 0:
        world.issr.threshold = float(cfg.issr_threshold)

    flights = build_random_flights(
        n_flights=cfg.n_flights,
        world=world,
        seed=cfg.seed,
        corridor_frac=cfg.corridor_frac or 0.05,
        snapshot_window_s=(0.0, cfg.snapshot_window_s or 300.0),
    )

    rng = np.random.default_rng(int(cfg.seed) * 1000 + 7)
    g = world.grid
    cx = (g.x_min_km + g.x_max_km) / 2.0
    cy = (g.y_min_km + g.y_max_km) / 2.0
    width = g.x_max_km - g.x_min_km
    height = g.y_max_km - g.y_min_km
    span = min(width, height)
    fl_bands = world.grid.fl_bands

    def _clamp(v: float, lo: float, hi: float) -> float:
        return min(max(v, lo), hi)

    for flight in flights:
        theta = rng.uniform(0.0, 2.0 * math.pi)
        length = rng.uniform(0.5, 0.85) * span
        # Midpoint near the center so chords cross and interact.
        mx = cx + rng.uniform(-0.12, 0.12) * width
        my = cy + rng.uniform(-0.12, 0.12) * height
        half = 0.5 * length
        ox = _clamp(mx - half * math.cos(theta), g.x_min_km, g.x_max_km)
        oy = _clamp(my - half * math.sin(theta), g.y_min_km, g.y_max_km)
        dx = _clamp(mx + half * math.cos(theta), g.x_min_km, g.x_max_km)
        dy = _clamp(my + half * math.sin(theta), g.y_min_km, g.y_max_km)
        flight.origin_km = (ox, oy)
        flight.destination_km = (dx, dy)

        # Duration follows the (possibly clamped) ground distance.
        dist_km = math.hypot(dx - ox, dy - oy)
        tas_ms = mach_to_ms(0.78, fl_to_m(360))
        duration_s = max(1.0, dist_km * 1000.0 / tas_ms)

        fl = int(rng.choice(fl_bands))
        flight.baseline = build_baseline_profile(
            flight.aircraft,
            total_duration_s=duration_s,
            initial_fl=fl,
        )
    return world, flights


def build_scenario_full(
    cfg: ScenarioConfig,
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


def solve_scenario(
    cfg: ScenarioConfig,
    on_progress: Callable[[int, float], None] | None = None,
) -> SolveResult:
    """Build the scenario for `cfg` and solve it with CP-SAT, in this process.

    `on_progress(improvement_index, objective)` is invoked once per improved
    incumbent — wire it straight to a UI signal for a live convergence curve.
    """
    _world, _flights, evals, conflicts, buckets = build_scenario_full(cfg)
    result = solve_cpsat(
        evals,
        conflicts,
        buckets,
        time_limit_s=cfg.time_limit_s or 10.0,
        on_progress=on_progress,
    )
    choices = [
        FlightChoice(
            flight_name=evals[i].flight_name,
            chosen_option=evals[i].option_index,
            fuel_kg=evals[i].fuel_kg,
            contrail_cells=evals[i].contrail_cells,
            disruption_flmin=evals[i].disruption_FLmin,
        )
        for i in result.chosen_eval_indices
    ]
    return SolveResult(
        objective=result.objective,
        status=result.status,
        wall_clock_s=result.wall_clock_s,
        n_conflicts=len(conflicts),
        n_options_total=len(evals),
        choices=choices,
    )
