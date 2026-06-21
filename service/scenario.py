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

import math

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
    """Build the world and flights for a config.

    Each flight gets (seed-deterministically):
      * a random heading — a chord crossing the central region, so flights
        come from any direction yet still overlap and conflict, and
      * its own baseline cruise flight level.
    """
    source = cfg.issr_source or "synthetic"
    if source == "synthetic":
        world = default_european_world(seed=cfg.seed, n_issr_blobs=cfg.n_issr_blobs)
        # Threshold drives what counts as a contrail (cell RHi-excess > threshold),
        # so it affects the solve, not just the picture. 0 means "use the default".
        if cfg.issr_threshold > 0:
            world.issr.threshold = float(cfg.issr_threshold)
    elif source == "ml":
        try:
            from contrail_ml.config import MLConfig
        except ImportError as exc:
            raise ValueError(
                "issr_source='ml' is not available in this deployment — "
                "the server image ships only the CP-SAT solver. "
                "Install contrail_ml locally with: pip install -e '.[ml]'"
            ) from exc

        ml_cfg = MLConfig()
        if cfg.issr_p_threshold > 0:
            ml_cfg = ml_cfg.with_overrides(issr_p_threshold=cfg.issr_p_threshold)
        world = default_european_world(
            seed=cfg.seed,
            issr_source="ml",
            issr_kwargs=dict(
                config=ml_cfg,
                met_source="gfs",          # operational forecast
                time=cfg.issr_time or None,
            ),
        )
    else:
        raise ValueError(f"unknown issr_source {source!r} (use 'synthetic' or 'ml')")

    # Flight source: synthetic random generator (default) or real historical
    # OpenSky traffic. The real path keeps its own origin/destination/baseline,
    # so it returns before the synthetic chord-geometry randomization below.
    flight_source = cfg.flight_source or "synthetic"
    if flight_source == "real":
        return world, _build_real_flights(cfg, world)
    if flight_source != "synthetic":
        raise ValueError(
            f"unknown flight_source {flight_source!r} (use 'synthetic' or 'real')"
        )

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


def _build_real_flights(cfg: solver_pb2.ScenarioConfig, world: World) -> list[Flight]:
    """Build real historical flights via contrail_flights (OpenSky).

    contrail_flights is imported lazily (and only here), so the core server
    image without the [flights] extra is unaffected unless a client explicitly
    asks for flight_source="real". Missing deps / credentials / network surface
    as a clear ValueError (mapped to a gRPC error by the Solve handler), never a
    silent crash or fabricated traffic.
    """
    try:
        from contrail_flights.build_dataset import flights_for
        from contrail_flights.config import FlightsConfig
        from contrail_flights.opensky_client import OpenSkyUnavailableError
    except ImportError as exc:
        raise ValueError(
            "flight_source='real' is not available in this deployment — the "
            "server image ships only the synthetic flight generator. Install "
            "contrail_flights locally with: pip install -e '.[flights]'"
        ) from exc

    # Derive the geo planning box from the world grid via the shared anchor, so
    # the real flights land in the same local frame the optimizer/grid use.
    fcfg = FlightsConfig()
    g = world.grid
    lon0, lat0 = fcfg.anchor.local_to_geo(g.x_min_km, g.y_min_km)
    lon1, lat1 = fcfg.anchor.local_to_geo(g.x_max_km, g.y_max_km)
    overrides: dict[str, object] = dict(
        lon_min=min(lon0, lon1), lon_max=max(lon0, lon1),
        lat_min=min(lat0, lat1), lat_max=max(lat0, lat1),
        snapshot_window_s=cfg.snapshot_window_s or 300.0,
    )
    if cfg.flight_start_time:
        overrides["start_time"] = cfg.flight_start_time
    if cfg.flight_end_time:
        overrides["end_time"] = cfg.flight_end_time
    fcfg = fcfg.with_overrides(**overrides)

    try:
        return flights_for(fcfg, max_flights=cfg.n_flights or None)
    except OpenSkyUnavailableError as exc:
        raise ValueError(str(exc)) from exc


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
