"""The seam: real (fixture) flights -> qubo -> CP-SAT, with no qubo/World edits.

Mirrors test_ml_world_integration.py but swaps in real-flight objects built from
hand-made ADS-B tracks via contrail_flights. The point of the feature: real
historical traffic drops into the existing optimizer untouched. Hermetic — no
network/credentials (tracks are passed directly to flights_for).
"""

import numpy as np

from contrail_env import (
    build_and_evaluate_flight,
    build_capacity_buckets,
    build_conflict_graph,
    default_european_world,
    fl_to_m,
    solve_cpsat,
)
from contrail_flights.build_dataset import flights_for
from contrail_flights.config import FlightsConfig
from contrail_flights.tracks import make_track


def _fixture_tracks(n_flights: int = 3):
    """A handful of overlapping west->east crossings at adjacent latitudes."""
    tracks = []
    for i in range(n_flights):
        n = 24
        t = np.arange(n) * 60.0
        lon = np.linspace(-4.5, 13.0, n)        # stays inside 0..1500 km east
        lat = np.full(n, 45.5 + i * 0.6)
        alt = np.full(n, fl_to_m(360))
        tracks.append(make_track(f"ac{i}", f"F{i}", t, lat, lon, alt))
    return tracks


def test_real_flights_build():
    cfg = FlightsConfig()
    flights = flights_for(cfg, tracks=_fixture_tracks(3), use_cache=False)
    assert len(flights) == 3
    for f in flights:
        assert 0.0 <= f.origin_km[0] <= 1500.0
        assert f.baseline.n_segments >= 1


def test_real_flights_solve_feasibly():
    cfg = FlightsConfig()
    flights = flights_for(cfg, tracks=_fixture_tracks(3), use_cache=False)
    world = default_european_world(seed=1)

    evals = []
    for f in flights:
        evals.extend(build_and_evaluate_flight(f, world))
    conflicts = build_conflict_graph(evals, world)
    buckets = build_capacity_buckets(evals, world)

    result = solve_cpsat(evals, conflicts, buckets, time_limit_s=10.0)
    selected = set(result.chosen_eval_indices)

    # One option per flight.
    assert len({evals[i].flight_name for i in selected}) == 3
    # No conflict edge fully chosen.
    for e in conflicts:
        assert not (e.i in selected and e.j in selected)
    # No capacity bucket exceeded.
    for b in buckets:
        assert sum(1 for m in b.members if m in selected) <= b.capacity
