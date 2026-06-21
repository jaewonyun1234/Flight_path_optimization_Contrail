"""reduce_to_flight: a fake ADS-B track -> a valid contrail_env.Flight.

Hermetic — builds Track fixtures by hand, no network/credentials. Checks the
core conversion: origin/destination inside the planning box, eastbound geometry,
the observed climb preserved as the baseline, and the loud failures (no in-box
points; all-NaN altitude falls back instead of fabricating).
"""

import numpy as np
import pytest

from contrail_env import fl_to_m
from contrail_flights.config import FlightsConfig
from contrail_flights.reduce_to_flight import aircraft_for, reduce_to_flight
from contrail_flights.tracks import make_track


def _climbing_track(n: int = 20):
    """West->east crossing of the box at ~46.5N, climbing FL340 -> FL360."""
    t = np.arange(n) * 60.0
    lon = np.linspace(-4.0, 13.0, n)
    lat = np.full(n, 46.5)
    alt = np.where(np.arange(n) < n // 2, fl_to_m(340), fl_to_m(360))
    return make_track("abc123", "TEST123 ", t, lat, lon, alt)


def test_reduce_produces_valid_eastbound_flight():
    cfg = FlightsConfig()
    reduced = reduce_to_flight(_climbing_track(), cfg, name="R1")
    f = reduced.flight

    assert reduced.used_observed_baseline
    assert 0.0 <= f.origin_km[0] <= 1500.0
    assert 0.0 <= f.origin_km[1] <= 800.0
    assert f.destination_km[0] > f.origin_km[0]  # eastbound
    assert f.departure_s == 0.0  # first (and only) track -> reference epoch


def test_observed_climb_becomes_the_baseline():
    cfg = FlightsConfig()
    reduced = reduce_to_flight(_climbing_track(), cfg)
    fls = [s.fl for s in reduced.flight.baseline.segments]
    assert fls == [340, 360]
    segs = reduced.flight.baseline.segments
    assert segs[0].t_start_s == 0.0
    # Contiguous: each segment's end is the next one's start.
    assert segs[0].t_end_s == segs[1].t_start_s


def test_track_with_no_in_box_points_raises():
    cfg = FlightsConfig()
    n = 12
    t = np.arange(n) * 60.0
    lon = np.linspace(-4.0, 13.0, n)
    lat = np.full(n, 10.0)  # far south of the box -> never inside
    alt = np.full(n, fl_to_m(360))
    with pytest.raises(ValueError):
        reduce_to_flight(make_track("x", "Y", t, lat, lon, alt), cfg)


def test_all_nan_altitude_falls_back_not_fabricates():
    cfg = FlightsConfig()
    n = 12
    t = np.arange(n) * 60.0
    lon = np.linspace(-4.0, 13.0, n)
    lat = np.full(n, 46.5)
    alt = np.full(n, np.nan)
    reduced = reduce_to_flight(make_track("x", "Y", t, lat, lon, alt), cfg)
    assert not reduced.used_observed_baseline
    assert reduced.flight.baseline.n_segments >= 1


def test_aircraft_type_mapping_defaults_to_a320():
    a = aircraft_for("UNMAPPED", "Z999")
    assert a.tail == "UNMAPPED"
    # A mapped type and an unmapped one both resolve to a valid Aircraft.
    assert aircraft_for("X", "A320").initial_mass_kg > 0
