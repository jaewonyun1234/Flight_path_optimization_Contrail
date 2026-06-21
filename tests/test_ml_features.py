"""Shared feature engineering: thermodynamics, atmosphere, encodings, geo.

These are the anti-skew keystone (one definition for train and serve), so they
get known-value and round-trip checks.
"""

import numpy as np
import pytest

from contrail_ml.features import (
    GeoAnchor,
    altitude_to_pressure_hpa,
    day_of_year_encoding,
    e_si_pa,
    pressure_to_altitude_m,
    rhi_excess,
    rhi_percent,
    solar_time_encoding,
)


def test_e_si_monotonic_and_positive():
    temps = np.array([210.0, 220.0, 230.0, 240.0, 250.0])
    esi = np.asarray(e_si_pa(temps))
    assert np.all(esi > 0)
    assert np.all(np.diff(esi) > 0)  # warmer air holds more vapour over ice


def test_rhi_hits_100_at_ice_saturation():
    # If vapour pressure equals the ice-saturation pressure, RHi must be 100%.
    T, p_hpa = 225.0, 225.0
    esi = float(e_si_pa(T))                 # Pa
    e = esi                                  # exactly saturated
    p_pa = p_hpa * 100.0
    eps = 0.621981
    q = eps * e / (p_pa - (1.0 - eps) * e)   # invert vapour-pressure formula
    assert abs(float(rhi_percent(T, q, p_hpa)) - 100.0) < 1e-6


def test_rhi_excess_semantics():
    assert float(rhi_excess(130.0)) == pytest.approx(0.3)
    assert float(rhi_excess(80.0)) == 0.0
    assert float(rhi_excess(100.0)) == 0.0


def test_altitude_pressure_roundtrip():
    for h in [9000.0, 10363.0, 10973.0, 11582.0, 12192.0]:
        p = float(altitude_to_pressure_hpa(h))
        h2 = float(pressure_to_altitude_m(p))
        assert abs(h - h2) < 1.0  # within a metre


def test_cyclical_encodings_unit_circle():
    s, c = solar_time_encoding(np.array([0.0, 6.0, 18.0]), np.array([0.0, 0.0, 0.0]))
    assert np.allclose(np.asarray(s) ** 2 + np.asarray(c) ** 2, 1.0)
    sd, cd = day_of_year_encoding(np.array([1.0, 90.0, 180.0, 365.0]))
    assert np.allclose(np.asarray(sd) ** 2 + np.asarray(cd) ** 2, 1.0)


def test_geo_anchor_roundtrip():
    anchor = GeoAnchor(origin_lat=43.0, origin_lon=-5.0)
    xs = np.array([0.0, 750.0, 1500.0])
    ys = np.array([0.0, 400.0, 800.0])
    lon, lat = anchor.local_to_geo(xs, ys)
    x2, y2 = anchor.geo_to_local(lon, lat)
    assert np.allclose(np.asarray(x2), xs, atol=1e-6)
    assert np.allclose(np.asarray(y2), ys, atol=1e-6)
