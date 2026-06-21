"""The flight loader's geo transform inverts cleanly and matches the model's.

geo_to_local(local_to_geo(x, y)) == (x, y), and the (deliberately duplicated)
contrail_flights anchor agrees with contrail_ml's so the map and the real
flights share one coordinate system.
"""

import numpy as np

from contrail_flights.geo import GeoAnchor


def test_roundtrip_vectorized():
    anchor = GeoAnchor()
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1500.0, size=200)
    y = rng.uniform(0.0, 800.0, size=200)
    lon, lat = anchor.local_to_geo(x, y)
    x2, y2 = anchor.geo_to_local(lon, lat)
    assert np.allclose(x, x2, atol=1e-6)
    assert np.allclose(y, y2, atol=1e-6)


def test_roundtrip_scalar_returns_floats():
    anchor = GeoAnchor()
    lon, lat = anchor.local_to_geo(750.0, 400.0)
    x, y = anchor.geo_to_local(lon, lat)
    assert isinstance(x, float) and isinstance(y, float)
    assert abs(x - 750.0) < 1e-6 and abs(y - 400.0) < 1e-6


def test_matches_contrail_ml_anchor():
    # The independent copy MUST agree with the model's anchor (same origin),
    # or the predicted field and the real flights would land in different places.
    from contrail_ml.config import MLConfig
    from contrail_ml.features import GeoAnchor as MLAnchor

    cfg = MLConfig()
    flights_anchor = GeoAnchor()
    ml_anchor = MLAnchor(origin_lat=cfg.origin_lat, origin_lon=cfg.origin_lon)
    lon_f, lat_f = flights_anchor.local_to_geo(500.0, 300.0)
    lon_m, lat_m = ml_anchor.local_to_geo(500.0, 300.0)
    assert abs(lon_f - lon_m) < 1e-9
    assert abs(lat_f - lat_m) < 1e-9
