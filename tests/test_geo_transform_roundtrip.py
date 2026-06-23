"""The map's geo transform inverts cleanly: geo_to_local(local_to_geo(x,y)) == (x,y).

The Map tab places the local sim box on a lon/lat basemap via contrail_env.geo;
this guards that the forward/inverse pair is consistent so routes and the risk
overlay land where they should.
"""

import numpy as np

from contrail_env.geo import EUROPEAN_ANCHOR, GeoAnchor


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


def test_canonical_anchor_sits_over_europe():
    # The default box anchor should place (x=0, y=0) at south-west Europe.
    lon, lat = EUROPEAN_ANCHOR.local_to_geo(0.0, 0.0)
    assert abs(lon - (-5.0)) < 1e-9
    assert abs(lat - 43.0) < 1e-9
