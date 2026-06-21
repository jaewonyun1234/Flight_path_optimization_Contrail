"""MLIssrField conforms to the ISSRField interface, with correct geo mapping."""

import numpy as np
import pytest

# Needs the [ml] extra (scipy interpolation). The core lint-type-test CI job
# installs only [dev], so skip there; the dedicated `ml` job runs these fully.
pytest.importorskip("scipy")

from contrail_env import fl_to_m
from contrail_ml.features import GeoAnchor, altitude_to_pressure_hpa
from contrail_ml.issr_field import MLIssrField


def _field_with_blob_at(local_x, local_y, fl):
    """Build an MLIssrField whose ISSR blob is centred on the geo point that
    the given local (x_km, y_km, FL) maps to."""
    anchor = GeoAnchor(origin_lat=43.0, origin_lon=-5.0)
    lon0, lat0 = anchor.local_to_geo(local_x, local_y)
    p0 = float(altitude_to_pressure_hpa(fl_to_m(fl)))

    lon_axis = np.arange(-30.0, 30.01, 2.0)
    lat_axis = np.arange(30.0, 70.01, 2.0)
    pressure_axis = np.array([150, 175, 200, 225, 250, 300], dtype=float)
    lon3, lat3, p3 = np.meshgrid(lon_axis, lat_axis, pressure_axis, indexing="ij")

    blob = np.exp(-(((lon3 - lon0) / 4.0) ** 2
                    + ((lat3 - lat0) / 3.0) ** 2
                    + ((p3 - p0) / 25.0) ** 2))
    return MLIssrField(
        lon_axis=lon_axis, lat_axis=lat_axis, pressure_axis=pressure_axis,
        rhi_excess_cube=0.4 * blob, p_issr_cube=blob, anchor=anchor,
        p_threshold=0.5, mode="prob",
    )


def test_implements_issrfield_interface():
    field = _field_with_blob_at(750.0, 400.0, 360)
    for attr in ("rhi_excess", "is_inside", "rhi_excess_grid", "mask_grid", "threshold"):
        assert hasattr(field, attr)


def test_point_inside_known_blob():
    field = _field_with_blob_at(750.0, 400.0, 360)
    z = fl_to_m(360)
    # The blob centre maps back to this local point -> inside.
    assert field.is_inside(750.0, 400.0, z) is True
    # A point near the opposite corner of the box -> outside the blob.
    assert field.is_inside(50.0, 50.0, z) is False


def test_grid_query_shapes():
    field = _field_with_blob_at(750.0, 400.0, 360)
    xs = np.linspace(50, 1450, 20)
    ys = np.linspace(50, 750, 8)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = np.full_like(xx, fl_to_m(360))
    mask = field.mask_grid(xx, yy, zz)
    exc = field.rhi_excess_grid(xx, yy, zz)
    assert mask.shape == xx.shape
    assert exc.shape == xx.shape
    assert mask.any()  # the blob shows up somewhere on the slice


def test_rhi_mode_uses_threshold():
    field = _field_with_blob_at(750.0, 400.0, 360)
    field.mode = "rhi"
    field.threshold = 0.3
    z = fl_to_m(360)
    # rhi_excess peaks at 0.4 at the centre (> 0.3) -> inside in rhi mode.
    assert field.is_inside(750.0, 400.0, z) is True
