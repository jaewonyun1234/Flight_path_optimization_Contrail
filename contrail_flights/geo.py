"""
geo.py — local sim frame <-> real (lon, lat), for the real-flight loader.

This is a DELIBERATE, independent mirror of
`contrail_ml.features.GeoAnchor`: the brief requires `contrail_flights` to be
a standalone sibling of `contrail_ml` (no import between the two), so the
transform is duplicated here rather than shared. The two MUST stay in sync —
same formula, same default origin — so the predicted ISSR field (placed via
the contrail_ml anchor) and the real flights (placed via this anchor) agree on
where things are on the map. `tests/test_geo_transform_roundtrip.py` checks the
inverse, and a cross-check test guards against drift from the contrail_ml copy.

    lat = origin_lat + y_km / KM_PER_DEG_LAT
    lon = origin_lon + x_km / (KM_PER_DEG_LAT * cos(lat))

Only numpy is needed, so this module imports cleanly in the lean install (the
[flights] extra is only for the actual OpenSky pull, not the geometry).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Must match contrail_ml.config.MLConfig defaults so flights and the predicted
# field share one coordinate system.
DEFAULT_ORIGIN_LAT = 43.0
DEFAULT_ORIGIN_LON = -5.0
KM_PER_DEG_LAT = 111.0


@dataclass(frozen=True)
class GeoAnchor:
    """Maps the local sim frame (x_km east, y_km north) to/from (lon, lat)."""

    origin_lat: float = DEFAULT_ORIGIN_LAT
    origin_lon: float = DEFAULT_ORIGIN_LON
    km_per_deg_lat: float = KM_PER_DEG_LAT

    def local_to_geo(
        self, x_km: np.ndarray | float, y_km: np.ndarray | float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """(x_km, y_km) -> (lon, lat) in degrees."""
        x = np.asarray(x_km, dtype=float)
        y = np.asarray(y_km, dtype=float)
        lat = self.origin_lat + y / self.km_per_deg_lat
        lon = self.origin_lon + x / (self.km_per_deg_lat * np.cos(np.radians(lat)))
        return _maybe_scalar(lon, x_km), _maybe_scalar(lat, y_km)

    def geo_to_local(
        self, lon_deg: np.ndarray | float, lat_deg: np.ndarray | float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """(lon, lat) -> (x_km, y_km). Inverse of local_to_geo."""
        lon = np.asarray(lon_deg, dtype=float)
        lat = np.asarray(lat_deg, dtype=float)
        y = (lat - self.origin_lat) * self.km_per_deg_lat
        x = (lon - self.origin_lon) * self.km_per_deg_lat * np.cos(np.radians(lat))
        return _maybe_scalar(x, lon_deg), _maybe_scalar(y, lat_deg)


def _maybe_scalar(arr: np.ndarray, like: np.ndarray | float) -> np.ndarray | float:
    """Return a Python float when the input was scalar, else the array."""
    if np.isscalar(like) or (isinstance(like, np.ndarray) and like.ndim == 0):
        return float(arr)
    return arr
