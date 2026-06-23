"""
geo.py — Where the synthetic world sits on Earth (local sim frame <-> lon/lat).

The optimizer works entirely in the local 1500x800 km Cartesian box; geography
only matters for the GUI Map tab, which needs to place that box on a real
basemap. This module owns that one mapping so the map has a home for it without
reaching into any heavier package.

    lat = origin_lat + y_km / KM_PER_DEG_LAT
    lon = origin_lon + x_km / (KM_PER_DEG_LAT * cos(lat))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _maybe_scalar(arr: np.ndarray, like: np.ndarray | float) -> np.ndarray | float:
    """Return a Python float when the input was scalar, else the array."""
    if np.isscalar(like) or (isinstance(like, np.ndarray) and like.ndim == 0):
        return float(arr)
    return arr


@dataclass(frozen=True)
class GeoAnchor:
    """Maps the local sim frame (x_km east, y_km north) to geography.

    The sim is a flat Cartesian box; the map lives on (lon, lat). One
    small-angle anchor ties them together so the risk overlay and the routes
    render over the same place.
    """

    origin_lat: float = 43.0
    origin_lon: float = -5.0
    km_per_deg_lat: float = 111.0

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


# The canonical anchor: places the 1500x800 km box over south-west -> central
# Europe (roughly Madrid to Frankfurt), matching the default world geometry.
EUROPEAN_ANCHOR = GeoAnchor(origin_lat=43.0, origin_lon=-5.0)
