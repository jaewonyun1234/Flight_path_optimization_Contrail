"""
config.py — Typed configuration for the real-flight (OpenSky) loader.

One dataclass (`FlightsConfig`) carries the planning box, the time window, the
cleaning thresholds, the snapshot-window normalization, the cache directory,
and the OpenSky credentials (read from the environment — never hardcoded). The
geographic box and anchor default to MATCH `contrail_ml.config.MLConfig`, so a
real flight's local (x_km, y_km) lands in the same frame the predicted ISSR
field uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

from .geo import DEFAULT_ORIGIN_LAT, DEFAULT_ORIGIN_LON, GeoAnchor


@dataclass(frozen=True)
class FlightsConfig:
    """All configuration for pulling, cleaning, and reducing real tracks."""

    # ---- Planning box (deg). Matches the default sim grid extent (0..1500 km
    # east, 0..800 km north) under the shared anchor below. ----------------
    lon_min: float = -5.0
    lon_max: float = 15.0
    lat_min: float = 43.0
    lat_max: float = 50.0

    # ---- Geographic anchor: local (x=0, y=0) -> (origin_lon, origin_lat).
    # MUST match contrail_ml.config.MLConfig so flights and field agree. ----
    origin_lat: float = DEFAULT_ORIGIN_LAT
    origin_lon: float = DEFAULT_ORIGIN_LON

    # ---- Time window (ISO 8601) for the historical pull --------------------
    start_time: str = "2023-06-01T12:00:00Z"
    end_time: str = "2023-06-01T13:00:00Z"

    # ---- Track cleaning ----------------------------------------------------
    min_track_points: int = 10      # drop sparse tracks
    min_track_km: float = 200.0     # drop tracks that barely cross the box

    # ---- Departure normalization (mirrors build_random_flights) -----------
    # Real entry times span the whole query window; we compress them into a
    # short snapshot window so flights compete for the same airspace.
    snapshot_window_s: float = 300.0

    # ---- Cache + credentials (read from env, never hardcoded) -------------
    cache_dir: str = "data/flights_cache"
    opensky_username: str | None = None
    opensky_password: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #

    @property
    def anchor(self) -> GeoAnchor:
        """The geographic anchor placing the local frame on (lon, lat)."""
        return GeoAnchor(origin_lat=self.origin_lat, origin_lon=self.origin_lon)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(lon_min, lon_max, lat_min, lat_max) for the OpenSky query."""
        return (self.lon_min, self.lon_max, self.lat_min, self.lat_max)

    def in_bbox(self, lon: float, lat: float) -> bool:
        """Is this (lon, lat) inside the planning box?"""
        return (self.lon_min <= lon <= self.lon_max
                and self.lat_min <= lat <= self.lat_max)

    def with_overrides(self, **kwargs: Any) -> FlightsConfig:
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)

    @classmethod
    def from_env(cls, prefix: str = "CONTRAIL_FLIGHTS_") -> FlightsConfig:
        """Build a config, overriding any field from `${PREFIX}FIELD` env vars.

        OpenSky credentials also fall back to the conventional OPENSKY_USERNAME
        / OPENSKY_PASSWORD names so they line up with pyopensky's own config.
        """
        base = cls()
        overrides: dict[str, Any] = {}
        for f in base.__dataclass_fields__.values():  # type: ignore[attr-defined]
            if f.name == "extra":
                continue
            env_key = f"{prefix}{f.name.upper()}"
            if env_key in os.environ:
                overrides[f.name] = _coerce(getattr(base, f.name), os.environ[env_key])
        overrides.setdefault("opensky_username", os.environ.get("OPENSKY_USERNAME"))
        overrides.setdefault("opensky_password", os.environ.get("OPENSKY_PASSWORD"))
        return replace(base, **overrides)


def _coerce(reference: Any, raw: str) -> Any:
    """Coerce an env-var string to the type of the dataclass default value."""
    if isinstance(reference, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(raw)
    if isinstance(reference, float):
        return float(raw)
    return raw


DEFAULT_CONFIG = FlightsConfig()
