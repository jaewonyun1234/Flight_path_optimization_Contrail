"""
cache.py — local parquet cache for pulled OpenSky tracks.

OpenSky bulk-history queries are rate-limited and credit-metered, so we cache
the cleaned tracks on disk keyed by (bbox, time window). Re-running the same
scenario then hits the cache instead of the network.

pandas/pyarrow are in the [flights] extra and imported lazily, so this module
imports in the lean install; only `load`/`save` touch the heavy deps.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from .config import FlightsConfig
from .tracks import Track

_COLUMNS = ("icao24", "callsign", "time_s", "lat", "lon", "baro_altitude_m")


def _key(config: FlightsConfig) -> str:
    """Stable filename for a config's bbox + time window."""
    raw = f"{config.bbox}|{config.start_time}|{config.end_time}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class TrackCache:
    """Reads/writes lists of `Track` as a single long-format parquet file."""

    def __init__(self, config: FlightsConfig) -> None:
        self._config = config
        self._path = os.path.join(config.cache_dir, f"tracks_{_key(config)}.parquet")

    @property
    def path(self) -> str:
        return self._path

    def exists(self) -> bool:
        return os.path.exists(self._path)

    def load(self) -> list[Track]:
        """Load cached tracks; raises FileNotFoundError if absent."""
        import pandas as pd

        df = pd.read_parquet(self._path)
        tracks: list[Track] = []
        for (icao24, callsign), grp in df.groupby(["icao24", "callsign"], sort=False):
            tracks.append(Track(
                icao24=str(icao24),
                callsign=str(callsign),
                time_s=grp["time_s"].to_numpy(dtype=float),
                lat=grp["lat"].to_numpy(dtype=float),
                lon=grp["lon"].to_numpy(dtype=float),
                baro_altitude_m=grp["baro_altitude_m"].to_numpy(dtype=float),
            ))
        return tracks

    def save(self, tracks: list[Track]) -> None:
        """Write tracks to the cache (creating the directory if needed)."""
        import pandas as pd

        os.makedirs(self._config.cache_dir, exist_ok=True)
        frames = []
        for tr in tracks:
            n = tr.n_points
            frames.append(pd.DataFrame({
                "icao24": np.repeat(tr.icao24, n),
                "callsign": np.repeat(tr.callsign, n),
                "time_s": tr.time_s,
                "lat": tr.lat,
                "lon": tr.lon,
                "baro_altitude_m": tr.baro_altitude_m,
            }))
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=list(_COLUMNS)
        )
        out.to_parquet(self._path, index=False)
