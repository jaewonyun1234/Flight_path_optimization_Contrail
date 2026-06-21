"""
tracks.py — Raw ADS-B tracks and their cleaning.

A `Track` is one aircraft's observed path: parallel arrays of (time, lat, lon,
baro_altitude) plus identity (icao24, callsign). The OpenSky client produces
these; `clean_tracks` drops the unusable ones (too few points, too short, or
flat noise) and normalizes each (sorted by time, de-duplicated timestamps).

Pure numpy — no pandas, no network — so it imports in the lean install and the
hermetic tests can build fixtures directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Mean Earth radius (km) for the haversine ground-distance estimate.
_EARTH_R_KM = 6371.0


@dataclass(frozen=True)
class Track:
    """One cleaned aircraft track. Arrays are parallel and time-ascending.

    time_s: UTC seconds (epoch). lat/lon: degrees. baro_altitude_m: metres
    (may contain NaN where the report lacked altitude).
    """

    icao24: str
    callsign: str
    time_s: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    baro_altitude_m: np.ndarray

    @property
    def n_points(self) -> int:
        return int(self.time_s.size)

    def ground_distance_km(self) -> float:
        """Great-circle path length summed over consecutive points (km)."""
        if self.n_points < 2:
            return 0.0
        return float(np.sum(_haversine_km(
            self.lat[:-1], self.lon[:-1], self.lat[1:], self.lon[1:]
        )))


def make_track(
    icao24: str,
    callsign: str,
    time_s,
    lat,
    lon,
    baro_altitude_m,
) -> Track:
    """Build a Track from raw sequences, sorted by time with duplicate
    timestamps removed (keeping the first). No filtering of short/sparse
    tracks here — that is `clean_tracks`' job."""
    t = np.asarray(time_s, dtype=float)
    order = np.argsort(t, kind="stable")
    t = t[order]
    lat_a = np.asarray(lat, dtype=float)[order]
    lon_a = np.asarray(lon, dtype=float)[order]
    alt_a = np.asarray(baro_altitude_m, dtype=float)[order]

    # Drop duplicate timestamps (ADS-B often repeats the last state vector).
    keep = np.concatenate(([True], np.diff(t) > 0))
    return Track(
        icao24=icao24,
        callsign=callsign,
        time_s=t[keep],
        lat=lat_a[keep],
        lon=lon_a[keep],
        baro_altitude_m=alt_a[keep],
    )


def clean_tracks(
    tracks: list[Track],
    min_points: int = 10,
    min_track_km: float = 200.0,
) -> list[Track]:
    """Keep only tracks with enough points AND enough ground distance.

    A loud, explicit filter — never silently fabricate or pad a thin track.
    """
    out: list[Track] = []
    for tr in tracks:
        if tr.n_points < min_points:
            continue
        if tr.ground_distance_km() < min_track_km:
            continue
        out.append(tr)
    return out


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance (km) between two arrays of (lat, lon) degrees."""
    p1 = np.radians(np.asarray(lat1, dtype=float))
    p2 = np.radians(np.asarray(lat2, dtype=float))
    dphi = p2 - p1
    dlam = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * _EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
