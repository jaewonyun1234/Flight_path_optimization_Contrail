"""
reduce_to_flight.py — one cleaned ADS-B track -> one contrail_env.Flight.

This is the seam that lets REAL historical traffic flow through the exact same
optimizer the synthetic generator feeds: the output is an ordinary
`contrail_env.Flight`, so option enumeration, the QUBO builder, all three
solvers, and the GUI map need no changes.

Honesty notes
=============
* The baseline is PREFERABLY the real observed altitude profile (resampled
  from the track's barometric altitude), so "baseline" means the actually-flown
  plan. Only when the altitude data is too sparse/noisy do we fall back to the
  synthetic fuel-optimal `build_baseline_profile`, and we say so (a flag on the
  returned info, logged by build_dataset).
* Aircraft type -> performance profile is an explicit, documented lookup. We do
  NOT invent a new performance model per type; unmapped types use a320_like.

Pure numpy + contrail_env (core) — no pandas/network — so it imports lean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contrail_env import (
    Aircraft,
    AltitudeProfile,
    AltitudeSegment,
    Flight,
    a320_like,
    build_baseline_profile,
    m_to_fl,
)

from .config import FlightsConfig
from .tracks import Track

# Explicit aircraft-type -> Aircraft factory table. Only the a320_like linear
# performance model exists today, so every mapped type points at it; the table
# is the place to add real BADA-class profiles later. Keys are ICAO type
# designators (as OpenSky's aircraft DB reports them).
AIRCRAFT_TYPE_TO_FACTORY = {
    "A319": a320_like,
    "A320": a320_like,
    "A321": a320_like,
    "B737": a320_like,
    "B738": a320_like,
}

# FL snap grid (RVSM 2000-ft steps) and the physical envelope we clamp to.
_FL_STEP = 20
_FL_MIN, _FL_MAX = 280, 420


@dataclass(frozen=True)
class ReducedFlight:
    """A Flight plus provenance: did we use the real profile or the fallback?"""

    flight: Flight
    used_observed_baseline: bool


def aircraft_for(tail: str, aircraft_type: str | None) -> Aircraft:
    """Map an ICAO aircraft type to an Aircraft, defaulting to a320_like."""
    factory = AIRCRAFT_TYPE_TO_FACTORY.get((aircraft_type or "").upper(), a320_like)
    return factory(tail)


def reduce_to_flight(
    track: Track,
    config: FlightsConfig,
    *,
    name: str | None = None,
    aircraft_type: str | None = None,
    reference_epoch_s: float = 0.0,
) -> ReducedFlight:
    """Convert one cleaned `Track` into a `contrail_env.Flight`.

    reference_epoch_s anchors departure times: departure_s = entry_time -
    reference_epoch_s (clamped at 0). build_dataset passes the earliest entry
    time across the batch so flights share one snapshot window.

    Raises ValueError if the track never enters the planning box (the caller
    should have filtered, but we fail loudly rather than fabricate a position).
    """
    anchor = config.anchor

    inside = (
        (track.lon >= config.lon_min) & (track.lon <= config.lon_max)
        & (track.lat >= config.lat_min) & (track.lat <= config.lat_max)
    )
    idx = np.flatnonzero(inside)
    if idx.size < 2:
        raise ValueError(
            f"track {track.icao24!r} has fewer than 2 points inside the planning "
            f"box {config.bbox}; cannot derive origin/destination"
        )

    i0, i1 = int(idx[0]), int(idx[-1])

    # origin/destination: first/last in-box position -> local km.
    ox, oy = anchor.geo_to_local(float(track.lon[i0]), float(track.lat[i0]))
    dx, dy = anchor.geo_to_local(float(track.lon[i1]), float(track.lat[i1]))

    departure_s = max(0.0, float(track.time_s[i0]) - reference_epoch_s)

    # In-box slice, times relative to entry (t=0 at entry).
    tt = track.time_s[i0:i1 + 1] - float(track.time_s[i0])
    alt = track.baro_altitude_m[i0:i1 + 1]
    total_duration_s = float(tt[-1])

    ac = aircraft_for(name or track.callsign or track.icao24, aircraft_type)

    profile = _observed_profile(tt, alt)
    used_observed = profile is not None
    if profile is None:
        # Too sparse/noisy to trust the observed altitudes — fall back to the
        # synthetic fuel-optimal climb. (build_dataset logs when this happens.)
        duration = total_duration_s if total_duration_s > 0 else 3600.0
        profile = build_baseline_profile(ac, total_duration_s=duration)

    flight = Flight(
        name=name or (track.callsign.strip() or track.icao24),
        origin_km=(float(ox), float(oy)),
        destination_km=(float(dx), float(dy)),
        departure_s=departure_s,
        aircraft=ac,
        baseline=profile,
    )
    return ReducedFlight(flight=flight, used_observed_baseline=used_observed)


def _observed_profile(tt: np.ndarray, alt_m: np.ndarray) -> AltitudeProfile | None:
    """Resample observed barometric altitude into contiguous AltitudeSegments.

    Returns None when the data can't support a trustworthy profile (zero
    duration, or every altitude sample missing), so the caller can fall back.
    """
    if tt.size < 2 or tt[-1] <= tt[0]:
        return None
    finite = np.isfinite(alt_m)
    if not finite.any():
        return None

    # Snap each finite altitude to the RVSM FL grid; carry the last good FL
    # across any NaN gaps (forward-fill) so a few missing reports don't break
    # the profile.
    fls = np.empty(alt_m.shape, dtype=int)
    last = None
    for k in range(alt_m.size):
        if finite[k]:
            last = _snap_fl(float(alt_m[k]))
        fls[k] = last if last is not None else _snap_fl(float(alt_m[finite][0]))

    # Group consecutive equal-FL runs into segments, boundaries at the time of
    # each FL change. Segments are contiguous: each end == next start.
    segments: list[AltitudeSegment] = []
    seg_start_t = float(tt[0])
    cur_fl = int(fls[0])
    for k in range(1, fls.size):
        if int(fls[k]) != cur_fl:
            segments.append(AltitudeSegment(cur_fl, seg_start_t, float(tt[k])))
            seg_start_t = float(tt[k])
            cur_fl = int(fls[k])
    segments.append(AltitudeSegment(cur_fl, seg_start_t, float(tt[-1])))

    return AltitudeProfile(segments=tuple(segments))


def _snap_fl(altitude_m: float) -> int:
    """Snap an altitude (m) to the nearest in-envelope RVSM flight level."""
    fl = int(round(m_to_fl(altitude_m) / _FL_STEP) * _FL_STEP)
    return max(_FL_MIN, min(_FL_MAX, fl))
