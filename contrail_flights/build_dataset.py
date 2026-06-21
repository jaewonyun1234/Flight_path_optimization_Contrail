"""
build_dataset.py — orchestrate: OpenSky -> cleaned tracks -> list[Flight].

`flights_for` is the one entry point the service/CLI call: it pulls (or loads
from cache) the tracks for the configured region+window, cleans them, and
reduces each to a `contrail_env.Flight`, all anchored to one shared snapshot
window so the flights genuinely compete for the same airspace.

Real European traffic naturally overlaps (it follows a small set of airways),
so — unlike the synthetic generator — no artificial "corridor" trick is needed.
"""

from __future__ import annotations

import logging

from contrail_env import Flight

from .config import FlightsConfig
from .reduce_to_flight import reduce_to_flight
from .tracks import Track, clean_tracks

log = logging.getLogger(__name__)


def flights_for(
    config: FlightsConfig,
    *,
    tracks: list[Track] | None = None,
    use_cache: bool = True,
    max_flights: int | None = None,
) -> list[Flight]:
    """Build a list of real `Flight`s for `config`.

    Parameters
    ----------
    tracks : pre-supplied cleaned tracks (used by tests/offline runs). When
        None, tracks come from the cache or a live OpenSky pull.
    use_cache : read/write the on-disk parquet cache around a live pull.
    max_flights : cap the number of flights returned (largest tracks first).

    Raises OpenSkyUnavailableError (from the client) if a live pull is needed
    but unavailable — never silently returns synthetic or empty data.
    """
    raw = tracks if tracks is not None else _load_or_pull(config, use_cache=use_cache)
    cleaned = clean_tracks(raw, config.min_track_points, config.min_track_km)
    if not cleaned:
        raise ValueError(
            "no tracks survived cleaning — every track was too short/sparse for "
            f"the thresholds (min_points={config.min_track_points}, "
            f"min_track_km={config.min_track_km})."
        )

    # Largest (longest) tracks first, then optionally cap.
    cleaned.sort(key=lambda t: t.ground_distance_km(), reverse=True)
    if max_flights is not None:
        cleaned = cleaned[:max_flights]

    # Shared reference epoch = earliest in-window time, so the first flight
    # departs at t=0 and the rest are offset relative to it.
    reference_epoch_s = min(float(t.time_s[0]) for t in cleaned)

    flights: list[Flight] = []
    n_fallback = 0
    for i, tr in enumerate(cleaned):
        try:
            reduced = reduce_to_flight(
                tr, config,
                name=f"R{i + 1}",
                reference_epoch_s=reference_epoch_s,
            )
        except ValueError as exc:
            log.warning("skipping track %s: %s", tr.icao24, exc)
            continue
        if not reduced.used_observed_baseline:
            n_fallback += 1
        flights.append(reduced.flight)

    if n_fallback:
        log.info("%d/%d flights used the synthetic baseline fallback "
                 "(observed altitude too sparse)", n_fallback, len(flights))
    if not flights:
        raise ValueError("no usable flights after reduction (all tracks skipped).")
    return flights


def _load_or_pull(config: FlightsConfig, *, use_cache: bool) -> list[Track]:
    """Return raw tracks from cache, else pull from OpenSky and cache them."""
    from .cache import TrackCache
    from .opensky_client import OpenSkyClient

    cache = TrackCache(config)
    if use_cache and cache.exists():
        log.info("loading tracks from cache %s", cache.path)
        return cache.load()

    tracks = OpenSkyClient(config).fetch_tracks()
    if use_cache:
        cache.save(tracks)
    return tracks
