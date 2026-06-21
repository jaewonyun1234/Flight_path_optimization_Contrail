"""
opensky_client.py — thin wrapper around the OpenSky Network historical API.

ACCESS NOTES (see docs/DATA.md)
===============================
* The public OpenSky REST API works unauthenticated for light, RECENT queries
  (a rolling ~1-2 hour window) but is heavily rate-limited.
* Bulk HISTORICAL queries (arbitrary past dates, whole regions) require an
  OpenSky research/Trino account — apply via https://opensky-network.org. Put
  the credentials in OPENSKY_USERNAME / OPENSKY_PASSWORD (or a pyopensky config
  file); never hardcode them.

HONESTY GUARANTEE
=================
This client NEVER fabricates traffic. If the `pyopensky` dependency is missing,
credentials are absent, or the network/query fails, it raises
`OpenSkyUnavailableError` with an actionable message. An empty result is also
surfaced as an error (a real region/time should have traffic) rather than
silently returning zero flights.
"""

from __future__ import annotations

from .config import FlightsConfig
from .tracks import Track, make_track


class OpenSkyUnavailableError(RuntimeError):
    """Raised when real OpenSky data cannot be obtained (deps/creds/network)."""


class OpenSkyClient:
    """Fetches and assembles raw `Track`s for a region + time window.

    The heavy `pyopensky` import is lazy and lives inside `_require_api`, so
    importing this module is cheap and the lean install is unaffected until a
    real pull is actually requested.
    """

    def __init__(self, config: FlightsConfig) -> None:
        self._config = config

    def _require_api(self):
        """Import pyopensky and build an authenticated handle, or fail loudly."""
        try:
            from pyopensky.trino import Trino
        except ImportError as exc:  # dependency not installed
            raise OpenSkyUnavailableError(
                "the 'flights' extra is not installed — real OpenSky access needs "
                "pyopensky. Install with: pip install -e '.[flights]'"
            ) from exc

        cfg = self._config
        if not (cfg.opensky_username and cfg.opensky_password):
            raise OpenSkyUnavailableError(
                "OpenSky credentials missing — set OPENSKY_USERNAME / "
                "OPENSKY_PASSWORD (research/Trino access required for historical "
                "bulk queries; see docs/DATA.md)."
            )
        try:
            return Trino()
        except Exception as exc:  # network / auth / config failure
            raise OpenSkyUnavailableError(
                f"could not connect to OpenSky Trino: {exc}"
            ) from exc

    def fetch_tracks(self) -> list[Track]:
        """Pull state vectors for the configured bbox+window and group them
        into per-aircraft `Track`s. Raises OpenSkyUnavailableError on any
        failure (including an empty result)."""
        api = self._require_api()
        cfg = self._config
        try:
            df = api.history(
                start=cfg.start_time,
                stop=cfg.end_time,
                bounds=cfg.bbox,
            )
        except Exception as exc:
            raise OpenSkyUnavailableError(f"OpenSky history query failed: {exc}") from exc

        if df is None or len(df) == 0:
            raise OpenSkyUnavailableError(
                f"OpenSky returned no traffic for {cfg.bbox} in "
                f"[{cfg.start_time}, {cfg.end_time}] — widen the window/box or "
                f"check access (it should not be empty for a real region/time)."
            )
        return tracks_from_state_dataframe(df)


def tracks_from_state_dataframe(df) -> list[Track]:
    """Group an OpenSky state-vector dataframe into per-icao24 `Track`s.

    Expects the pyopensky column names (icao24, callsign, time, latitude,
    longitude, baroaltitude). Kept separate from the network call so it can be
    unit-tested with a small in-memory frame.
    """
    tracks: list[Track] = []
    for icao24, grp in df.groupby("icao24"):
        callsign = ""
        if "callsign" in grp.columns and len(grp["callsign"]):
            callsign = str(grp["callsign"].iloc[0] or "").strip()
        tracks.append(make_track(
            icao24=str(icao24),
            callsign=callsign,
            time_s=grp["time"].to_numpy(),
            lat=grp["latitude"].to_numpy(),
            lon=grp["longitude"].to_numpy(),
            baro_altitude_m=grp["baroaltitude"].to_numpy(),
        ))
    return tracks
