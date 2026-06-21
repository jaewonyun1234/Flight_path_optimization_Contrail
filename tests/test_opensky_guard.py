"""The OpenSky client fails loudly — it never returns empty/fabricated data.

Whether or not pyopensky is installed, fetching without credentials must raise
OpenSkyUnavailableError (missing dependency OR missing credentials), so a real
pull can never silently degrade into zero or invented flights.
"""

import pytest

from contrail_flights.config import FlightsConfig
from contrail_flights.opensky_client import (
    OpenSkyClient,
    OpenSkyUnavailableError,
    tracks_from_state_dataframe,
)


def test_fetch_raises_without_deps_or_credentials():
    cfg = FlightsConfig(opensky_username=None, opensky_password=None)
    client = OpenSkyClient(cfg)
    with pytest.raises(OpenSkyUnavailableError):
        client.fetch_tracks()


def test_dataframe_grouping_is_unit_testable():
    # The pure parsing step works on an in-memory frame (no network), so the
    # grouping logic is covered even though the live query can't be in CI.
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "icao24": ["a", "a", "b"],
        "callsign": ["AAA", "AAA", "BBB"],
        "time": [0.0, 60.0, 0.0],
        "latitude": [46.0, 46.1, 47.0],
        "longitude": [0.0, 0.5, 1.0],
        "baroaltitude": [10000.0, 10500.0, 11000.0],
    })
    tracks = tracks_from_state_dataframe(df)
    assert {t.icao24 for t in tracks} == {"a", "b"}
    assert next(t for t in tracks if t.icao24 == "a").n_points == 2
