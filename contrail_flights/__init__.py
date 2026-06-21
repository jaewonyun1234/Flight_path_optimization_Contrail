"""
contrail_flights — real historical traffic (OpenSky) as a Flight source.

An alternative to the synthetic `build_random_flights`: pull actual flown
ADS-B tracks for a European region + time window from the OpenSky Network and
reduce each to a `contrail_env.Flight`, so the existing option enumeration,
QUBO builder, all three solvers, and the GUI map consume them unchanged.

DESIGN INVARIANTS
=================
1. Real data only — never fabricate traffic. Missing deps/credentials/network,
   or an empty result, raise `OpenSkyUnavailableError` (see opensky_client).
2. Independent of contrail_ml: this package is a sibling, not a dependency. It
   carries its own geo transform (geo.py), kept in sync with the model's anchor.
3. Lean import surface: only numpy + contrail_env at import time. pyopensky /
   pandas / pyarrow (the [flights] extra) are imported lazily where used.
4. Default stays synthetic: the service only builds real flights when explicitly
   asked (flight_source="real").

HONEST NAMING
=============
OpenSky gives real flown HISTORICAL tracks, not published schedules. This is
"demonstrated on real historical European traffic" — the same evaluation method
used in real contrail-avoidance trials — NOT "schedule optimization".
"""

from __future__ import annotations

from .config import DEFAULT_CONFIG, FlightsConfig
from .geo import GeoAnchor

__all__ = ["FlightsConfig", "DEFAULT_CONFIG", "GeoAnchor", "flights_for", "reduce_to_flight"]


def __getattr__(name: str):
    # Lazy re-export so the common helpers are importable from the package root
    # without importing the heavier orchestration modules at package load.
    if name == "flights_for":
        from .build_dataset import flights_for

        return flights_for
    if name == "reduce_to_flight":
        from .reduce_to_flight import reduce_to_flight

        return reduce_to_flight
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
