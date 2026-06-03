"""
units.py — SI <-> aviation unit conversions.

DESIGN PRINCIPLE
================
The simulator uses SI units INTERNALLY everywhere:
    distance: meters (m) and kilometers (km)
    speed:    m/s
    mass:     kilograms (kg)
    time:     seconds (s)
    temp:     Kelvin (K)

Aviation conventions (feet, knots, nautical miles, Flight Levels, Mach)
appear ONLY at input/output boundaries — when a user types "FL360" or
when we print a result back to them.

Why bother? Two reasons:
    1. Mixing units inside the code is the #1 cause of bugs in flight
       simulators (e.g., the Mars Climate Orbiter loss in 1999).
    2. Quantum/optimization code expects pure numbers; staying in SI
       means we never have to ask "is this number feet or meters?"
       while reading a QUBO matrix.
"""

from __future__ import annotations
import math


# =============================================================================
# DISTANCE CONVERSIONS
# =============================================================================
# These constants come from international standards (ICAO, BIPM).

M_PER_FT  = 0.3048              # exact, by international agreement (1959)
FT_PER_M  = 1.0 / M_PER_FT      # ≈ 3.28084 ft/m

M_PER_NM  = 1852.0              # exact, by international agreement (1929)
NM_PER_M  = 1.0 / M_PER_NM
NM_PER_KM = 1000.0 * NM_PER_M   # ≈ 0.5400 NM/km
KM_PER_NM = 1.852               # convenience

# Approximate km per degree of latitude (constant everywhere on Earth, to
# 0.5% accuracy — we ignore Earth's oblateness for a mid-fidelity sim).
KM_PER_DEG_LAT = 111.0


# =============================================================================
# SPEED CONVERSIONS
# =============================================================================

MS_PER_KT = M_PER_NM / 3600.0   # ≈ 0.5144 m/s per knot
KT_PER_MS = 1.0 / MS_PER_KT     # ≈ 1.9438 knots per m/s


# =============================================================================
# FLIGHT LEVEL <-> METRIC CONVERSIONS
# =============================================================================
# A "Flight Level" (FL) is altitude in HUNDREDS OF FEET above the standard
# pressure datum (1013.25 hPa). FL360 = 36,000 ft = 10,972.8 m.
# Aviation always rounds to flight levels, so we treat FL as an integer.

def fl_to_ft(fl: int) -> float:
    """Convert Flight Level to feet. FL360 -> 36000 ft."""
    return fl * 100.0


def ft_to_fl(ft: float) -> int:
    """Convert feet to nearest Flight Level. 36000 ft -> FL360."""
    return int(round(ft / 100.0))


def fl_to_m(fl: int) -> float:
    """Convert Flight Level to meters. FL360 -> 10972.8 m."""
    return fl_to_ft(fl) * M_PER_FT


def m_to_fl(m: float) -> int:
    """Convert meters to nearest Flight Level."""
    return ft_to_fl(m * FT_PER_M)


# =============================================================================
# MACH NUMBER -> AIRSPEED
# =============================================================================
# Mach number is the ratio of true airspeed to local speed of sound.
# Speed of sound depends on temperature: a_sound = 20.05 * sqrt(T_kelvin).
#
# ISA (International Standard Atmosphere) gives temperature vs altitude:
#   - Below 11 km (tropopause):  T = 288.15 - 0.0065 * h_m   [K]
#   - Above 11 km (stratosphere): T = 216.65 K (constant up to ~20 km)
#
# At cruise altitudes (FL340-FL400 ≈ 10.4-12.2 km), we're right around the
# tropopause, so we apply the piecewise formula carefully.


def isa_temperature(altitude_m: float) -> float:
    """
    International Standard Atmosphere temperature in Kelvin.

    Valid for altitudes 0 to ~20 km. Beyond that we'd need the upper
    stratosphere model, which is irrelevant for civil aviation.
    """
    if altitude_m <= 11_000.0:
        return 288.15 - 0.0065 * altitude_m   # troposphere (lapse rate)
    elif altitude_m <= 20_000.0:
        return 216.65                          # lower stratosphere (constant)
    else:
        # We don't expect to be here, but be defensive.
        return 216.65


def speed_of_sound_ms(altitude_m: float) -> float:
    """
    Speed of sound at given altitude using ISA temperature.

    a = 20.05 * sqrt(T_K) m/s, where 20.05 = sqrt(gamma * R / M_air).
    """
    T_k = isa_temperature(altitude_m)
    return 20.05 * math.sqrt(T_k)


def mach_to_ms(mach: float, altitude_m: float) -> float:
    """
    Convert Mach number to true airspeed in m/s at a given altitude.

    Example: Mach 0.78 at FL360 (10973 m) -> ~230 m/s -> ~447 knots.
    This matches what an A320 cruise card shows.
    """
    return mach * speed_of_sound_ms(altitude_m)


def ms_to_mach(speed_ms: float, altitude_m: float) -> float:
    """Inverse of mach_to_ms."""
    return speed_ms / speed_of_sound_ms(altitude_m)


# =============================================================================
# SELF-TEST
# =============================================================================
# Run `python -m contrail_env.units` to verify the conversions.

if __name__ == "__main__":
    # Standard A320 cruise: FL360, Mach 0.78
    fl = 360
    mach = 0.78

    h_m = fl_to_m(fl)
    h_ft = fl_to_ft(fl)
    tas_ms = mach_to_ms(mach, h_m)
    tas_kt = tas_ms * KT_PER_MS
    a_ms = speed_of_sound_ms(h_m)
    T_K = isa_temperature(h_m)

    print(f"FL{fl}:")
    print(f"  altitude       = {h_m:8.1f} m   ({h_ft:.0f} ft)")
    print(f"  ISA temp       = {T_K:8.2f} K   ({T_K - 273.15:.1f} C)")
    print(f"  speed of sound = {a_ms:8.2f} m/s ({a_ms * KT_PER_MS:.0f} kt)")
    print(f"  Mach {mach}      = {tas_ms:8.2f} m/s ({tas_kt:.0f} kt TAS)")

    # Sanity checks
    assert abs(h_m - 10972.8) < 0.1, "FL360 should be 10972.8 m"
    assert abs(tas_kt - 447) < 5, "Mach 0.78 at FL360 should be ~447 kt"
    print("\nAll self-tests passed.")
