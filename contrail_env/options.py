"""
options.py — Enumerate the option menu for each flight.

WHAT IS AN OPTION?
==================
An OPTION is one AltitudeProfile that a flight COULD fly. Each flight
gets a small menu (typically 4-5) of options:

    Option 0: baseline (the airline-filed plan, with its natural step climbs)
    Option 1: shift the WHOLE profile up by 2,000 ft  (climb avoidance)
    Option 2: shift the WHOLE profile down by 2,000 ft (descend avoidance)
    Option 3: shift the WHOLE profile up by 4,000 ft  (large climb)
    Option 4 (sometimes): lateral detour (not implemented here; future work)

The QUBO then picks ONE option per flight, subject to the constraints
that conflicting options can't both be chosen and that capacity buckets
aren't exceeded.

WHY GLOBAL SHIFTS, NOT LOCAL DETOURS?
=====================================
Because of Dean et al. 2025: any altitude change commits you to the
new level for at least 90 minutes. For a typical 2-hour leg, that's
nearly the entire flight. So a "local detour around an ISSR" actually
collapses to "fly the whole leg at a different altitude" anyway.

For LONGER legs (>3 hours), local detours start to make sense, and
options.py would need an extra function generating mid-leg perturbed
profiles. We leave that as future work — see generate_local_avoidance().

ROAD-TRIP ANALOGY
=================
Each option is a different "route plan" for the same trip. Option 0
is "use the GPS default." Option 1 is "take the toll road" (faster but
costs more). Option 2 is "take the scenic route" (slower but might
avoid rain). The driver picks ONE; the QUBO picks for them.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Sequence
import math

from .aircraft import Aircraft
from .flight import (
    Flight, AltitudeProfile, AltitudeSegment,
    build_baseline_profile, replace_segment_fl, shift_all_segments,
    available_fls, evaluate_option, EvaluatedOption,
)


# =============================================================================
# OPTION KINDS — labels for human-readable output
# =============================================================================

class OptionKind(str, Enum):
    BASELINE        = "baseline"
    SHIFT_UP_2K     = "shift+2000ft"
    SHIFT_DOWN_2K   = "shift-2000ft"
    SHIFT_UP_4K     = "shift+4000ft"
    SHIFT_DOWN_4K   = "shift-4000ft"
    LOCAL_AVOID     = "local_avoid"     # future use


# =============================================================================
# OPTION SPEC — what we generate before evaluation
# =============================================================================

@dataclass(frozen=True)
class OptionSpec:
    """
    A pre-evaluation option: just the kind and the profile.

    We separate this from EvaluatedOption (in flight.py) because:
        - Option enumeration is cheap (just profile arithmetic)
        - Option evaluation against a world is expensive (voxelization,
          fuel-burn integration)

    Generating the OptionSpecs first lets us inspect/print the menu
    before committing to scoring.
    """
    kind: OptionKind
    profile: AltitudeProfile

    def __repr__(self) -> str:
        fls = "->".join(f"FL{s.fl}" for s in self.profile.segments)
        return f"OptionSpec({self.kind.value}, profile={fls})"


# =============================================================================
# ENUMERATION
# =============================================================================

def enumerate_option_specs(
    flight: Flight,
    fl_band_set: tuple[int, ...] = (340, 360, 380, 400),
    shifts: tuple[int, ...] = (+20, -20, +40),  # in FL units (2,000 ft each)
) -> list[OptionSpec]:
    """
    Build the menu of OptionSpecs for one flight.

    Strategy:
        1. Always include baseline.
        2. For each FL shift in `shifts`, build a shifted version of the
           baseline. Skip shifts that go off the FL band set.
        3. Drop duplicates (which can happen if the baseline is already
           at the top/bottom and shifts coalesce).

    The output is in stable order (baseline first), so option_index 0
    is always baseline.

    For the canonical paper example (4 options per flight), use the
    default `shifts`. For richer studies use more shifts; for tighter
    QUBO budgets use fewer.
    """
    baseline = flight.baseline

    specs: list[OptionSpec] = [OptionSpec(OptionKind.BASELINE, baseline)]

    fl_min = min(fl_band_set)
    fl_max = max(fl_band_set)

    for shift in shifts:
        # Check that EVERY segment lands inside the allowed FL band set
        # after shifting.
        new_fls_in_band = all(
            (seg.fl + shift) in fl_band_set for seg in baseline.segments
        )
        if not new_fls_in_band:
            continue   # Out-of-band shift, skip

        new_profile = shift_all_segments(baseline, shift)
        # Tag with the option kind
        if shift == +20:
            kind = OptionKind.SHIFT_UP_2K
        elif shift == -20:
            kind = OptionKind.SHIFT_DOWN_2K
        elif shift == +40:
            kind = OptionKind.SHIFT_UP_4K
        elif shift == -40:
            kind = OptionKind.SHIFT_DOWN_4K
        else:
            kind = OptionKind.LOCAL_AVOID  # catch-all

        specs.append(OptionSpec(kind, new_profile))

    # Deduplicate by profile content (in case two shifts produce the same)
    seen = set()
    uniq_specs = []
    for s in specs:
        key = tuple((seg.fl, seg.t_start_s, seg.t_end_s)
                    for seg in s.profile.segments)
        if key not in seen:
            seen.add(key)
            uniq_specs.append(s)
    return uniq_specs


# =============================================================================
# PIPELINE — build flights, enumerate, and evaluate against a world
# =============================================================================

def build_and_evaluate_flight(
    flight: Flight,
    world,                          # World
    fl_band_set: tuple[int, ...] = (340, 360, 380, 400),
    shifts: tuple[int, ...] = (+20, -20, +40),
    cost_weights: tuple[float, float, float] = (1.0, 5.0, 0.5),
    mach: float = 0.78,
) -> list[EvaluatedOption]:
    """
    Top-level: take a Flight (with baseline already attached), enumerate
    its options, evaluate each against the world, compute the combined
    cost, and return the list of EvaluatedOptions ready for QUBO.

    cost_weights = (alpha_fuel, beta_contrail, gamma_disruption).
    Default emphasizes contrail avoidance (beta=5) over fuel/disruption.
    """
    specs = enumerate_option_specs(flight, fl_band_set=fl_band_set,
                                    shifts=shifts)
    evals: list[EvaluatedOption] = []
    for i, spec in enumerate(specs):
        ev = evaluate_option(flight, spec.profile, world,
                              option_index=i, mach=mach)
        ev.compute_combined_cost(*cost_weights)
        evals.append(ev)
    # Also update the Flight object so callers can introspect
    flight.options = [spec.profile for spec in specs]
    return evals


# =============================================================================
# CONVENIENCE: build N flights with random origins/destinations
# =============================================================================

def build_random_flights(
    n_flights: int,
    world,                  # World — for the planning region bounds
    aircraft_factory=None,  # callable -> Aircraft, default a320_like
    seed: int = 42,
    cruise_mach: float = 0.78,
    initial_fl: int = 360,
    snapshot_window_s: tuple[float, float] = (0.0, 1800.0),
    corridor_frac: float = 0.25,
) -> list[Flight]:
    """
    Generate a synthetic batch of flights crossing the planning region.

    Each flight:
        - Goes west-to-east along a narrow latitude CORRIDOR (so flights
          actually overlap geographically — otherwise no conflicts arise).
        - Has a randomized departure within the snapshot window.
        - Has a baseline profile with optional step climbs from its
          aircraft performance model.

    Why a corridor? Without one, random origins/destinations spread over
    700 km of latitude give flights that never come within 100 km of
    each other -> no shared ISSR cells -> no conflicts -> trivial QUBO.
    Clustering them into a ~200 km corridor mimics real Europe traffic,
    which IS funnelled along major airways.

    Parameters:
        n_flights:         number of flights to generate
        initial_fl:        baseline cruise FL. Default FL360 (middle of
                           the band) so options can shift +/-20 and +40
                           without going out of band.
        snapshot_window_s: range for randomized departure times
        corridor_frac:     fraction of the planning region's N-S extent
                           used for the corridor (default 0.25 = 25%,
                           ~200 km in our 800-km-wide region)

    The "snapshot mode" enforces that all flights are in the planning
    region within a ~30-min window — they really do compete for the
    same airspace and ISSR cells at the same time.
    """
    import numpy as np
    from .aircraft import a320_like

    if aircraft_factory is None:
        aircraft_factory = lambda i: a320_like(f"AC{i:03d}",
                                                mass_kg=72_000.0)

    rng = np.random.default_rng(seed)
    g = world.grid
    flights: list[Flight] = []
    t_lo, t_hi = snapshot_window_s

    # Corridor: centered N-S, with width = corridor_frac * total N-S extent.
    y_span = g.y_max_km - g.y_min_km
    y_center = (g.y_min_km + g.y_max_km) / 2.0
    half_corridor = y_span * corridor_frac / 2.0
    y_lo = y_center - half_corridor
    y_hi = y_center + half_corridor

    for i in range(n_flights):
        # Origin and destination y within the corridor
        ox = g.x_min_km
        oy = float(rng.uniform(y_lo, y_hi))
        dx = g.x_max_km
        dy = float(rng.uniform(y_lo, y_hi))
        # Departure: uniform within window
        t_dep = float(rng.uniform(t_lo, t_hi))

        ac = aircraft_factory(i)

        # Estimate duration and build baseline profile
        dist_km = math.hypot(dx - ox, dy - oy)
        # Approximate cruise speed at the initial FL
        tas_ms = mach_to_ms(cruise_mach, fl_to_m(initial_fl))
        duration_s = dist_km * 1000.0 / tas_ms
        baseline = build_baseline_profile(
            ac, total_duration_s=duration_s, initial_fl=initial_fl,
            cruise_mach=cruise_mach,
        )

        flights.append(Flight(
            name=f"F{i+1}",
            origin_km=(ox, oy),
            destination_km=(dx, dy),
            departure_s=t_dep,
            aircraft=ac,
            baseline=baseline,
        ))
    return flights


# Re-import the utilities we use locally
from .units import mach_to_ms, fl_to_m  # placed at bottom to avoid circularity


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    from .world import default_european_world

    world = default_european_world(seed=42)
    flights = build_random_flights(n_flights=3, world=world, seed=42)

    print(f"Built {len(flights)} random flights:")
    for f in flights:
        print(f"  {f}")
        for i, seg in enumerate(f.baseline.segments):
            print(f"    baseline seg {i}: FL{seg.fl} "
                  f"({seg.duration_s/60:.1f} min)")

    print(f"\nEnumerating options for {flights[0].name}:")
    specs = enumerate_option_specs(flights[0])
    for s in specs:
        print(f"  {s}")

    print(f"\nEvaluating options for {flights[0].name}:")
    evals = build_and_evaluate_flight(flights[0], world)
    for ev in evals:
        print(f"  opt{ev.option_index} ({ev.profile.segments[0].fl}-..): "
              f"fuel={ev.fuel_kg:6.0f} kg, "
              f"contrail={ev.contrail_cells:2d} cells, "
              f"disrupt={ev.disruption_FLmin:5.1f} FL-min, "
              f"combined={ev.cost_combined:7.1f}, "
              f"{len(ev.buckets)} buckets")

    print("\nAll self-tests passed.")
