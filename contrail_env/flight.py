"""
flight.py — Flight specifications, altitude profiles, and cost evaluation.

CORE CONCEPTS
=============

    Flight        = "abstract" flight (origin, destination, departure time,
                    aircraft, list of option AltitudeProfiles).
                    A Flight is what you'd file with ATC: WHO is flying
                    WHERE and WHEN. It does NOT specify all the details
                    of the trajectory yet.

    AltitudeProfile = a sequence of constant-altitude segments. This is
                    the ONLY thing that varies between options for the
                    same Flight. Different profiles = different options.

    Trajectory    = Flight + chosen AltitudeProfile -> a fully-specified
                    spatiotemporal path. This is what you actually fly.

    EvaluatedOption = Trajectory + its scalar costs (fuel kg, contrail
                    cells, disruption FL-minutes) + the (sector,
                    time-bucket) IDs it visits. This is what becomes
                    a QUBO variable.

ROAD-TRIP ANALOGY
=================
- A Flight is "I'm going from Boston to DC on Friday morning in my Civic."
- An AltitudeProfile is "I'll take I-95 in the right lane the whole way"
  vs "I'll take I-95 but switch to the carpool lane after Stamford."
- A Trajectory is the full physical drive with timestamps.
- An EvaluatedOption is the trip with mileage, fuel cost, and a list
  of every toll plaza ID you drove through.

WHY THIS LAYERED ABSTRACTION?
=============================
Because the QUBO needs SCALARS, not trajectories. By separating
"flight metadata" (Flight) from "altitude choice" (AltitudeProfile)
from "scored trajectory" (EvaluatedOption), we get:

    - Clean option enumeration: just generate AltitudeProfiles.
    - Cheap option scoring: only the chosen profile gets evaluated.
    - One QUBO variable per EvaluatedOption.

The Flight stays the same across all options; only the profile changes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence
import math
import numpy as np

from .units import M_PER_FT, fl_to_m, mach_to_ms, KM_PER_NM
from .aircraft import Aircraft


# =============================================================================
# ALTITUDE PROFILE — sequence of constant-FL segments
# =============================================================================

@dataclass(frozen=True)
class AltitudeSegment:
    """
    A constant-altitude leg.

    The pair (fl, t_start_s, t_end_s) says "stay at flight level `fl`
    from t_start_s to t_end_s seconds after departure."

    Segments are ALWAYS bounded by time, not by waypoints. This matches
    how flight plans are actually filed: "cruise FL360 from POSITION
    POINT_A to POSITION_B" eventually reduces to "FL360 from t=0 to
    t=5400 s" once you know the ground speed.
    """
    fl: int
    t_start_s: float
    t_end_s: float

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s


@dataclass(frozen=True)
class AltitudeProfile:
    """
    Piecewise-constant altitude profile.

    A profile is a tuple of segments. The segments' times are guaranteed
    to be CONTIGUOUS and NON-OVERLAPPING (we assert this on construction).

    BASELINE PROFILE convention:
        A flight's baseline is the airline-filed plan: typically start at
        FL340, step climb to FL360 after ~90 min of fuel burn, step climb
        again to FL380 after ~90 more min, etc. (See build_baseline_profile.)

    AVOIDANCE PROFILE convention:
        An avoidance variant changes ONE segment's FL to dodge an ISSR
        (or merges/splits segments to introduce a temporary detour level).

    Per Dean et al. 2025, each segment must be AT LEAST 90 min long
    (5400 s). Shorter segments are physically possible but the
    operational utility drops sharply — controllers and pilots avoid
    "yo-yo" altitude profiles.
    """
    segments: tuple[AltitudeSegment, ...]

    # Operational constraint: any segment must be at least this long.
    MIN_SEGMENT_S: float = 5400.0   # 90 minutes (Dean et al. 2025)

    def __post_init__(self):
        # Sanity: segments must be in order and contiguous, non-empty
        assert len(self.segments) >= 1, "Profile must have at least 1 segment"
        for i in range(len(self.segments) - 1):
            assert self.segments[i].t_end_s == self.segments[i+1].t_start_s, (
                f"Segments {i} and {i+1} are not contiguous")
        # We don't enforce MIN_SEGMENT_S in __post_init__ because the
        # *baseline* profile generator may produce short final segments
        # at the end of a leg. Use is_operationally_valid() to check.

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def total_duration_s(self) -> float:
        return self.segments[-1].t_end_s - self.segments[0].t_start_s

    def fl_at(self, t_s: float) -> int:
        """Flight level at a given time. Times past the end clamp to last FL."""
        for seg in self.segments:
            if seg.t_start_s <= t_s < seg.t_end_s:
                return seg.fl
        # Past the end — return last FL (e.g., for end-time boundary)
        return self.segments[-1].fl

    def n_changes(self) -> int:
        """How many altitude changes occur (= n_segments - 1)."""
        return self.n_segments - 1

    def is_operationally_valid(self) -> bool:
        """
        Returns True iff every segment is at least MIN_SEGMENT_S long
        (modulo the LAST segment, which is allowed to be short because
        it ends with the flight reaching its destination).
        """
        for seg in self.segments[:-1]:
            if seg.duration_s < self.MIN_SEGMENT_S:
                return False
        return True

    # -------------------------------------------------------------------------
    # Comparison to baseline (for disruption cost)
    # -------------------------------------------------------------------------

    def deviation_from(self, other: "AltitudeProfile",
                       n_samples: int = 100) -> float:
        """
        Integral of |fl_self(t) - fl_other(t)| over the overlapping window.
        Units: FL-seconds. Used for disruption cost.

        We sample uniformly in time. Higher n_samples = more accurate
        integration, but for piecewise-constant signals 100 is overkill.
        """
        t_start = max(self.segments[0].t_start_s,
                      other.segments[0].t_start_s)
        t_end   = min(self.segments[-1].t_end_s,
                      other.segments[-1].t_end_s)
        if t_end <= t_start:
            return 0.0
        dt = (t_end - t_start) / n_samples
        total = 0.0
        for i in range(n_samples):
            t = t_start + (i + 0.5) * dt
            total += abs(self.fl_at(t) - other.fl_at(t)) * dt
        return total


# =============================================================================
# AVAILABLE FL MENU — RVSM grid, direction-dependent rules
# =============================================================================

def available_fls(initial_fl: int, fl_band_set: Sequence[int] = (340, 360, 380, 400)
                  ) -> list[int]:
    """
    Which Flight Levels can this flight access?

    Real ICAO rules: even FLs eastbound, odd FLs westbound (semicircular
    rule). For our toy we use a coarse 2,000-ft grid so all FLs are
    "even" anyway — we just return the full menu.

    Adding direction-dependent FLs would be a one-line change here.
    """
    return list(fl_band_set)


# =============================================================================
# BASELINE PROFILE — what the airline files
# =============================================================================

def build_baseline_profile(
    aircraft: Aircraft,
    total_duration_s: float,
    initial_fl: int = 340,
    min_segment_s: float = 5400.0,
    cruise_mach: float = 0.78,
) -> AltitudeProfile:
    """
    Build the airline-filed baseline profile with step climbs.

    Logic:
        - Start at initial_fl with current mass = aircraft.initial_mass_kg.
        - Walk forward in 1-minute increments, burning fuel.
        - When the optimal FL has risen at least 20 above the current FL
          AND at least min_segment_s has elapsed since the last change,
          step climb by 20 FL (2000 ft).
        - Continue until total_duration_s is reached.

    The 90-min minimum-segment constraint (Dean et al. 2025) is what
    keeps the profile from "yo-yo"ing every time fuel drops a little.

    ROAD-TRIP ANALOGY: like setting cruise control, then upshifting only
    when both (a) the engine is asking for the upshift AND (b) you've
    been in this gear long enough for the upshift to be worth the
    "click."
    """
    perf = aircraft.performance
    mass = aircraft.initial_mass_kg
    current_fl = initial_fl
    seg_start = 0.0
    segments: list[AltitudeSegment] = []

    dt = 60.0      # 1-minute integration step (good enough for fuel burn)
    t = 0.0
    while t < total_duration_s:
        # Burn fuel for this minute
        burn = perf.cruise_burn_kgs(mass, current_fl, cruise_mach)
        mass -= burn * dt

        # Has it become beneficial AND legal to step climb?
        opt = perf.optimal_fl(mass)
        time_in_seg = t - seg_start
        if (opt >= current_fl + 20
                and time_in_seg >= min_segment_s
                and t + min_segment_s <= total_duration_s):
            # ... and we have at least min_segment_s of flight left
            # to spend in the new segment (otherwise the climb cost
            # isn't worth it).
            segments.append(AltitudeSegment(
                fl=current_fl,
                t_start_s=seg_start,
                t_end_s=t,
            ))
            current_fl += 20      # step climb 2000 ft
            seg_start = t
            # Climb takes a couple of minutes; we skip ahead so the next
            # iteration doesn't immediately try to climb again.
            _extra_fuel, climb_time = perf.climb_cost(mass, current_fl-20, current_fl)
            t += climb_time
            continue

        t += dt

    # Close the final segment at total_duration_s
    segments.append(AltitudeSegment(
        fl=current_fl,
        t_start_s=seg_start,
        t_end_s=total_duration_s,
    ))
    return AltitudeProfile(segments=tuple(segments))


# =============================================================================
# AVOIDANCE PROFILES — perturbations of the baseline
# =============================================================================

def replace_segment_fl(profile: AltitudeProfile, seg_index: int,
                       new_fl: int) -> AltitudeProfile:
    """
    Build a new profile by changing ONE segment's FL.

    The segment's start and end times are preserved. This is the simplest
    "avoid the ISSR by spending one full leg at a different altitude"
    perturbation. Per Dean et al., changing the whole segment is more
    realistic than a tight detour because forecast uncertainty makes
    short detours unreliable.

    Returns: a new AltitudeProfile (immutable; the input is unchanged).
    """
    new_segs = list(profile.segments)
    old = new_segs[seg_index]
    new_segs[seg_index] = AltitudeSegment(
        fl=new_fl,
        t_start_s=old.t_start_s,
        t_end_s=old.t_end_s,
    )
    return AltitudeProfile(segments=tuple(new_segs))


def shift_all_segments(profile: AltitudeProfile, delta_fl: int
                       ) -> AltitudeProfile:
    """
    Shift EVERY segment's FL by delta_fl. Useful for "fly the whole leg
    2000 ft higher" options.
    """
    new_segs = tuple(
        AltitudeSegment(seg.fl + delta_fl, seg.t_start_s, seg.t_end_s)
        for seg in profile.segments
    )
    return AltitudeProfile(segments=new_segs)


# =============================================================================
# THE FLIGHT — abstract spec, no chosen profile yet
# =============================================================================

@dataclass
class Flight:
    """
    An abstract flight: WHO, WHERE, WHEN.

    Members:
        name:        human-readable label (e.g., "AF1234 MAD-FRA")
        origin_km:   (x_km, y_km) of departure
        destination_km: (x_km, y_km) of arrival
        departure_s: timestamp of takeoff/entry into planning region
        aircraft:    the Aircraft instance flying this leg
        baseline:    the airline-filed AltitudeProfile (used for disruption)
        options:     list of candidate AltitudeProfiles (including baseline)

    The .options list is what enumerates_options() will fill in.
    """
    name: str
    origin_km:      tuple[float, float]
    destination_km: tuple[float, float]
    departure_s: float
    aircraft: Aircraft
    baseline: AltitudeProfile
    options: list[AltitudeProfile] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------------

    @property
    def horizontal_distance_km(self) -> float:
        """Great-circle distance approximated by Euclidean in our local frame."""
        x0, y0 = self.origin_km
        x1, y1 = self.destination_km
        return math.hypot(x1 - x0, y1 - y0)

    def estimated_duration_s(self, mach: float = 0.78,
                              cruise_fl: int = 360) -> float:
        """Crude duration estimate at constant cruise speed."""
        tas_ms = mach_to_ms(mach, fl_to_m(cruise_fl))
        return (self.horizontal_distance_km * 1000.0) / tas_ms

    def __repr__(self) -> str:
        return (f"Flight({self.name!r}, "
                f"{self.origin_km} -> {self.destination_km}, "
                f"t_dep={self.departure_s:.0f}s, "
                f"{len(self.options)} option(s))")


# =============================================================================
# TRAJECTORY — Flight + chosen profile -> concrete spatiotemporal path
# =============================================================================

def waypoints_for(flight: Flight, profile: AltitudeProfile,
                   sample_dt_s: float = 60.0, mach: float = 0.78
                   ) -> list[tuple[float, float, float, float]]:
    """
    Sample the (x, y, z, t) trajectory for a flight flying `profile`.

    Assumption: the flight follows a great-circle (here straight-line
    in our local Cartesian frame) at constant Mach, and its altitude
    follows `profile` exactly. We sample every `sample_dt_s` seconds.

    Returned waypoints are in WORLD coordinates: x_km, y_km, z_m, t_s.

    Limitation (mid-fidelity): we ignore wind effects on ground speed,
    so distance/time is just straight TAS. For high fidelity you'd
    integrate the wind field along the path.
    """
    x0, y0 = flight.origin_km
    x1, y1 = flight.destination_km
    dist_m = math.hypot(x1 - x0, y1 - y0) * 1000.0
    dx_km = x1 - x0
    dy_km = y1 - y0

    # Estimate ground speed assuming constant Mach. Use an average altitude.
    avg_fl = sum(s.fl for s in profile.segments) // len(profile.segments)
    avg_alt_m = fl_to_m(avg_fl)
    gs_ms = mach_to_ms(mach, avg_alt_m)
    total_time_s = dist_m / gs_ms

    # Number of samples
    n = max(2, int(math.ceil(total_time_s / sample_dt_s)) + 1)
    ts = np.linspace(0.0, total_time_s, n)

    waypoints: list[tuple[float, float, float, float]] = []
    for t_along_path in ts:
        # Fraction of path completed
        f = t_along_path / total_time_s if total_time_s > 0 else 0.0
        x = x0 + f * dx_km
        y = y0 + f * dy_km

        # Wall-clock time for the world (relative to planning-window start)
        t_world = flight.departure_s + float(t_along_path)

        # Altitude from the profile, indexed by time-along-this-flight
        fl = profile.fl_at(float(t_along_path))
        z = fl_to_m(fl)

        waypoints.append((x, y, z, t_world))

    return waypoints


# =============================================================================
# EVALUATED OPTION — the QUBO variable, with scalar costs
# =============================================================================

@dataclass
class EvaluatedOption:
    """
    One concrete option for one flight, with all costs and bucket
    occupancies computed.

    THIS IS WHAT BECOMES A QUBO VARIABLE x_{f,k}.

    Members:
        flight_name:        which flight (string for readability)
        option_index:       which option (0 = baseline by convention)
        profile:            the AltitudeProfile we chose
        fuel_kg:            total fuel burn for this option
        contrail_cells:     # of ISSR cells the trajectory visits
        disruption_FLmin:   integral |delta_FL| dt (FL-minutes)
        buckets:            set of (sector, time-bucket) pairs visited
        cells_visited:      list of 4D cells (for conflict-graph construction)
        cost_combined:      alpha*fuel + beta*contrail + gamma*disruption
                            (filled by caller via .compute_combined_cost)

    The QUBO objective uses cost_combined as the q_{f,k} coefficient.
    """
    flight_name: str
    option_index: int
    profile: AltitudeProfile

    fuel_kg: float
    contrail_cells: int
    disruption_FLmin: float

    buckets: set[tuple[int, int]]
    cells_visited: list[tuple[int, int, int, int]]

    cost_combined: float = 0.0

    def compute_combined_cost(self, alpha: float = 1.0, beta: float = 5.0,
                              gamma: float = 0.5) -> float:
        """
        Combine the three sub-costs into one QUBO coefficient.

        Default weights:
            alpha = 1.0   : 1 kg fuel -> 1 unit cost
            beta  = 5.0   : 1 contrail cell -> 5 units cost
            gamma = 0.5   : 1 FL-minute of deviation -> 0.5 units cost

        These weights are POLICY choices. Climate-focused planners use
        beta >> alpha (heavily penalize contrails); cost-focused airlines
        use alpha >> beta. The QUBO doesn't care; only the relative
        weights matter for the optimum.
        """
        self.cost_combined = (
            alpha * self.fuel_kg
            + beta  * self.contrail_cells
            + gamma * self.disruption_FLmin
        )
        return self.cost_combined


# =============================================================================
# EVALUATION — Flight + Profile + World -> EvaluatedOption
# =============================================================================

def evaluate_option(
    flight: Flight,
    profile: AltitudeProfile,
    world,             # World instance, but we avoid the import cycle
    option_index: int,
    mach: float = 0.78,
    sample_dt_s: float = 60.0,
    densify_km: float = 25.0,
) -> EvaluatedOption:
    """
    Score one option (flight + altitude profile) against a world.

    Steps:
        1. Generate waypoints for the trajectory (constant Mach, fixed route).
        2. Voxelize the waypoints into 4D cells via World.voxelize().
        3. Count ISSR cells -> contrail_cells.
        4. Collect buckets -> for capacity constraints.
        5. Walk the trajectory cell-by-cell, integrating fuel burn from
           the aircraft's performance model. Add step-climb fuel surcharges.
        6. Compute disruption FL-minutes vs the flight's baseline.

    Returns an EvaluatedOption ready to be combined and placed into a QUBO.
    """
    # 1. Sample waypoints
    waypoints = waypoints_for(flight, profile, sample_dt_s=sample_dt_s,
                              mach=mach)

    # 2. Voxelize against the world grid
    cells = world.voxelize(waypoints, densify_to_km=densify_km)

    # 3. Count ISSR cells
    issr_cells = [c for c in cells if world.is_issr_cell(c)]
    n_contrail = len(issr_cells)

    # 4. Buckets visited
    buckets = world.buckets_visited(cells)

    # 5. Fuel burn — walk the trajectory, integrating burn rate
    perf = flight.aircraft.performance
    mass = flight.aircraft.initial_mass_kg
    total_fuel = 0.0

    # Cruise burn segment by segment
    for seg in profile.segments:
        # How long do we cruise in this segment?
        dur_s = seg.duration_s
        if dur_s <= 0:
            continue
        # Burn rate at MIDPOINT mass of this segment (good enough at
        # mid-fidelity; for higher accuracy, integrate burn(t) over the
        # segment).
        burn_kgs = perf.cruise_burn_kgs(mass, seg.fl, mach)
        seg_fuel = burn_kgs * dur_s
        total_fuel += seg_fuel
        mass -= seg_fuel

    # Step-climb fuel surcharges
    for i in range(profile.n_changes()):
        seg_a = profile.segments[i]
        seg_b = profile.segments[i + 1]
        extra_fuel, _t = perf.climb_cost(mass, seg_a.fl, seg_b.fl)
        total_fuel += extra_fuel

    # 6. Disruption: integral |delta_FL| dt, converted to FL-minutes
    disruption_s = profile.deviation_from(flight.baseline)
    disruption_FLmin = disruption_s / 60.0   # FL-seconds to FL-minutes

    return EvaluatedOption(
        flight_name=flight.name,
        option_index=option_index,
        profile=profile,
        fuel_kg=total_fuel,
        contrail_cells=n_contrail,
        disruption_FLmin=disruption_FLmin,
        buckets=buckets,
        cells_visited=cells,
    )


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    from .aircraft import a320_like

    # Build a baseline profile for a 100-minute leg
    ac = a320_like("TEST", mass_kg=72_000.0)
    prof = build_baseline_profile(ac, total_duration_s=100 * 60)

    print("Baseline profile for a 100-min leg (A320, 72t at start):")
    for i, seg in enumerate(prof.segments):
        print(f"  seg {i}: FL{seg.fl} from {seg.t_start_s/60:5.1f} min "
              f"to {seg.t_end_s/60:5.1f} min "
              f"({seg.duration_s/60:.1f} min long)")
    print(f"Total: {prof.n_segments} segments, "
          f"{prof.n_changes()} altitude changes")
    print(f"Operationally valid (>= 90 min/segment): "
          f"{prof.is_operationally_valid()}")

    # Build an avoidance variant
    alt = replace_segment_fl(prof, seg_index=0, new_fl=360)
    print(f"\nAvoidance variant (seg 0 -> FL360):")
    for i, seg in enumerate(alt.segments):
        print(f"  seg {i}: FL{seg.fl}")
    print(f"  Deviation from baseline: "
          f"{alt.deviation_from(prof)/60:.1f} FL-min")

    print("\nAll self-tests passed.")
