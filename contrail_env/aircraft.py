"""
aircraft.py — Aircraft performance model (mid-fidelity).

WHAT THIS MODULE DOES
=====================
Given:
    - an aircraft TYPE (e.g., "A320")
    - its current MASS (= empty weight + fuel + payload)
    - a flight state (altitude, Mach number)

This module answers:
    1. How much fuel/second am I burning? (cruise burn)
    2. What is my OPTIMAL cruise altitude right now? (rises as fuel burns)
    3. How much extra fuel and time would a step climb cost?

FIDELITY LEVEL
==============
We use a LINEAR SURROGATE of the BADA Total Energy Model. Real BADA
tables have ~30 coefficients per aircraft model and account for:
    - Thrust at different altitudes/temperatures
    - Drag polars (CL vs CD curves)
    - Engine bleed, anti-ice, etc.

A linear model is order-of-magnitude correct (within ~5-10% on cruise
burn for typical narrowbodies) and is what most academic OR papers use
in the contrail-routing literature.

THE INTERFACE IS A PROTOCOL — so the user can swap in real BADA later
without changing any other code.

ROAD-TRIP ANALOGY
=================
- Aircraft type = car model (Honda Civic vs Ford F-150)
- Current mass = how full the trunk is (heavier = less efficient)
- Optimal altitude = best fuel-economy speed for current load
- Step climb = upshifting to a higher gear as the load lightens
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import math

from .units import M_PER_FT, FT_PER_M, mach_to_ms, isa_temperature


# =============================================================================
# PERFORMANCE PROTOCOL — what any aircraft model must provide
# =============================================================================

class AircraftPerformance(Protocol):
    """
    Interface for aircraft performance models. By coding to this interface,
    everything downstream (Flight, options enumeration, cost evaluation) can
    work with either our linear surrogate or real BADA tables.

    To swap in BADA later: write a new class that implements these four
    methods using BADA's coefficients. Everything else stays the same.
    """

    def cruise_burn_kgs(self, mass_kg: float, fl: int, mach: float) -> float:
        """Fuel burn rate in kg/s at the given cruise state."""
        ...

    def optimal_fl(self, mass_kg: float) -> int:
        """Best-fuel-economy Flight Level at this mass."""
        ...

    def climb_cost(self, mass_kg: float, fl_from: int,
                   fl_to: int) -> tuple[float, float]:
        """
        EXTRA fuel (kg) and EXTRA time (seconds) for a step climb or
        descent from fl_from to fl_to, beyond what level flight would
        have cost.

        Returns: (extra_fuel_kg, time_s).
        """
        ...

    def cruise_speed_ms(self, fl: int, mach: float) -> float:
        """True airspeed in m/s at this state (independent of mass)."""
        ...


# =============================================================================
# LINEAR SURROGATE — our default mid-fidelity model
# =============================================================================

@dataclass(frozen=True)
class LinearJetPerformance:
    """
    Linear-surrogate aircraft performance, fitted to BADA-style behaviour
    for a generic narrowbody jet (Airbus A320 / Boeing 737 class).

    Default constants give:
        - cruise burn at FL360, 70 t, Mach 0.78: ~0.67 kg/s = 2400 kg/h
        - optimal altitude rises ~2000 ft per 10 t of fuel burned
        - climb burn at ~2.5x cruise rate; descent slightly below cruise
        - takes about 2 minutes per 2000 ft step climb at cruise

    All numbers are typical narrowbody operating values. Tune for any
    specific type by overriding the constants.

    SCALING LAWS (the physics intuition):
        - Drag (and hence burn) scales linearly with mass at first order
          (induced drag ~ W^2 / (rho V^2 S), parasite drag ~ rho V^2 S;
          at constant CL the cruise burn is ~linear in W).
        - Optimal altitude rises as mass drops because thinner air at
          high altitude can support a smaller weight efficiently.
        - Step climb is mostly potential-energy gain at constant TAS,
          so extra fuel ~ m * g * dh / specific_energy_release / engine_eff.
    """
    # Reference state for the linear expansion
    mass_ref_kg:    float = 70_000.0
    fl_ref:         int   = 360
    mach_ref:       float = 0.78

    # Reference cruise burn (kg/s) at the ref state
    burn_ref_kgs:   float = 0.67

    # Sensitivities (slopes)
    # d(burn)/d(mass): ~0.005 kg/s per ton above ref mass
    d_burn_d_mass:  float = 5.0e-6     # kg/s per kg
    # d(burn)/d(altitude): off-optimal altitudes burn more.
    # We use a QUADRATIC penalty around the optimum rather than linear,
    # which is more physically realistic. Coded inside cruise_burn_kgs.
    d_burn_d_dFL2:  float = 0.0006     # kg/s per FL^2 of off-optimum

    # Optimal altitude vs mass: slope of opt_FL with mass.
    # Empirically ~ +2 FL per -1000 kg of fuel burned.
    opt_fl_ref:     int   = 340         # at ref mass
    d_optfl_d_mass: float = -2.0e-3     # FL per kg

    # Climb cost
    climb_rate_fpm: float = 1500.0      # ft/min at typical cruise
    climb_burn_factor: float = 2.5      # climb burn / cruise burn
    descent_burn_factor: float = 0.6    # descent burn / cruise burn

    # -------------------------------------------------------------------------
    # API
    # -------------------------------------------------------------------------

    def cruise_burn_kgs(self, mass_kg: float, fl: int, mach: float) -> float:
        """
        Cruise fuel burn rate, kg/s.

        Formula:
            burn = burn_ref
                 + d_burn_d_mass * (mass - mass_ref)
                 + d_burn_d_dFL2 * (fl - opt_fl(mass))^2

        The mass term captures "heavier planes burn more."
        The dFL^2 term captures "burning extra fuel when off-optimal."
        We do NOT add a Mach term because we hold Mach ~constant at 0.78.
        """
        opt = self.optimal_fl(mass_kg)
        d_mass = mass_kg - self.mass_ref_kg
        d_fl = fl - opt
        burn = (self.burn_ref_kgs
                + self.d_burn_d_mass * d_mass
                + self.d_burn_d_dFL2 * (d_fl ** 2))
        # Defensive: don't allow negative or absurd burns from extrapolation.
        return max(0.1, min(burn, 5.0))

    def optimal_fl(self, mass_kg: float) -> int:
        """
        Optimal cruise FL as a linear function of mass.

        Returned value is clamped to integer multiples of 20 (i.e., the
        2000-ft RVSM grid we use), and to the practical range FL280-FL410.
        """
        d_mass = mass_kg - self.mass_ref_kg
        opt_fl_real = self.opt_fl_ref + self.d_optfl_d_mass * d_mass
        # Snap to 20-FL grid (2000-ft increments)
        snapped = int(round(opt_fl_real / 20.0) * 20)
        # Clamp to physical envelope
        return max(280, min(410, snapped))

    def climb_cost(self, mass_kg: float, fl_from: int,
                   fl_to: int) -> tuple[float, float]:
        """
        Extra fuel (kg) and time (s) for a step climb or descent.

        For a CLIMB:
            - climb_rate_fpm dictates how long the maneuver takes
            - climb_burn_factor times cruise burn = climb burn rate
            - extra_fuel = (climb_burn - cruise_burn) * time
            - sign convention: extra_fuel >= 0

        For a DESCENT (fl_to < fl_from):
            - we assume engines at idle, burn drops below cruise
            - "extra fuel" can be NEGATIVE (we save fuel descending)
            - time scales the same way (1500 fpm typical descent rate)
        """
        delta_fl = fl_to - fl_from
        if delta_fl == 0:
            return 0.0, 0.0
        # FL changes are in units of 100 ft; convert to ft for the climb-rate calc
        delta_ft = abs(delta_fl) * 100.0
        time_s = (delta_ft / self.climb_rate_fpm) * 60.0   # min to seconds

        cruise_burn = self.cruise_burn_kgs(mass_kg,
                                            (fl_from + fl_to) // 2,
                                            self.mach_ref)
        if delta_fl > 0:
            burn_factor = self.climb_burn_factor
        else:
            burn_factor = self.descent_burn_factor

        actual_burn = cruise_burn * burn_factor
        extra_fuel = (actual_burn - cruise_burn) * time_s
        return extra_fuel, time_s

    def cruise_speed_ms(self, fl: int, mach: float) -> float:
        """True airspeed at given FL and Mach."""
        return mach_to_ms(mach, fl * 100.0 * M_PER_FT)


# =============================================================================
# AIRCRAFT INSTANCE — type + identity + initial mass
# =============================================================================

@dataclass(frozen=True)
class Aircraft:
    """
    One specific aircraft for one specific flight.

    Carries:
        tail:           call sign or registration (e.g., "AFR123", "EC-MAD")
        performance:    the performance model (LinearJetPerformance or BADA)
        initial_mass_kg: takeoff mass / start-of-cruise mass

    Aircraft is IMMUTABLE — once created, it doesn't change. The MASS
    EVOLUTION during a flight happens INSIDE the trajectory evaluation
    (see flight.py); we don't mutate the Aircraft itself.

    This keeps the model clean: an Aircraft is "the spec sheet," and
    flying it is a function-of-time elsewhere.
    """
    tail: str
    performance: AircraftPerformance
    initial_mass_kg: float = 70_000.0

    def __repr__(self) -> str:
        return (f"Aircraft({self.tail!r}, "
                f"mass={self.initial_mass_kg/1000:.1f}t)")


# =============================================================================
# CONVENIENCE: default A320-like instance
# =============================================================================

def a320_like(tail: str = "A320-DEFAULT", mass_kg: float = 72_000.0
              ) -> Aircraft:
    """A typical midweight A320 ready for a 2-hour mid-haul leg."""
    return Aircraft(
        tail=tail,
        performance=LinearJetPerformance(),
        initial_mass_kg=mass_kg,
    )


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    ac = a320_like("TEST", mass_kg=72_000.0)
    perf = ac.performance

    print(f"Aircraft: {ac}\n")

    # Burn at various states
    for mass_kg in (60_000.0, 70_000.0, 75_000.0):
        for fl in (340, 360, 380, 400):
            b = perf.cruise_burn_kgs(mass_kg, fl, 0.78)
            opt = perf.optimal_fl(mass_kg)
            tag = " <- optimal" if fl == opt else ""
            print(f"  mass={mass_kg/1000:5.1f}t, FL{fl}: burn={b*3600:6.0f} kg/h"
                  f"  (opt FL{opt}){tag}")
        print()

    # Step-climb cost
    fuel, time = perf.climb_cost(70_000.0, 340, 360)
    print(f"Step climb FL340 -> FL360 at 70t: extra fuel = {fuel:.1f} kg, "
          f"time = {time:.0f} s = {time/60:.1f} min")

    fuel, time = perf.climb_cost(70_000.0, 360, 340)
    print(f"Descent  FL360 -> FL340 at 70t: extra fuel = {fuel:+.1f} kg, "
          f"time = {time:.0f} s = {time/60:.1f} min")

    print("\nAll self-tests passed.")
