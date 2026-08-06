"""
problem.py — Minimal synthetic scenario generator (the trusted input layer).

THE MODEL, IN PLAIN LANGUAGE
============================
Picture a small rectangular patch of sky seen from above: a grid with
`n_x` columns (west -> east) and `n_y` rows (south -> north). Every flight
crosses the whole patch west to east, one column per time step, so
"column x" and "time step x" are the same thing. A flight's route is a
straight line from its entry row on the west edge to its exit row on the
east edge, rounded to grid rows.

Each flight may fly that same route at one of `n_options` altitude levels
(level 0, 1, 2, ... — think numbered highway lanes stacked vertically).
Choosing the level is the whole decision.

The weather is a handful of rectangular ISSR blobs (ice-supersaturated
regions — the "rain puddles" where contrails form and persist). A blob
covers a rectangle of grid cells AND a contiguous band of altitude levels.
Flying through a blob cell at a covered level costs contrail exposure.

COSTS
=====
Each (flight, level) option gets one scalar cost:

    cost = fuel_proxy + alpha * contrail_cells

    fuel_proxy     = base fuel for the flight
                     + climb_cost * |level - preferred level|
                     + tiny per-option jitter (breaks exact ties)
    contrail_cells = number of route cells inside an ISSR blob at
                     that level

CONFLICTS
=========
Two options CONFLICT when they belong to DIFFERENT flights, sit on the
SAME altitude level, and their routes pass through the SAME ISSR cell in
the same column (= same time step). Physically: two aircraft seeding the
same supersaturated patch at once amplifies contrail formation, so the
optimizer must not pick both.

Options of the same flight never appear as conflicts here — "pick exactly
one level per flight" is the one-hot constraint, handled in qubo.py.

Everything is drawn from one seeded `np.random.default_rng(seed)`, so the
same seed always produces byte-identical scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Grid defaults: small on purpose. 8 columns x 8 rows is enough structure
# for routes to cross inside blobs while staying easy to draw by hand.
N_X = 8
N_Y = 8
N_BLOBS = 3
ALPHA = 10.0        # weight of one contrail cell relative to fuel units
CLIMB_COST = 3.0    # fuel units per level away from the preferred level


@dataclass(frozen=True)
class Scenario:
    """One problem instance, fully described by three arrays.

    Attributes:
        n_flights: F
        n_options: K (altitude levels per flight)
        costs:     shape (F*K,), cost of variable i = option k of flight f,
                   flat index i = f * K + k
        flight_of: shape (F*K,), flight_of[i] = f  (which flight owns var i)
        conflicts: cross-flight pairs (i, j) with i < j that must not both
                   be chosen
    """

    n_flights: int
    n_options: int
    costs: np.ndarray
    flight_of: np.ndarray
    conflicts: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_vars(self) -> int:
        return self.n_flights * self.n_options

    def groups(self) -> list[list[int]]:
        """Variable ids per flight: groups()[f] = [f*K, ..., f*K + K-1]."""
        k = self.n_options
        return [list(range(f * k, (f + 1) * k)) for f in range(self.n_flights)]


def make_scenario(
    n_flights: int = 4,
    n_options: int = 3,
    seed: int = 0,
    *,
    n_blobs: int = N_BLOBS,
    alpha: float = ALPHA,
) -> Scenario:
    """Generate one seeded scenario: routes, blobs, costs, conflicts."""
    rng = np.random.default_rng(seed)

    # --- 1. Routes: entry/exit rows per flight, rounded straight lines ----
    # routes[f, x] = grid row of flight f in column x (= time step x).
    entry = rng.integers(0, N_Y, size=n_flights)
    exit_ = rng.integers(0, N_Y, size=n_flights)
    xs = np.arange(N_X)
    routes = np.rint(
        entry[:, None] + (exit_ - entry)[:, None] * xs[None, :] / (N_X - 1)
    ).astype(int)

    # --- 2. ISSR blobs: (x0, x1, y0, y1, k_lo, k_hi), all inclusive ------
    blobs: list[tuple[int, int, int, int, int, int]] = []
    for _ in range(n_blobs):
        x0 = int(rng.integers(0, N_X - 2))
        y0 = int(rng.integers(0, N_Y - 2))
        x1 = min(N_X - 1, x0 + int(rng.integers(2, 4)))
        y1 = min(N_Y - 1, y0 + int(rng.integers(2, 4)))
        k_lo = int(rng.integers(0, n_options))
        k_hi = min(n_options - 1, k_lo + int(rng.integers(0, 2)))
        blobs.append((x0, x1, y0, y1, k_lo, k_hi))

    def in_blob(x: int, y: int, k: int) -> bool:
        return any(
            x0 <= x <= x1 and y0 <= y <= y1 and k_lo <= k <= k_hi
            for x0, x1, y0, y1, k_lo, k_hi in blobs
        )

    # --- 3. Costs: fuel proxy + alpha * contrail exposure -----------------
    base_fuel = rng.uniform(40.0, 60.0, size=n_flights)
    preferred = rng.integers(0, n_options, size=n_flights)

    n = n_flights * n_options
    costs = np.zeros(n)
    flight_of = np.zeros(n, dtype=int)
    contrail = np.zeros((n_flights, n_options), dtype=int)
    for f in range(n_flights):
        for k in range(n_options):
            i = f * n_options + k
            flight_of[i] = f
            contrail[f, k] = sum(1 for x in range(N_X) if in_blob(x, int(routes[f, x]), k))
            costs[i] = (
                base_fuel[f]
                + CLIMB_COST * abs(k - int(preferred[f]))
                + rng.uniform(0.0, 0.5)          # tie-breaking jitter
                + alpha * contrail[f, k]
            )

    # --- 4. Conflicts: same ISSR cell, same column, same level ------------
    conflicts: list[tuple[int, int]] = []
    for f in range(n_flights):
        for g in range(f + 1, n_flights):
            # Columns where the two routes sit in the same cell.
            shared_cols = [int(x) for x in xs if routes[f, x] == routes[g, x]]
            if not shared_cols:
                continue
            for k in range(n_options):
                if any(in_blob(x, int(routes[f, x]), k) for x in shared_cols):
                    conflicts.append((f * n_options + k, g * n_options + k))

    return Scenario(
        n_flights=n_flights,
        n_options=n_options,
        costs=costs,
        flight_of=flight_of,
        conflicts=conflicts,
    )
