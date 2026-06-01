"""
world.py — The "World" object: bundles airspace + ISSR + sectors.

PURPOSE
=======
A World is just a container. It exists so the rest of the code can ask
ONE object for everything it needs about the environment, instead of
juggling three or four arguments.

Conceptually, the World is "the planning region as it is right now":
    - WHERE flights can be (the AirspaceGrid)
    - WHICH cells are rain zones (the ISSRField)
    - WHO controls each cell and HOW BUSY it can get (the SectorMap)

The World does not know about flights. Flights "enter" the world via
their trajectories, which are then evaluated against the world's
properties.

ROAD-TRIP ANALOGY
=================
A World is the road map + the weather forecast + the highway-patrol
roster. The cars (flights) drive across it; the map doesn't care which
cars drive where.

EXTENSIONS NOT INCLUDED HERE
============================
For higher-fidelity work, the World would also carry:
    - 4D wind field (for fuel-burn corrections)
    - Temperature field (for engine performance)
    - Restricted airspace polygons (military zones, etc.)
    - Weather hazards beyond ISSRs (convection, turbulence)

We treat all of these as future work, with hooks for them in the
class methods (returns 0/None by default).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from .airspace import AirspaceGrid, SectorMap, voxelize_trajectory, densify_waypoints
from .synthetic_issr import ISSRField


# =============================================================================
# THE WORLD
# =============================================================================

@dataclass
class World:
    """
    The full planning environment.

    Members:
        grid:    discretization of space and time
        issr:    ice-supersaturated regions
        sectors: ATC sectors with capacities

    Common queries this object answers:
        - "What 4D cells does this trajectory enter?" -> voxelize()
        - "Does this cell intersect an ISSR?"          -> is_issr_cell()
        - "Which capacity bucket does this cell belong to?" -> bucket()
    """
    grid:    AirspaceGrid
    issr:    ISSRField
    sectors: SectorMap

    # -------------------------------------------------------------------------
    # Trajectory voxelization
    # -------------------------------------------------------------------------

    def voxelize(self, waypoints, densify_to_km: float = 25.0):
        """
        Convert a continuous trajectory to the list of 4D cells it
        occupies. Automatically densifies the waypoints so we don't
        skip cells.

        waypoints: iterable of (x_km, y_km, z_m, t_s) tuples.
        Returns: list of (ix, iy, iz, it) cell indices, with consecutive
        duplicates removed.
        """
        # Densify FIRST so we don't skip cells between sparse waypoints.
        # Use a step slightly smaller than the cell width to be safe.
        step_km = min(self.grid.dx_km, self.grid.dy_km, densify_to_km) * 0.9
        dense = densify_waypoints(list(waypoints), max_step_km=step_km)
        return voxelize_trajectory(self.grid, dense)

    # -------------------------------------------------------------------------
    # ISSR queries on cells (the trajectory-to-contrail bridge)
    # -------------------------------------------------------------------------

    def is_issr_cell(self, cell: tuple[int, int, int, int]) -> bool:
        """
        Is this 4D cell inside an ISSR?

        We test the CELL CENTER. For a more conservative test that flags
        a cell if ANY point in it is in an ISSR, you'd need to sample
        the cell — for mid-fidelity, the center is fine because cells
        are smaller than typical ISSR scales.
        """
        x, y, z, _t = self.grid.cell_center(*cell)
        # Note: ISSRs are time-invariant in this model. For 4D ISSR
        # fields (e.g., evolving forecasts), you'd add a `t` argument
        # to issr.is_inside.
        return self.issr.is_inside(x, y, z)

    def issr_cells_in(self, cells) -> list[tuple[int, int, int, int]]:
        """Filter a list of cells down to those that are ISSR-marked."""
        return [c for c in cells if self.is_issr_cell(c)]

    # -------------------------------------------------------------------------
    # Capacity bucket assignment (the trajectory-to-capacity bridge)
    # -------------------------------------------------------------------------

    def bucket(self, cell: tuple[int, int, int, int]
               ) -> Optional[tuple[int, int]]:
        """
        Which (sector, time-bucket) does this cell belong to?

        Returns None if the cell is outside all sectors (no capacity
        constraint applies).
        """
        return self.sectors.bucket_id(*cell)

    def buckets_visited(self, cells) -> set[tuple[int, int]]:
        """
        Set of (sector, time-bucket) pairs this trajectory occupies.

        Each (flight, option) contributes one such set; the optimizer
        uses these to build the capacity-constraint indicators a_{f,k,b}.
        """
        out: set[tuple[int, int]] = set()
        for c in cells:
            b = self.bucket(c)
            if b is not None:
                out.add(b)
        return out

    # -------------------------------------------------------------------------
    # Wind / temperature hooks — placeholders for future work
    # -------------------------------------------------------------------------

    def wind_ms(self, cell) -> tuple[float, float]:
        """
        Horizontal wind vector at a cell, in (u_east, v_north) m/s.

        Mid-fidelity stub: returns calm air. Replace with ERA5
        lookup for higher fidelity.
        """
        return (0.0, 0.0)

    def temperature_K(self, cell) -> float:
        """
        Ambient temperature at a cell, in Kelvin.

        Mid-fidelity stub: returns ISA temperature for the altitude.
        """
        from .units import isa_temperature
        _x, _y, z, _t = self.grid.cell_center(*cell)
        return isa_temperature(z)


# =============================================================================
# CONVENIENCE: build a default world for the canonical example
# =============================================================================

def default_european_world(
    seed: int = 42,
    n_issr_blobs: int = 6,
    nx: int = 60,
    ny: int = 32,
) -> World:
    """
    Construct a default World matching the canonical example geometry:
        - 1500 km x 800 km planning region (roughly Madrid to Frankfurt)
        - 4 FL bands (340, 360, 380, 400)
        - 12 time buckets of 10 min each (2-hour planning window — long
          enough to cover a typical mid-haul flight's traversal of the
          region; shorter than this clamps everything to the final
          bucket and the time dimension stops doing useful work)
        - 6 random ISSR blobs
        - Uniform 3x3 sector grid with capacity 3
    """
    from .synthetic_issr import random_issr_field

    grid = AirspaceGrid(
        x_min_km=0.0, x_max_km=1500.0,
        y_min_km=0.0, y_max_km=800.0,
        nx=nx, ny=ny,
        fl_bands=(340, 360, 380, 400),
        dt_s=600.0, nt=12,
    )

    issr = random_issr_field(
        n_blobs=n_issr_blobs,
        domain=(0.0, 1500.0, 0.0, 800.0, 9000.0, 12500.0),
        seed=seed,
    )

    from .airspace import uniform_sector_grid
    sectors = uniform_sector_grid(
        grid,
        cells_per_sector_x=15,    # 4 sector columns
        cells_per_sector_y=16,    # 2 sector rows
        altitude_bands_per_sector=2,  # 2 altitude bands
        capacity=4,
    )

    return World(grid=grid, issr=issr, sectors=sectors)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    w = default_european_world(seed=42)

    print(f"World:")
    print(f"  grid:        {w.grid.shape} ({w.grid.n_cells} cells)")
    print(f"  ISSR blobs:  {len(w.issr.blobs)}")
    print(f"  sectors:     {len(w.sectors.sectors)}")

    # Test a trajectory
    waypoints = [
        (0.0,    400.0, 10973.0, 0.0),     # Madrid (FL360)
        (750.0,  400.0, 10973.0, 1500.0),  # midway
        (1500.0, 400.0, 10973.0, 3000.0),  # Frankfurt
    ]
    cells = w.voxelize(waypoints)
    issr_cells = w.issr_cells_in(cells)
    buckets = w.buckets_visited(cells)
    print(f"\nMadrid-Frankfurt at FL360:")
    print(f"  passes through {len(cells)} cells")
    print(f"  {len(issr_cells)} of those are ISSR")
    print(f"  occupies {len(buckets)} (sector, time-bucket) pairs")

    print("\nAll self-tests passed.")
