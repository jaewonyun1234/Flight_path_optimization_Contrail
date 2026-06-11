"""
airspace.py — Airspace grid, sectors, time buckets, and voxelization.

WHAT THIS MODULE PROVIDES
=========================
1. AirspaceGrid: a 4D discretization (x, y, altitude, time) of the
   planning region. Real trajectories are CONTINUOUS in space and time;
   we discretize so the optimizer can reason about countable cells.

2. SectorMap: which 4D cells belong to which Air Traffic Control sector,
   and what each sector's capacity is. This becomes the capacity
   constraint in the QUBO.

3. Voxelization functions: take a continuous trajectory (list of
   (x, y, z, t) waypoints) and return the list of grid cells it passes
   through. This is the bridge between "physics" and "optimizer".

KEY DESIGN DECISIONS
====================
- Coordinates: pure Cartesian (km in x and y, meters in z).
  The user converts from lat/lon BEFORE calling our code, or just
  works in km directly for a synthetic planning region.

- Voxel size defaults: 25 km x 25 km x 1000 ft x 5 min, picked to be
  larger than typical aircraft uncertainty and small enough to capture
  ISSR features. Tunable per instance.

- Snapshot mode: the time axis exists but for the canonical example
  we use a SINGLE time bucket (snapshot of all flights crossing the
  region within ±15 min). The infrastructure supports multi-bucket
  for future extensions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

from .units import fl_to_m, m_to_fl, FT_PER_M


# =============================================================================
# AIRSPACE GRID — the 4D discretization
# =============================================================================

@dataclass(frozen=True)
class AirspaceGrid:
    """
    4D discretization of the planning region.

    The grid is ALWAYS uniform in (x, y, time) and uses a discrete set of
    Flight Levels (FL340, FL360, etc.) for altitude. This matches how
    real airspace is organized: pilots fly at quantized levels, ATC
    organizes sectors by altitude bands, ISSRs are best resolved in
    horizontal slabs.

    Coordinate conventions:
        x: km, increasing east
        y: km, increasing north
        z: discrete FL list (e.g., [340, 360, 380, 400])
        t: seconds since flight planning window start (typically 0)

    Cell indices are 4-tuples (ix, iy, iz, it). The "physical center" of
    cell (ix, iy, iz, it) is at:
        x_center = x_min + (ix + 0.5) * dx_km
        y_center = y_min + (iy + 0.5) * dy_km
        z_center = fl_to_m(fl_bands[iz])
        t_center = (it + 0.5) * dt_s

    A cell's "ID" is the tuple itself; we use these as keys throughout.
    """
    # Horizontal extent (km).
    x_min_km: float
    x_max_km: float
    y_min_km: float
    y_max_km: float

    # Number of cells in each horizontal direction.
    nx: int
    ny: int

    # Allowed flight levels (RVSM grid). For the toy we use 2,000-ft
    # spacing: [340, 360, 380, 400]. Future work: 1,000-ft RVSM grid.
    fl_bands: tuple[int, ...]

    # Time-bucket discretization (seconds).
    dt_s: float    = 300.0          # 5 minutes per bucket
    nt:   int      = 6              # 6 buckets = 30-minute planning window

    # -------------------------------------------------------------------------
    # Derived properties — cell sizes, centers, shape
    # -------------------------------------------------------------------------

    @property
    def dx_km(self) -> float:
        return (self.x_max_km - self.x_min_km) / self.nx

    @property
    def dy_km(self) -> float:
        return (self.y_max_km - self.y_min_km) / self.ny

    @property
    def n_alts(self) -> int:
        return len(self.fl_bands)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Shape of any numpy array indexed by cell: (nx, ny, n_alts, nt)."""
        return (self.nx, self.ny, self.n_alts, self.nt)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.n_alts * self.nt

    # -------------------------------------------------------------------------
    # Coordinate <-> index conversions
    # -------------------------------------------------------------------------

    def cell_index_xy(self, x_km: float, y_km: float) -> tuple[int, int]:
        """
        Find the (ix, iy) of the cell containing this (x, y) point.

        Out-of-bounds points are CLAMPED to the nearest edge cell. This
        matches the physical intuition: a flight extending past the
        planning region still has its last bit attributable to the edge
        of the region.
        """
        ix = int(np.clip(np.floor((x_km - self.x_min_km) / self.dx_km),
                         0, self.nx - 1))
        iy = int(np.clip(np.floor((y_km - self.y_min_km) / self.dy_km),
                         0, self.ny - 1))
        return ix, iy

    def cell_index_fl(self, fl: int) -> int:
        """Find the FL-band index for a given Flight Level."""
        # We use the nearest allowed FL, not a strict containment.
        return int(np.argmin(np.abs(np.array(self.fl_bands) - fl)))

    def cell_index_z(self, z_m: float) -> int:
        """Find the FL-band index for a given altitude in meters."""
        return self.cell_index_fl(m_to_fl(z_m))

    def cell_index_t(self, t_s: float) -> int:
        """Find the time-bucket index for a given timestamp (seconds)."""
        return int(np.clip(np.floor(t_s / self.dt_s), 0, self.nt - 1))

    def cell_index(self, x_km: float, y_km: float, z_m: float,
                   t_s: float) -> tuple[int, int, int, int]:
        """Full 4D cell index from physical coordinates."""
        ix, iy = self.cell_index_xy(x_km, y_km)
        iz = self.cell_index_z(z_m)
        it = self.cell_index_t(t_s)
        return ix, iy, iz, it

    def cell_center(self, ix: int, iy: int, iz: int,
                    it: int) -> tuple[float, float, float, float]:
        """Physical center of a 4D cell, in (km, km, m, s)."""
        x = self.x_min_km + (ix + 0.5) * self.dx_km
        y = self.y_min_km + (iy + 0.5) * self.dy_km
        z = fl_to_m(self.fl_bands[iz])
        t = (it + 0.5) * self.dt_s
        return x, y, z, t

    # -------------------------------------------------------------------------
    # Numpy mesh helpers — for visualization and bulk ISSR queries
    # -------------------------------------------------------------------------

    def meshgrid_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """2D meshgrid of cell-center (x_km, y_km) values."""
        x = self.x_min_km + (np.arange(self.nx) + 0.5) * self.dx_km
        y = self.y_min_km + (np.arange(self.ny) + 0.5) * self.dy_km
        return np.meshgrid(x, y, indexing="ij")


# =============================================================================
# SECTORS AND CAPACITY BUCKETS
# =============================================================================

@dataclass(frozen=True)
class Sector:
    """
    One ATC sector: a region of airspace with a maximum simultaneous
    aircraft count.

    In real life, sector definitions are complex 3D polygons published
    by air navigation service providers (e.g., DSNA in France, DFS in
    Germany). For the toy, a sector is just:
        - a list of horizontal cell indices (which (ix, iy) it covers)
        - an altitude band (which iz indices it includes)
        - a name and a capacity

    A SECTOR'S CAPACITY-BUCKET is the pair (sector, time-bucket): the
    capacity constraint applies to each (sector, time-bucket) pair
    independently, NOT to the sector as a whole over all time.
    """
    name: str
    xy_cells: frozenset[tuple[int, int]]
    iz_range: tuple[int, int]    # (iz_min_inclusive, iz_max_inclusive)
    capacity: int                # max aircraft per (sector, time bucket)


@dataclass(frozen=True)
class SectorMap:
    """
    Container for all sectors in a planning region.

    Builds a fast lookup table mapping each 4D cell to its sector id
    (or -1 if uncovered, which the optimizer treats as "no capacity
    constraint applies").
    """
    grid: AirspaceGrid
    sectors: list[Sector]

    # Filled in __post_init__: shape == grid.shape, entries == sector index
    # in self.sectors, or -1 for uncovered cells.
    _cell_to_sector: np.ndarray = field(init=False)

    def __post_init__(self):
        # -1 means "no sector owns this cell" (uncovered airspace).
        cs = np.full(self.grid.shape, -1, dtype=np.int32)
        for s_idx, sec in enumerate(self.sectors):
            for ix, iy in sec.xy_cells:
                # Skip cells that fall outside the grid boundary.
                if 0 <= ix < self.grid.nx and 0 <= iy < self.grid.ny:
                    iz_lo, iz_hi = sec.iz_range
                    # Clamp to valid altitude indices in case the sector
                    # definition extends beyond the grid's altitude range.
                    iz_lo = max(0, iz_lo)
                    iz_hi = min(self.grid.n_alts - 1, iz_hi)
                    # A sector covers ALL time buckets in its xy/z region.
                    cs[ix, iy, iz_lo:iz_hi+1, :] = s_idx
        # Use object.__setattr__ because @dataclass is  frozen.
        object.__setattr__(self, '_cell_to_sector', cs)

    def sector_at(self, ix: int, iy: int, iz: int, it: int) -> int:
        """Return sector index for a given cell, or -1 if uncovered."""
        return int(self._cell_to_sector[ix, iy, iz, it])

    def bucket_id(self, ix: int, iy: int, iz: int,
                  it: int) -> tuple[int, int] | None:
        """
        Return the (sector_index, time_bucket_index) for this cell.

        Returns None if the cell is not covered by any sector — in which
        case no capacity constraint applies. The optimizer treats None
        as "no capacity bucket to charge to."
        """
        s_idx = self.sector_at(ix, iy, iz, it)
        if s_idx < 0:
            return None
        return (s_idx, it)

    def all_buckets(self) -> list[tuple[int, int]]:
        """
        Enumerate every active (sector, time-bucket) pair.

        These are the b indices in the QUBO capacity constraints
        sum_{(f,k)} a_{f,k,b} x_{f,k} <= cap_b.
        """
        out = []
        for s_idx in range(len(self.sectors)):
            for it in range(self.grid.nt):
                out.append((s_idx, it))
        return out

    def capacity_of(self, sector_idx: int, time_bucket_idx: int) -> int:
        """Look up the capacity for a (sector, bucket) pair."""
        # In this simple model, capacity depends only on the sector,
        # not on the time of day. Real ATFM has hour-varying capacities;
        # extension would be a 2D capacity table.
        return self.sectors[sector_idx].capacity


# =============================================================================
# CONVENIENCE CONSTRUCTORS
# =============================================================================

def uniform_sector_grid(
    grid: AirspaceGrid,
    cells_per_sector_x: int = 3,
    cells_per_sector_y: int = 3,
    altitude_bands_per_sector: int = 2,
    capacity: int = 3,
) -> SectorMap:
    """
    Partition the grid into a uniform tiling of "default" sectors,
    each capacity-limited identically.

    This is the simplest sector model: every group of (3 x 3) horizontal
    cells x (2 altitude bands) is one sector with capacity 3. Real
    European sectors vary in size from <100 km^2 (terminal areas) to
    >10,000 km^2 (oceanic), but a uniform tiling is the right starting
    point for a synthetic study.
    """
    sectors: list[Sector] = []
    # Tile xy
    for sx in range(0, grid.nx, cells_per_sector_x):
        for sy in range(0, grid.ny, cells_per_sector_y):
            xy_cells = frozenset(
                (ix, iy)
                for ix in range(sx, min(sx + cells_per_sector_x, grid.nx))
                for iy in range(sy, min(sy + cells_per_sector_y, grid.ny))
            )
            # Tile altitude bands
            for sz in range(0, grid.n_alts, altitude_bands_per_sector):
                iz_lo = sz
                iz_hi = min(sz + altitude_bands_per_sector - 1,
                            grid.n_alts - 1)
                fl_range = (grid.fl_bands[iz_lo],
                            grid.fl_bands[iz_hi])
                name = (f"S_{sx:02d}_{sy:02d}_FL{fl_range[0]}"
                        f"-FL{fl_range[1]}")
                sectors.append(Sector(
                    name=name,
                    xy_cells=xy_cells,
                    iz_range=(iz_lo, iz_hi),
                    capacity=capacity,
                ))
    return SectorMap(grid=grid, sectors=sectors)


# =============================================================================
# VOXELIZATION — turn a continuous trajectory into discrete cell IDs
# =============================================================================

def voxelize_trajectory(
    grid: AirspaceGrid,
    waypoints: Sequence[tuple[float, float, float, float]],
) -> list[tuple[int, int, int, int]]:
    """
    Convert a continuous trajectory (a list of physical waypoints)
    into the sequence of 4D cells it occupies.

    Each waypoint is (x_km, y_km, z_m, t_s). The output is a list of
    (ix, iy, iz, it) tuples in trajectory order, with CONSECUTIVE
    DUPLICATES removed (we care about cell membership, not how long
    we sat in each one).

    Why not insert intermediate cells when two waypoints are far apart?
    Because mid-fidelity: if your waypoints are spaced 5 minutes apart
    (~75 km at cruise) and your cells are 25 km, you'd skip 2-3 cells
    between waypoints. For the toy that's fine — the ISSR field is
    smooth and the conflict-edge construction only cares about which
    cells you DEMONSTRABLY enter.

    For higher-fidelity: replace this with Bresenham's line algorithm
    in 4D, or interpolate waypoints to <= 1 cell diameter spacing
    before voxelizing.
    """
    cells: list[tuple[int, int, int, int]] = []
    last_cell: tuple[int, int, int, int] | None = None

    for (x_km, y_km, z_m, t_s) in waypoints:
        c = grid.cell_index(x_km, y_km, z_m, t_s)
        if c != last_cell:
            cells.append(c)
            last_cell = c
    return cells


def densify_waypoints(
    waypoints: Sequence[tuple[float, float, float, float]],
    max_step_km: float = 25.0,
) -> list[tuple[float, float, float, float]]:
    """
    Linearly interpolate a waypoint list so consecutive samples are at
    most `max_step_km` apart in the horizontal plane. Preserves the
    original waypoints, just inserts more in between.

    Use this BEFORE voxelize_trajectory() if your waypoint spacing is
    larger than your cell size, otherwise voxelization will skip cells.
    """
    out: list[tuple[float, float, float, float]] = []
    for i in range(len(waypoints) - 1):
        x0, y0, z0, t0 = waypoints[i]
        x1, y1, z1, t1 = waypoints[i + 1]
        dist_km = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        n_sub = max(1, int(np.ceil(dist_km / max_step_km)))
        for k in range(n_sub):
            f = k / n_sub
            out.append((
                x0 + f * (x1 - x0),
                y0 + f * (y1 - y0),
                z0 + f * (z1 - z0),
                t0 + f * (t1 - t0),
            ))
    # Add the final waypoint
    out.append(tuple(waypoints[-1]))
    return out


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    grid = AirspaceGrid(
        x_min_km=0.0, x_max_km=1500.0,
        y_min_km=0.0, y_max_km=800.0,
        nx=60, ny=32,
        fl_bands=(340, 360, 380, 400),
        dt_s=300.0, nt=6,
    )

    print(f"Grid:")
    print(f"  shape (nx, ny, n_alts, nt) = {grid.shape}")
    print(f"  total cells               = {grid.n_cells}")
    print(f"  dx, dy                    = {grid.dx_km:.1f}, {grid.dy_km:.1f} km")
    print(f"  FL bands                  = {grid.fl_bands}")
    print(f"  time buckets              = {grid.nt} x {grid.dt_s}s "
          f"= {grid.nt * grid.dt_s / 60:.0f} min total")

    # Test snapping
    c = grid.cell_index(750.0, 400.0, 10973.0, 600.0)
    cc = grid.cell_center(*c)
    print(f"\n(750 km, 400 km, FL360, t=600s) snaps to cell {c}")
    print(f"  center at ({cc[0]:.1f}, {cc[1]:.1f}, {cc[2]:.0f}m, {cc[3]:.0f}s)")

    # Build a uniform sector map
    sec_map = uniform_sector_grid(grid)
    print(f"\nSectorMap: {len(sec_map.sectors)} sectors")
    print(f"  first: {sec_map.sectors[0].name}, "
          f"covers {len(sec_map.sectors[0].xy_cells)} xy-cells, "
          f"cap={sec_map.sectors[0].capacity}")

    # Test voxelization
    traj = [(0.0, 400.0, 10973.0, 0.0),
            (375.0, 400.0, 10973.0, 1500.0),
            (750.0, 400.0, 10973.0, 3000.0),
            (1125.0, 400.0, 10973.0, 4500.0),
            (1500.0, 400.0, 10973.0, 6000.0)]
    dense = densify_waypoints(traj, max_step_km=25.0)
    cells = voxelize_trajectory(grid, dense)
    print(f"\nVoxelization of 5-waypoint trajectory:")
    print(f"  densified to {len(dense)} samples")
    print(f"  passes through {len(cells)} unique cells")

    print("\nAll self-tests passed.")
