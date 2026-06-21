"""
gfs.py — Pull a GFS forecast for the serving features (plan §7).

At serve time you need to know where ISSR WILL be when the aircraft arrives, not
where it is now — so the operational input is a FORECAST. GFS is NOAA's global
forecast model: free, public-domain, refreshed every 6 hours, delivered as GRIB
(hence cfgrib in the [ml] extra). pycontrails ships a `GFSForecast` loader.

Same honesty contract as era5.py: lazy heavy imports, loud actionable failure
if deps/data are missing, never fabricate. Features are restricted to the
ERA5∩GFS intersection so the model never sees a missing column.

VERIFY AGAINST DOCS: GFS variable names / the GFSForecast signature before a
real pull.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..features import RAW_NWP_COLUMNS

if TYPE_CHECKING:
    from ..config import MLConfig
    from ..predict import MetCube

# GFS exposes the same physical fields under its own names. w is the GFS
# vertical velocity (Pa/s), matching ERA5's omega so the feature is consistent.
GFS_VARIABLES: dict[str, str] = {
    "T": "air_temperature",
    "specific_humidity": "specific_humidity",
    "u": "eastward_wind",
    "v": "northward_wind",
    "w": "lagrangian_tendency_of_air_pressure",
    "cloud_cover": "fraction_of_cloud_cover",
    "cloud_ice": "specific_cloud_ice_water_content",
}


def _require_gfs() -> Any:
    try:
        from pycontrails.datalib.gfs import GFSForecast
    except ImportError as exc:  # pragma: no cover - needs extras
        raise ImportError(
            "GFS access needs the [ml] extra (pycontrails, cfgrib). Install with: "
            "pip install -e \".[ml]\". See docs/DATA.md."
        ) from exc
    return GFSForecast


def open_gfs(cfg: MLConfig, time: Any, *, variables: tuple[str, ...] = RAW_NWP_COLUMNS):
    """Open a GFSForecast MetDataset for `time` over the configured region/levels."""
    GFSForecast = _require_gfs()
    cf_vars = [GFS_VARIABLES[v] for v in variables]
    # NOTE: verify constructor kwargs against the installed pycontrails version.
    gfs = GFSForecast(
        time=time,
        variables=cf_vars,
        pressure_levels=list(cfg.pressure_levels_hpa),
    )
    met = gfs.open_metdataset()
    return getattr(met, "data", met)


def load_forecast_cube(
    cfg: MLConfig,
    time: str | None = None,
    *,
    grid_res_deg: float = 1.0,
) -> MetCube:
    """Load a single forecast valid-time as a serving MetCube."""
    from .era5 import _dataset_to_metcube  # shared regrid/pack logic

    t = time or cfg.ref_time
    ds = open_gfs(cfg, t)
    ds = ds.sel(
        longitude=slice(cfg.lon_min, cfg.lon_max),
        latitude=slice(cfg.lat_max, cfg.lat_min),
    )
    # GFS CF names match the ERA5 mapping values here, so the shared packer works.
    cube = _dataset_to_metcube(ds, cfg, grid_res_deg, t)
    return cube


def _gfs_names_match_era5() -> bool:
    """Guard used by the packer: the two mappings must share CF names so the
    shared regrid code can read either dataset."""
    from .era5 import ERA5_VARIABLES

    return all(GFS_VARIABLES[k] == ERA5_VARIABLES[k] for k in RAW_NWP_COLUMNS)
