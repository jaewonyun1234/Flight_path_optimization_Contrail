"""
era5.py — Pull ERA5 reanalysis over the region/levels/time (training input).

ERA5 is the best reconstruction of PAST weather (reanalysis): it assimilates
real observations, so it is the right source for the model INPUTS at training
time. We use the ARCO-ERA5 zarr mirror on a public Google Cloud bucket via
pycontrails, which needs NO Copernicus CDS credentials.

NOT hermetic: this needs the `[ml]` geo stack (pycontrails/xarray/zarr/gcsfs)
and network. Every entry point fails loudly with an actionable message if a
dependency or the data is missing — it never fabricates fields (plan §0.1). The
heavy imports are deferred into the functions so importing this module is cheap.

VERIFY AGAINST DOCS before a real run: the pycontrails ARCO-ERA5 loader class
and the exact CF variable names occasionally change between releases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..features import RAW_NWP_COLUMNS

if TYPE_CHECKING:
    from ..config import MLConfig
    from ..predict import MetCube

# Map our raw feature names -> ERA5/CF standard names used by pycontrails.
# (Vertical velocity in ERA5 is omega = Lagrangian tendency of pressure, Pa/s.)
ERA5_VARIABLES: dict[str, str] = {
    "T": "air_temperature",
    "specific_humidity": "specific_humidity",
    "u": "eastward_wind",
    "v": "northward_wind",
    "w": "lagrangian_tendency_of_air_pressure",
    "cloud_cover": "fraction_of_cloud_cover",
    "cloud_ice": "specific_cloud_ice_water_content",
}


def _require_pycontrails() -> Any:
    try:
        import pycontrails  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only with extras
        raise ImportError(
            "ERA5 access needs the [ml] extra (pycontrails, xarray, zarr, gcsfs). "
            "Install with: pip install -e \".[ml]\". See docs/DATA.md."
        ) from exc
    from pycontrails.datalib.ecmwf import arco_era5

    return arco_era5


def open_era5(
    cfg: MLConfig,
    time: Any,
    *,
    variables: tuple[str, ...] = RAW_NWP_COLUMNS,
):
    """Open an ARCO-ERA5 MetDataset for `time` over the configured region/levels.

    `time` is a single timestamp or a (start, end) pair. Returns a pycontrails
    MetDataset (xarray-backed). Variable selection is restricted to the
    ERA5∩GFS intersection (RAW_NWP_COLUMNS) so nothing is missing at serve time.
    """
    arco_era5 = _require_pycontrails()
    cf_vars = [ERA5_VARIABLES[v] for v in variables]
    # NOTE: verify the exact constructor/kwargs against the installed
    # pycontrails version before a real pull.
    met = arco_era5.open_arco_era5(
        time=time,
        variables=cf_vars,
        pressure_levels=list(cfg.pressure_levels_hpa),
    )
    return _subset_region(met, cfg)


def _subset_region(met: Any, cfg: MLConfig) -> Any:
    """Crop a MetDataset/xarray to the configured lon/lat box."""
    ds = getattr(met, "data", met)
    lon_min, lon_max, lat_min, lat_max = cfg.bbox
    return ds.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))


def load_met_cube(
    cfg: MLConfig,
    time: str | None = None,
    *,
    grid_res_deg: float = 1.0,
) -> MetCube:
    """Load a single-time ERA5 snapshot as a serving MetCube (retrospective
    replay alternative to GFS)."""

    t = time or cfg.ref_time
    ds = _subset_region(open_era5(cfg, t), cfg)
    return _dataset_to_metcube(ds, cfg, grid_res_deg, t)


def _dataset_to_metcube(ds: Any, cfg: MLConfig, grid_res_deg: float, t: Any) -> MetCube:
    """Regrid an xarray dataset onto an ascending (lon, lat, pressure) grid and
    pack it into a MetCube."""
    from ..predict import MetCube

    lon_axis = np.arange(cfg.lon_min, cfg.lon_max + 1e-6, grid_res_deg)
    lat_axis = np.arange(cfg.lat_min, cfg.lat_max + 1e-6, grid_res_deg)
    pressure_axis = np.array(sorted(cfg.pressure_levels_hpa), dtype=float)

    interp = ds.interp(longitude=lon_axis, latitude=lat_axis, level=pressure_axis)
    fields: dict[str, np.ndarray] = {}
    for name in RAW_NWP_COLUMNS:
        cf = ERA5_VARIABLES[name]
        da = interp[cf]
        # squeeze any singleton time dim and order axes (lon, lat, level)
        arr = np.asarray(da.transpose("longitude", "latitude", "level").values, dtype=float)
        fields[name] = arr
    return MetCube(lon_axis=lon_axis, lat_axis=lat_axis, pressure_axis=pressure_axis,
                   fields=fields, time=t)
