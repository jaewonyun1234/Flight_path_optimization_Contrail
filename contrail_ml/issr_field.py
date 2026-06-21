"""
issr_field.py — MLIssrField: the model, dressed up as an ISSRField.

THE SEAM (plan §1)
==================
Everything in the optimizer that touches ISSR goes through the duck-typed
`ISSRField` interface from `contrail_env.synthetic_issr`:

    rhi_excess(x_km, y_km, z_m) -> float
    is_inside(x_km, y_km, z_m) -> bool
    rhi_excess_grid(x, y, z: ndarray) -> ndarray
    mask_grid(x, y, z: ndarray) -> ndarray
    threshold: float

`World.is_issr_cell` and `qubo.build_conflict_graph` consume ONLY that
interface. So the trained model integrates by implementing the same contract —
no change to `World` or `qubo.py`. This file is that implementation.

The model lives on a real geographic grid `(lon, lat, pressure)`. When the
optimizer asks about a local sim point `(x_km, y_km, z_m)`, we map it through
the shared `GeoAnchor` (x/y -> lon/lat) and the standard atmosphere
(z_m -> pressure), then interpolate the predicted field. The local<->geo
transform is the SAME one the features and the GUI use, so everyone agrees on
where things are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .features import GeoAnchor, altitude_to_pressure_hpa

if TYPE_CHECKING:
    from .config import MLConfig

# Local sim box (x_km, y_km, z_m), matching ISSRField.domain.
_SIM_DOMAIN = (0.0, 1500.0, 0.0, 800.0, 9000.0, 12500.0)


@dataclass
class MLIssrField:
    """A predicted ISSR-risk field exposing the `ISSRField` interface.

    Holds the corrected-RHi-excess and calibrated P(ISSR) on an ascending
    `(lon, lat, pressure)` grid and interpolates them on query.

    Parameters
    ----------
    lon_axis, lat_axis, pressure_axis : 1-D ascending grid axes.
    rhi_excess_cube, p_issr_cube : arrays of shape (n_lon, n_lat, n_pressure).
    anchor : the geographic anchor (must match the one used for features/GUI).
    threshold : RHi-excess cut for the pure-RHi mode (parity with ISSRField).
    p_threshold : calibrated-probability cut for the default risk-aware mode.
    mode : "prob" (is_inside = P(ISSR) >= p_threshold) or "rhi"
           (is_inside = rhi_excess > threshold).
    """

    lon_axis: np.ndarray
    lat_axis: np.ndarray
    pressure_axis: np.ndarray
    rhi_excess_cube: np.ndarray
    p_issr_cube: np.ndarray
    anchor: GeoAnchor
    threshold: float = 0.3
    p_threshold: float = 0.5
    mode: str = "prob"
    domain: tuple[float, float, float, float, float, float] = _SIM_DOMAIN
    source: str = "ml"

    _rhi_interp: Any = field(default=None, repr=False)
    _p_interp: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from scipy.interpolate import RegularGridInterpolator

        axes = (np.asarray(self.lon_axis, dtype=float),
                np.asarray(self.lat_axis, dtype=float),
                np.asarray(self.pressure_axis, dtype=float))
        self._rhi_interp = RegularGridInterpolator(
            axes, np.asarray(self.rhi_excess_cube, dtype=float),
            bounds_error=False, fill_value=0.0,
        )
        self._p_interp = RegularGridInterpolator(
            axes, np.asarray(self.p_issr_cube, dtype=float),
            bounds_error=False, fill_value=0.0,
        )

    # ------------------------------------------------------------------ #
    # coordinate mapping
    # ------------------------------------------------------------------ #
    def _to_query_points(self, x_km, y_km, z_m) -> tuple[np.ndarray, tuple[int, ...]]:
        """Map local (x_km, y_km, z_m) to interpolator points (lon, lat, p)."""
        x = np.asarray(x_km, dtype=float)
        y = np.asarray(y_km, dtype=float)
        z = np.asarray(z_m, dtype=float)
        shape = np.broadcast(x, y, z).shape
        xb, yb, zb = np.broadcast_arrays(x, y, z)
        lon, lat = self.anchor.local_to_geo(xb.ravel(), yb.ravel())
        pressure = altitude_to_pressure_hpa(zb.ravel())
        pts = np.column_stack([np.asarray(lon), np.asarray(lat), np.asarray(pressure)])
        return pts, shape

    # ------------------------------------------------------------------ #
    # ISSRField interface — pointwise
    # ------------------------------------------------------------------ #
    def rhi_excess(self, x_km: float, y_km: float, z_m: float) -> float:
        pts, _ = self._to_query_points(x_km, y_km, z_m)
        return float(self._rhi_interp(pts)[0])

    def p_issr(self, x_km: float, y_km: float, z_m: float) -> float:
        pts, _ = self._to_query_points(x_km, y_km, z_m)
        return float(self._p_interp(pts)[0])

    def is_inside(self, x_km: float, y_km: float, z_m: float) -> bool:
        if self.mode == "rhi":
            return self.rhi_excess(x_km, y_km, z_m) > self.threshold
        return self.p_issr(x_km, y_km, z_m) >= self.p_threshold

    # ------------------------------------------------------------------ #
    # ISSRField interface — vectorized
    # ------------------------------------------------------------------ #
    def rhi_excess_grid(self, x_km: np.ndarray, y_km: np.ndarray,
                        z_m: np.ndarray) -> np.ndarray:
        pts, shape = self._to_query_points(x_km, y_km, z_m)
        return self._rhi_interp(pts).reshape(shape)

    def p_issr_grid(self, x_km: np.ndarray, y_km: np.ndarray,
                    z_m: np.ndarray) -> np.ndarray:
        pts, shape = self._to_query_points(x_km, y_km, z_m)
        return self._p_interp(pts).reshape(shape)

    def mask_grid(self, x_km: np.ndarray, y_km: np.ndarray,
                  z_m: np.ndarray) -> np.ndarray:
        if self.mode == "rhi":
            return self.rhi_excess_grid(x_km, y_km, z_m) > self.threshold
        return self.p_issr_grid(x_km, y_km, z_m) >= self.p_threshold


# ===========================================================================
# Factory — build an MLIssrField for a region/time (plan §7)
# ===========================================================================


def ml_issr_field(
    config: MLConfig | None = None,
    met_source: str = "gfs",
    time: str | None = None,
    *,
    model: Any = None,
    allow_synthetic: bool = False,
    grid_res_deg: float = 1.0,
    seed: int = 0,
) -> MLIssrField:
    """Build an `MLIssrField` for the configured region.

    met_source:
      "gfs" / "era5" — operational: load a registered model and real weather
          (where ISSR *will* be, from forecast, or replay from reanalysis).
      "synthetic"    — hermetic test/demo path: a clearly-labelled fake met
          cube and a model trained on the synthetic fallback. Requires
          allow_synthetic=True (the guard keeps fake weather out of real runs).

    Delegates the actual grid construction to `predict.py` so the data and
    model plumbing lives in one place.
    """
    from . import predict  # lazy: avoids a heavy import at module load
    from .config import DEFAULT_CONFIG

    cfg = config or DEFAULT_CONFIG
    if met_source == "synthetic":
        return predict.synthetic_ml_issr_field(
            cfg, model=model, allow_synthetic=allow_synthetic,
            grid_res_deg=grid_res_deg, seed=seed,
        )
    return predict.real_ml_issr_field(
        cfg, met_source=met_source, time=time, model=model,
        grid_res_deg=grid_res_deg,
    )
