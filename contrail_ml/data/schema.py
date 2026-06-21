"""
schema.py — Column contract for the training table.

One row = one IAGOS waypoint, with ERA5 fields collocated onto it. Keeping the
column names, dtypes and groupings in one place lets every stage
(build_dataset, model, evaluate, monitor) agree without guessing.

Groups
======
keys     : where/when the row is (for grouped CV and provenance).
features : the model inputs (defined in features.FEATURES — single source).
labels   : the supervised targets derived from the IAGOS measurement.
meta     : quality flags, source tag, optional sampling weight.
"""

from __future__ import annotations

from ..features import FEATURES

# ---- key columns ----------------------------------------------------------
KEY_COLUMNS: tuple[str, ...] = ("lat", "lon", "pressure_hpa", "time")

# ---- label columns --------------------------------------------------------
# rhi_iagos : measured RHi (%) — the ground truth, recomputed from IAGOS T/q/p
#             with the SAME thermodynamics as the features (no label skew).
# y_issr    : (rhi_iagos >= 100) as int — the classification target.
# delta     : rhi_iagos - rhi(nwp) — the residual the corrector predicts.
LABEL_RHI = "rhi_iagos"
LABEL_ISSR = "y_issr"
LABEL_DELTA = "delta"
LABEL_COLUMNS: tuple[str, ...] = (LABEL_RHI, LABEL_ISSR, LABEL_DELTA)

# ---- meta columns ---------------------------------------------------------
META_COLUMNS: tuple[str, ...] = ("qc_flag", "source", "sample_weight")

# Full ordered table schema.
TABLE_COLUMNS: tuple[str, ...] = (
    *KEY_COLUMNS,
    *FEATURES,
    *LABEL_COLUMNS,
    *META_COLUMNS,
)


def issr_label(rhi_iagos, rhi_threshold: float = 100.0):
    """y_issr = (measured RHi >= threshold). Works on scalars or arrays."""
    import numpy as np

    return (np.asarray(rhi_iagos, dtype=float) >= rhi_threshold).astype(int)


def validate_table(df) -> None:
    """Raise if the dataframe is missing any required column. A loud schema
    check beats a model that silently trains on the wrong thing."""
    missing = [c for c in TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"training table is missing required columns {missing}; "
            f"expected schema = {list(TABLE_COLUMNS)}"
        )
