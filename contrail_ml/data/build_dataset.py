"""
build_dataset.py — Orchestrate IAGOS + ERA5 -> one versioned training table.

Pipeline: load IAGOS waypoints (truth) -> collocate ERA5 onto them (inputs) ->
compute the shared features -> attach labels (rhi_iagos, y_issr, residual delta)
-> enforce the schema -> write a content-addressed parquet plus a dataset_card
describing coverage and class balance. If DVC is available the parquet is
`dvc add`-ed; otherwise the content hash is the version.

The training table is the single source of truth every model run records (by
hash), so any reported metric is reproducible from a known dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING

import pandas as pd

from ..features import add_derived_features
from .schema import (
    LABEL_DELTA,
    LABEL_ISSR,
    LABEL_RHI,
    TABLE_COLUMNS,
    issr_label,
    validate_table,
)

if TYPE_CHECKING:
    from ..config import MLConfig


def build_training_table(cfg: MLConfig, *, sampling_weight: bool = False) -> pd.DataFrame:
    """Build the labelled training table from real IAGOS + ERA5 data."""
    from .collocate import add_sampling_weight, collocate_era5
    from .iagos import load_iagos_waypoints

    iagos = load_iagos_waypoints(cfg)
    collocated = collocate_era5(iagos, cfg)
    df = add_derived_features(collocated, pressure_col="pressure_hpa")

    # Labels: IAGOS measured RHi is the truth; the residual is what the model
    # learns; y_issr is the binary supervised target.
    df[LABEL_RHI] = collocated["rhi_iagos"].to_numpy()
    df[LABEL_DELTA] = df[LABEL_RHI].to_numpy() - df["rhi"].to_numpy()
    df[LABEL_ISSR] = issr_label(df[LABEL_RHI].to_numpy(), cfg.issr_rhi_threshold)

    df["qc_flag"] = collocated.get("qc_flag", 0)
    df["source"] = "iagos+era5"
    df = add_sampling_weight(df, enable=sampling_weight)

    table = df.loc[:, list(TABLE_COLUMNS)].reset_index(drop=True)
    validate_table(table)
    return table


def content_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()[:16]


def write_versioned(df: pd.DataFrame, cfg: MLConfig) -> tuple[str, str]:
    """Write the table to a content-addressed parquet + a dataset_card.json.

    Returns (parquet_path, card_path).
    """
    out_dir = os.path.join(cfg.data_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)
    digest = content_hash(df)
    parquet_path = os.path.join(out_dir, f"issr_training_{digest}.parquet")
    df.to_parquet(parquet_path, index=False)

    card = _dataset_card(df, cfg, digest)
    card_path = os.path.join(out_dir, f"issr_training_{digest}.card.json")
    with open(card_path, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2, default=str)

    _maybe_dvc_add(parquet_path)
    return parquet_path, card_path


def _dataset_card(df: pd.DataFrame, cfg: MLConfig, digest: str) -> dict:
    t = pd.to_datetime(df["time"])
    return {
        "hash": digest,
        "n_rows": int(len(df)),
        "time_start": str(t.min()),
        "time_end": str(t.max()),
        "lon_range": [float(df["lon"].min()), float(df["lon"].max())],
        "lat_range": [float(df["lat"].min()), float(df["lat"].max())],
        "pressure_levels_hpa": list(cfg.pressure_levels_hpa),
        "issr_rate": float(df[LABEL_ISSR].mean()),
        "rhi_bias_raw": float((df["rhi"] - df[LABEL_RHI]).mean()),
        "source": "iagos+era5",
        "schema": list(TABLE_COLUMNS),
    }


def _maybe_dvc_add(path: str) -> None:
    """Best-effort `dvc add` for data versioning; silent if DVC isn't set up."""
    import shutil
    import subprocess

    if shutil.which("dvc") is None:
        return
    try:
        subprocess.run(["dvc", "add", path], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def build_and_write(cfg: MLConfig, *, sampling_weight: bool = False) -> tuple[str, str]:
    df = build_training_table(cfg, sampling_weight=sampling_weight)
    return write_versioned(df, cfg)
