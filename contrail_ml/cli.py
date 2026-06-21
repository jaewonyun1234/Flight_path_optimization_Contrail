"""
cli.py — `python -m contrail_ml <command>` entry points (plan §9).

Commands
========
  build-dataset : IAGOS + ERA5 -> versioned parquet (+ dataset card).
                  --synthetic writes a fallback table for offline work.
  train         : CV, fit, calibrate, evaluate, log to MLflow, register Staging.
  evaluate      : recompute the model-vs-baselines table for a dataset.
  predict       : build an ISSR field for a region/time and save it (NPZ).
  serve-check   : build an MLIssrField, drop it into the world via issr_source
                  ="ml", run the existing CP-SAT solve — a smoke test of the
                  whole serving seam.
  monitor       : rolling skill + feature drift over a window; print a retrain
                  recommendation.

`--synthetic` keeps train/predict/serve-check hermetic (no network/credentials),
behind the same loud guards as the rest of the package.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd

from .config import MLConfig


def _cfg(args: argparse.Namespace) -> MLConfig:
    cfg = MLConfig.from_env()
    if getattr(args, "mlflow_uri", None):
        cfg = cfg.with_overrides(mlflow_tracking_uri=args.mlflow_uri)
    return cfg


# --------------------------------------------------------------------------- #
# build-dataset
# --------------------------------------------------------------------------- #
def cmd_build_dataset(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    if args.synthetic:
        from .data.synthetic_fallback import make_synthetic_training_table

        df = make_synthetic_training_table(n=args.n, seed=args.seed, allow_synthetic=True)
        import os

        os.makedirs(f"{cfg.data_dir}/processed", exist_ok=True)
        path = f"{cfg.data_dir}/processed/synthetic_{args.seed}.parquet"
        df.to_parquet(path, index=False)
        print(f"[synthetic] wrote {len(df)} rows -> {path}")
        return 0

    from .data.build_dataset import build_and_write

    parquet, card = build_and_write(cfg, sampling_weight=args.sampling_weight)
    print(f"wrote {parquet}\ncard  {card}")
    return 0


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    df = _load_dataset(cfg, args)
    from .train import run_training

    res = run_training(cfg, df, log_mlflow=not args.no_mlflow, do_cv=not args.no_cv)
    print("dataset_hash:", res.dataset_hash)
    if res.cv_metrics:
        print("cv:", {k: round(v, 4) for k, v in res.cv_metrics.items()})
    print("holdout:", {k: round(v, 4) for k, v in res.holdout_metrics.items()})
    print("\nmodel vs baselines:")
    print(res.comparison.round(4).to_string())
    if res.mlflow:
        print("\nmlflow:", res.mlflow)
    return 0


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    df = _load_dataset(cfg, args)
    from .baselines import default_baselines
    from .evaluate import comparison_table
    from .registry import load_model

    model = load_model(cfg, stage=args.stage)
    baselines = [b.fit(df) for b in default_baselines(cfg.issr_rhi_threshold)]
    table = comparison_table(df, model, baselines, p_threshold=cfg.issr_p_threshold)
    print(table.round(4).to_string())
    return 0


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #
def cmd_predict(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from .issr_field import ml_issr_field

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field = ml_issr_field(
            cfg,
            met_source="synthetic" if args.synthetic else args.met_source,
            time=args.time,
            allow_synthetic=args.synthetic,
            grid_res_deg=args.grid_res,
        )
    out = args.out or "issr_field.npz"
    np.savez(
        out,
        lon=field.lon_axis, lat=field.lat_axis, pressure=field.pressure_axis,
        rhi_excess=field.rhi_excess_cube, p_issr=field.p_issr_cube,
    )
    frac = float((field.p_issr_cube >= field.p_threshold).mean())
    print(f"wrote {out}  cube={field.rhi_excess_cube.shape}  "
          f"ISSR fraction={frac:.3f}")
    return 0


# --------------------------------------------------------------------------- #
# serve-check
# --------------------------------------------------------------------------- #
def cmd_serve_check(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from contrail_env import (
        build_and_evaluate_flight,
        build_capacity_buckets,
        build_conflict_graph,
        build_random_flights,
        default_european_world,
        solve_cpsat,
    )

    issr_kwargs: dict[str, Any] = dict(
        config=cfg, met_source="synthetic" if args.synthetic else args.met_source,
        allow_synthetic=args.synthetic, grid_res_deg=args.grid_res,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world = default_european_world(seed=args.seed, issr_source="ml",
                                       issr_kwargs=issr_kwargs)
        flights = build_random_flights(n_flights=args.n_flights, world=world,
                                       seed=args.seed, corridor_frac=0.05,
                                       snapshot_window_s=(0.0, 300.0))
    evals = []
    for f in flights:
        evals.extend(build_and_evaluate_flight(f, world))
    conflicts = build_conflict_graph(evals, world)
    buckets = build_capacity_buckets(evals, world)
    result = solve_cpsat(evals, conflicts, buckets, time_limit_s=10.0)

    print(f"ISSR source : ml ({issr_kwargs['met_source']})")
    print(f"status      : {result.status}")
    print(f"objective   : {result.objective:.1f}")
    print(f"conflicts   : {len(conflicts)}")
    print("chosen options:")
    for i in result.chosen_eval_indices:
        ev = evals[i]
        print(f"  {ev.flight_name}: option {ev.option_index}  "
              f"fuel={ev.fuel_kg:.0f}kg contrail={ev.contrail_cells} "
              f"disrupt={ev.disruption_FLmin:.1f}")
    return 0


# --------------------------------------------------------------------------- #
# monitor
# --------------------------------------------------------------------------- #
def cmd_monitor(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from .monitor import monitor
    from .registry import load_model

    reference = pd.read_parquet(args.reference)
    current = pd.read_parquet(args.current)
    model = load_model(cfg, stage=args.stage)
    report = monitor(model, reference, current, cfg)
    print(json.dumps(report.to_dict(), indent=2, default=float))
    return 0


# --------------------------------------------------------------------------- #
def _load_dataset(cfg: MLConfig, args: argparse.Namespace) -> pd.DataFrame:
    if getattr(args, "synthetic", False):
        from .data.synthetic_fallback import make_synthetic_training_table

        return make_synthetic_training_table(n=args.n, seed=args.seed, allow_synthetic=True)
    if not getattr(args, "data", None):
        raise SystemExit("provide --data <parquet> or --synthetic")
    return pd.read_parquet(args.data)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m contrail_ml")
    p.add_argument("--mlflow-uri", default=None, help="override MLflow tracking URI")
    sub = p.add_subparsers(dest="command", required=True)

    bd = sub.add_parser("build-dataset", help="build the training table")
    bd.add_argument("--synthetic", action="store_true")
    bd.add_argument("--n", type=int, default=5000)
    bd.add_argument("--seed", type=int, default=0)
    bd.add_argument("--sampling-weight", action="store_true")
    bd.set_defaults(func=cmd_build_dataset)

    tr = sub.add_parser("train", help="train + calibrate + register")
    tr.add_argument("--data", default=None, help="training parquet path")
    tr.add_argument("--synthetic", action="store_true")
    tr.add_argument("--n", type=int, default=8000)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--no-mlflow", action="store_true")
    tr.add_argument("--no-cv", action="store_true")
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate", help="model-vs-baselines table")
    ev.add_argument("--data", default=None)
    ev.add_argument("--synthetic", action="store_true")
    ev.add_argument("--n", type=int, default=8000)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--stage", default="Staging")
    ev.set_defaults(func=cmd_evaluate)

    pr = sub.add_parser("predict", help="build + save an ISSR field")
    pr.add_argument("--synthetic", action="store_true")
    pr.add_argument("--met-source", default="gfs", choices=["gfs", "era5"])
    pr.add_argument("--time", default=None)
    pr.add_argument("--grid-res", type=float, default=1.0)
    pr.add_argument("--out", default=None)
    pr.set_defaults(func=cmd_predict)

    sc = sub.add_parser("serve-check", help="solve with the ML ISSR field")
    sc.add_argument("--synthetic", action="store_true", default=True)
    sc.add_argument("--real", dest="synthetic", action="store_false")
    sc.add_argument("--met-source", default="gfs", choices=["gfs", "era5"])
    sc.add_argument("--grid-res", type=float, default=2.0)
    sc.add_argument("--seed", type=int, default=1)
    sc.add_argument("--n-flights", type=int, default=4)
    sc.set_defaults(func=cmd_serve_check)

    mo = sub.add_parser("monitor", help="rolling skill + drift")
    mo.add_argument("--reference", required=True, help="training reference parquet")
    mo.add_argument("--current", required=True, help="new labelled window parquet")
    mo.add_argument("--stage", default="Staging")
    mo.set_defaults(func=cmd_monitor)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
