"""
run.py — End-to-end experiment: brute force vs random vs analog-QAOA.

For each seed: build a scenario, assemble the QUBO, find the exact
optimum, run the uniform-random baseline, run the analog-QAOA solver,
and write one CSV row per (seed, solver). Fixed seeds give
byte-identical results.

    python run.py --flights 4 --options 3 --seeds 10 --shots 1000 --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from contrail_env import (
    approximation_ratio,
    assemble_qubo,
    brute_force_optimum,
    make_scenario,
    mean_random_cost,
    solve_pasqal_analog,
    solve_random,
)

FIELDS = [
    "seed", "solver", "n_vars", "E_min", "best_cost",
    "approx_ratio", "feasibility_rate", "wall_clock_s",
]


def make_row(
    seed: int,
    solver: str,
    n_vars: int,
    e_min: float,
    e_rand_mean: float,
    best_cost: float,
    feas: float,
    wall: float,
) -> dict[str, object]:
    return {
        "seed": seed,
        "solver": solver,
        "n_vars": n_vars,
        "E_min": round(e_min, 3),
        "best_cost": round(best_cost, 3),
        "approx_ratio": round(approximation_ratio(best_cost, e_min, e_rand_mean), 4),
        "feasibility_rate": round(feas, 4),
        "wall_clock_s": round(wall, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="contrail QUBO experiment")
    parser.add_argument("--flights", type=int, default=4)
    parser.add_argument("--options", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--bo-iters", type=int, default=16)
    parser.add_argument("--csv", default="results.csv")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for seed in range(args.seeds):
        scenario = make_scenario(args.flights, args.options, seed)
        qubo = assemble_qubo(scenario)

        t0 = time.perf_counter()
        _z_opt, e_min = brute_force_optimum(qubo)
        brute_wall = time.perf_counter() - t0
        e_rand_mean = mean_random_cost(scenario)

        rand = solve_random(scenario, qubo, n_samples=args.shots, seed=seed)
        pasqal = solve_pasqal_analog(
            scenario, qubo, n_shots=args.shots, bo_iters=args.bo_iters, seed=seed
        )
        n = scenario.n_vars
        rows.append(make_row(seed, "brute-force", n, e_min, e_rand_mean, e_min, 1.0, brute_wall))
        rows.append(make_row(seed, rand.solver, n, e_min, e_rand_mean,
                             rand.best_cost, rand.feasibility_rate, rand.wall_clock_s))
        rows.append(make_row(seed, pasqal.solver, n, e_min, e_rand_mean,
                             pasqal.best_cost, pasqal.feasibility_rate, pasqal.wall_clock_s))
        print(f"seed {seed}: E_min = {e_min:.2f}, "
              f"random = {rand.best_cost:.2f}, pasqal = {pasqal.best_cost:.2f}")

    with open(args.csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.csv}\n")

    # Summary: mean +/- std per solver.
    print(f"{'solver':<14} {'approx_ratio':>16} {'feasibility':>16} {'wall_clock_s':>16}")
    for solver in ("brute-force", "random", "pasqal-analog"):
        sub = [r for r in rows if r["solver"] == solver]
        for_col = {
            col: (
                float(np.mean([float(str(r[col])) for r in sub])),
                float(np.std([float(str(r[col])) for r in sub])),
            )
            for col in ("approx_ratio", "feasibility_rate", "wall_clock_s")
        }
        print(
            f"{solver:<14}"
            f" {for_col['approx_ratio'][0]:8.3f} ± {for_col['approx_ratio'][1]:5.3f}"
            f" {for_col['feasibility_rate'][0]:8.3f} ± {for_col['feasibility_rate'][1]:5.3f}"
            f" {for_col['wall_clock_s'][0]:8.3f} ± {for_col['wall_clock_s'][1]:5.3f}"
        )


if __name__ == "__main__":
    main()
