"""
run.py — End-to-end experiment: brute force vs greedy vs random vs analog-QAOA.

For each seed: build a scenario, assemble the QUBO, find the exact
optimum, run the baselines and the analog-QAOA solver, and write one CSV
row per (seed, solver). Fixed seeds give byte-identical results.

WHICH NUMBERS MATTER
====================
Raw QUBO energy (`raw_best_E`, `raw_mean_E`, penalties included) is the
PRIMARY evidence of sampler quality: the repair pass alone already
solves easy instances from pure noise, so repaired cost — and the
approximation ratio defined on it — measures the pipeline, not the
sampler. The repaired-cost table is still printed (secondary), with the
repair heuristic itself listed as the named solver "greedy".

Scenarios with fewer than --min-conflicts conflict edges are skipped
with a printed notice and deterministically replaced by seed + 1000
(then + 2000, ...); the CSV records the substituted seed.

    python run.py --flights 5 --options 3 --seeds 10 --shots 1000 --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from contrail_env import (
    SolveResult,
    approximation_ratio,
    assemble_qubo,
    brute_force_optimum,
    make_scenario,
    mean_random_cost,
    solve_greedy,
    solve_pasqal_analog,
    solve_random,
)

FIELDS = [
    "seed", "solver", "n_vars", "E_min", "best_cost",
    "approx_ratio", "feasibility_rate", "raw_best_E", "raw_mean_E",
    "wall_clock_s",
]


def pick_scenario(seed: int, n_flights: int, n_options: int, min_conflicts: int):
    """Return (actual_seed, scenario), substituting seed+1000, +2000, ...

    A documented, visible exclusion of trivial instances — not silent
    resampling inside the generator, which would bias the ensemble.
    """
    actual = seed
    while True:
        scenario = make_scenario(n_flights, n_options, actual)
        if len(scenario.conflicts) >= min_conflicts:
            return actual, scenario
        print(f"NOTICE: seed {actual} has {len(scenario.conflicts)} conflict(s) "
              f"< --min-conflicts {min_conflicts}; substituting seed {actual + 1000}")
        actual += 1000


def result_row(
    seed: int, n_vars: int, e_min: float, e_rand_mean: float, r: SolveResult
) -> dict[str, object]:
    return {
        "seed": seed,
        "solver": r.solver,
        "n_vars": n_vars,
        "E_min": round(e_min, 3),
        "best_cost": round(r.best_cost, 3),
        "approx_ratio": round(approximation_ratio(r.best_cost, e_min, e_rand_mean), 4),
        "feasibility_rate": round(r.feasibility_rate, 4),
        "raw_best_E": "" if r.raw_best_E is None else round(r.raw_best_E, 3),
        "raw_mean_E": "" if r.raw_mean_E is None else round(r.raw_mean_E, 3),
        "wall_clock_s": round(r.wall_clock_s, 3),
    }


def summarize(rows: list[dict[str, object]], solver: str, col: str) -> str:
    vals = [float(str(r[col])) for r in rows if r["solver"] == solver and r[col] != ""]
    if not vals:
        return f"{'—':>18}"
    return f"{np.mean(vals):10.3f} ± {np.std(vals):6.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="contrail QUBO experiment")
    parser.add_argument("--flights", type=int, default=4)
    parser.add_argument("--options", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--bo-iters", type=int, default=16)
    parser.add_argument("--min-conflicts", type=int, default=1)
    parser.add_argument("--csv", default="results.csv")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for requested in range(args.seeds):
        seed, scenario = pick_scenario(
            requested, args.flights, args.options, args.min_conflicts
        )
        qubo = assemble_qubo(scenario)

        t0 = time.perf_counter()
        z_opt, e_min = brute_force_optimum(qubo)
        brute = SolveResult(
            solver="brute-force", best_z=z_opt, best_cost=e_min,
            feasibility_rate=1.0, n_samples=1,
            wall_clock_s=time.perf_counter() - t0,
        )
        e_rand_mean = mean_random_cost(scenario)

        greedy = solve_greedy(scenario, qubo)
        rand = solve_random(scenario, qubo, n_samples=args.shots, seed=seed)
        pasqal = solve_pasqal_analog(
            scenario, qubo, n_shots=args.shots, bo_iters=args.bo_iters, seed=seed
        )
        for r in (brute, greedy, rand, pasqal):
            rows.append(result_row(seed, scenario.n_vars, e_min, e_rand_mean, r))
        print(f"seed {seed}: E_min = {e_min:.2f}, greedy = {greedy.best_cost:.2f}, "
              f"random = {rand.best_cost:.2f}, pasqal = {pasqal.best_cost:.2f} | "
              f"raw_best random = {rand.raw_best_E:.1f}, pasqal = {pasqal.raw_best_E:.1f}")

    with open(args.csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.csv}\n")

    print("RAW QUBO ENERGY (primary — what the sampler itself achieves; lower is better)")
    print(f"{'solver':<14} {'raw_best_E':>18} {'raw_mean_E':>18}")
    for solver in ("random", "pasqal-analog"):
        print(f"{solver:<14} {summarize(rows, solver, 'raw_best_E')} "
              f"{summarize(rows, solver, 'raw_mean_E')}")
    e_min_summary = summarize(rows, "brute-force", "E_min")
    print(f"{'(E_min ref)':<14} {e_min_summary}")

    print("\nREPAIRED COST (secondary — repair is itself the 'greedy' heuristic)")
    print(f"{'solver':<14} {'approx_ratio':>18} {'feasibility':>18} {'wall_clock_s':>18}")
    for solver in ("brute-force", "greedy", "random", "pasqal-analog"):
        print(f"{solver:<14} {summarize(rows, solver, 'approx_ratio')} "
              f"{summarize(rows, solver, 'feasibility_rate')} "
              f"{summarize(rows, solver, 'wall_clock_s')}")


if __name__ == "__main__":
    main()
