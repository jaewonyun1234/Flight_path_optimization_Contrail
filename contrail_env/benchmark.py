"""
benchmark.py — Head-to-head benchmark protocol (plan §10).

For each seeded instance:

    1. Solve with CP-SAT       -> ground-truth optimum E* + wall clock.
    2. Run Pasqal analog-QAOA  -> best repaired cost E_P, feasibility, time.
    3. Run Xanadu GBS          -> best repaired cost E_X, feasibility, time.

and tabulate the three metrics of §10.2 per solver:

    approximation ratio   r = E* / E_solver   (1.0 = optimal; <= 1 always)
    wall-clock time       QUBO in -> bitstring out
    raw feasibility rate  fraction of pre-repair samples satisfying all
                          constraints

Statistical hygiene (§10.3): repeated over independent seeds, aggregated
with bootstrap 95% confidence intervals on the approximation ratio.

The scenario is supplied as a factory `seed -> (evals, conflicts, buckets)`
so this module stays decoupled from any caller: the GUI passes a factory
built from its ScenarioConfig; the CLI below builds one from contrail_env
defaults.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .flight import EvaluatedOption
from .options import build_and_evaluate_flight, build_random_flights
from .pasqal_analog import solve_pasqal_analog
from .quantum_common import BackendBudgetError, QuantumResult
from .qubo import (
    CapacityBucket,
    ConflictEdge,
    build_capacity_buckets,
    build_conflict_graph,
)
from .solver_cpsat import solve_cpsat
from .world import default_european_world
from .xanadu_gbs import solve_xanadu_gbs

ScenarioFactory = Callable[
    [int], tuple[list[EvaluatedOption], list[ConflictEdge], list[CapacityBucket]]
]
ProgressCallback = Callable[[str, int, float], None]
ResultCallback = Callable[[int, "SolverRun"], None]
# Liveness heartbeat: (message, fraction of the CURRENT solver cell in [0, 1]).
# Fires much more often than on_progress — ~20x per Schrödinger integration —
# so a GUI can prove the sweep is alive even when one BO eval takes minutes.
PhaseCallback = Callable[[str, float], None]

SOLVER_NAMES = ("cpsat", "pasqal-analog", "xanadu-gbs")


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class SolverRun:
    """One (solver, instance) cell of the benchmark table."""

    solver: str
    backend: str
    status: str                      # "OK" | "SKIPPED" | "FAILED"
    best_cost: float = math.nan
    approx_ratio: float = math.nan   # E* / best_cost
    feasibility_rate: float = math.nan
    wall_clock_s: float = math.nan
    n_samples: int = 0
    history: list[tuple[int, float]] = field(default_factory=list)
    note: str = ""


@dataclass
class InstanceResult:
    """All three solver runs on one seeded instance."""

    seed: int
    optimum: float
    n_options: int
    n_conflicts: int
    n_buckets: int
    runs: list[SolverRun] = field(default_factory=list)

    def run_for(self, solver: str) -> SolverRun | None:
        for run in self.runs:
            if run.solver == solver:
                return run
        return None


@dataclass
class SolverStats:
    """Aggregate of one solver across all benchmark instances."""

    solver: str
    n_ok: int
    ratio_mean: float
    ratio_ci_low: float
    ratio_ci_high: float
    feasibility_mean: float
    wall_clock_mean_s: float


@dataclass
class BenchmarkReport:
    """The full benchmark: per-instance rows + aggregation helpers."""

    instances: list[InstanceResult] = field(default_factory=list)

    def aggregate(self, rng_seed: int = 0) -> dict[str, SolverStats]:
        """Per-solver means with bootstrap 95% CIs on the approx ratio."""
        stats: dict[str, SolverStats] = {}
        for solver in SOLVER_NAMES:
            ratios: list[float] = []
            feas: list[float] = []
            walls: list[float] = []
            for inst in self.instances:
                run = inst.run_for(solver)
                if run is not None and run.status == "OK":
                    ratios.append(run.approx_ratio)
                    feas.append(run.feasibility_rate)
                    walls.append(run.wall_clock_s)
            if not ratios:
                continue
            lo, hi = bootstrap_ci(ratios, rng=np.random.default_rng(rng_seed))
            stats[solver] = SolverStats(
                solver=solver,
                n_ok=len(ratios),
                ratio_mean=float(np.mean(ratios)),
                ratio_ci_low=lo,
                ratio_ci_high=hi,
                feasibility_mean=float(np.mean(feas)),
                wall_clock_mean_s=float(np.mean(walls)),
            )
        return stats

    def format_table(self) -> str:
        """Plain-text summary table (one row per solver)."""
        lines = [
            f"{'solver':<14} {'n':>3} {'ratio':>7} {'95% CI':>17} "
            f"{'feas%':>7} {'wall':>9}",
        ]
        for s in self.aggregate().values():
            lines.append(
                f"{s.solver:<14} {s.n_ok:>3} {s.ratio_mean:>7.4f} "
                f"[{s.ratio_ci_low:.4f}, {s.ratio_ci_high:.4f}] "
                f"{100 * s.feasibility_mean:>6.1f} {1000 * s.wall_clock_mean_s:>7.0f}ms"
            )
        return "\n".join(lines)

    def to_csv(self, path: str) -> None:
        """One row per (seed, solver) — the raw data behind the paper plots."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "seed", "solver", "backend", "status", "optimum", "best_cost",
                "approx_ratio", "feasibility_rate", "wall_clock_s",
                "n_samples", "n_options", "n_conflicts", "note",
            ])
            for inst in self.instances:
                for run in inst.runs:
                    writer.writerow([
                        inst.seed, run.solver, run.backend, run.status,
                        f"{inst.optimum:.2f}", f"{run.best_cost:.2f}",
                        f"{run.approx_ratio:.6f}", f"{run.feasibility_rate:.4f}",
                        f"{run.wall_clock_s:.4f}", run.n_samples,
                        inst.n_options, inst.n_conflicts, run.note,
                    ])


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (§10.3)."""
    rng = rng or np.random.default_rng()
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


# =============================================================================
# THE PROTOCOL
# =============================================================================

def _quantum_to_run(result: QuantumResult, optimum: float, wall_clock_s: float) -> SolverRun:
    """Convert a QuantumResult into a benchmark cell."""
    return SolverRun(
        solver=result.solver,
        backend=result.backend,
        status="OK",
        best_cost=result.best_cost,
        approx_ratio=optimum / result.best_cost if result.best_cost > 0 else math.nan,
        feasibility_rate=result.feasibility_rate,
        wall_clock_s=wall_clock_s,
        n_samples=result.n_samples,
        history=list(result.history),
    )


def _run_instance(
    seed: int,
    evals: list[EvaluatedOption],
    conflicts: list[ConflictEdge],
    buckets: list[CapacityBucket],
    *,
    n_shots: int,
    bo_iters: int,
    cpsat_time_limit_s: float,
    pasqal_backend: str,
    xanadu_backend: str,
    on_progress: ProgressCallback | None,
    on_result: ResultCallback | None,
    on_phase: PhaseCallback | None,
) -> InstanceResult:
    """Steps 1-5 of the §10.1 protocol on one instance."""
    instance = InstanceResult(
        seed=seed,
        optimum=math.nan,
        n_options=len(evals),
        n_conflicts=len(conflicts),
        n_buckets=len(buckets),
    )

    def emit(run: SolverRun) -> None:
        instance.runs.append(run)
        if on_result is not None:
            on_result(seed, run)

    # --- 1. CP-SAT ground truth -----------------------------------------
    if on_phase is not None:
        on_phase(f"seed {seed}: CP-SAT proving the optimum ({len(evals)} options)", 0.0)
    cpsat_history: list[tuple[int, float]] = []

    def cpsat_progress(improvement: int, objective: float) -> None:
        cpsat_history.append((improvement, objective))
        if on_progress is not None:
            on_progress("cpsat", improvement, objective)

    cpsat = solve_cpsat(
        evals, conflicts, buckets,
        time_limit_s=cpsat_time_limit_s,
        on_progress=cpsat_progress,
    )
    instance.optimum = cpsat.objective
    emit(SolverRun(
        solver="cpsat",
        backend="ortools",
        status="OK" if cpsat.status in ("OPTIMAL", "FEASIBLE") else "FAILED",
        best_cost=cpsat.objective,
        approx_ratio=1.0,
        feasibility_rate=1.0,
        wall_clock_s=cpsat.wall_clock_s,
        n_samples=1,
        history=cpsat_history,
        note=cpsat.status,
    ))
    if instance.runs[0].status != "OK":
        return instance  # no ground truth -> ratios undefined; skip quantum

    # --- 2. Pasqal analog-QAOA + BO ---------------------------------------
    def pasqal_progress(step: int, cost: float) -> None:
        if on_progress is not None:
            on_progress("pasqal-analog", step, cost)

    def pasqal_phase(message: str, frac: float) -> None:
        if on_phase is not None:
            on_phase(f"seed {seed}: Pasqal {message}", frac)

    try:
        t0 = time.perf_counter()
        pasqal = solve_pasqal_analog(
            evals, conflicts, buckets,
            n_shots=n_shots, bo_iters=bo_iters, seed=seed,
            backend=pasqal_backend, on_progress=pasqal_progress,
            on_phase=pasqal_phase,
        )
        emit(_quantum_to_run(pasqal, instance.optimum, time.perf_counter() - t0))
    except BackendBudgetError as exc:
        emit(SolverRun(solver="pasqal-analog", backend="-", status="SKIPPED", note=str(exc)))
    except Exception as exc:  # surface, don't kill the sweep
        emit(SolverRun(solver="pasqal-analog", backend="-", status="FAILED", note=str(exc)))

    # --- 3. Xanadu GBS ------------------------------------------------------
    def xanadu_progress(step: int, cost: float) -> None:
        if on_progress is not None:
            on_progress("xanadu-gbs", step, cost)
        if on_phase is not None:
            on_phase(
                f"seed {seed}: GBS sampling {step}/{n_shots} subsets",
                min(1.0, step / n_shots),
            )

    try:
        t0 = time.perf_counter()
        xanadu = solve_xanadu_gbs(
            evals, conflicts, buckets,
            n_samples=n_shots, seed=seed,
            backend=xanadu_backend, on_progress=xanadu_progress,
        )
        emit(_quantum_to_run(xanadu, instance.optimum, time.perf_counter() - t0))
    except BackendBudgetError as exc:
        emit(SolverRun(solver="xanadu-gbs", backend="-", status="SKIPPED", note=str(exc)))
    except Exception as exc:
        emit(SolverRun(solver="xanadu-gbs", backend="-", status="FAILED", note=str(exc)))

    return instance


def run_benchmark(
    factory: ScenarioFactory,
    seeds: Sequence[int],
    *,
    n_shots: int = 1000,
    bo_iters: int = 15,
    cpsat_time_limit_s: float = 10.0,
    pasqal_backend: str = "auto",
    xanadu_backend: str = "auto",
    on_progress: ProgressCallback | None = None,
    on_result: ResultCallback | None = None,
    on_phase: PhaseCallback | None = None,
) -> BenchmarkReport:
    """Run the full §10.1 protocol over the given seeds.

    Callbacks (all optional, used by the GUI):
        on_progress(solver, step, best_cost_so_far) — live convergence;
        on_result(seed, run)                        — one finished cell;
        on_phase(message, cell_fraction)            — liveness heartbeat.
    """
    report = BenchmarkReport()
    for seed in seeds:
        if on_phase is not None:
            on_phase(f"seed {seed}: building the instance", 0.0)
        evals, conflicts, buckets = factory(seed)
        report.instances.append(_run_instance(
            seed, evals, conflicts, buckets,
            n_shots=n_shots,
            bo_iters=bo_iters,
            cpsat_time_limit_s=cpsat_time_limit_s,
            pasqal_backend=pasqal_backend,
            xanadu_backend=xanadu_backend,
            on_progress=on_progress,
            on_result=on_result,
            on_phase=on_phase,
        ))
    return report


# =============================================================================
# DEFAULT SCENARIO FACTORY + CLI
# =============================================================================

def default_scenario_factory(
    *,
    n_flights: int = 4,
    n_issr_blobs: int = 8,
    corridor_frac: float = 0.04,
    cost_weights: tuple[float, float, float] = (1.0, 5.0, 0.5),
) -> ScenarioFactory:
    """Factory over contrail_env defaults — what the CLI benchmarks."""

    def build(
        seed: int,
    ) -> tuple[list[EvaluatedOption], list[ConflictEdge], list[CapacityBucket]]:
        world = default_european_world(seed=seed, n_issr_blobs=n_issr_blobs)
        flights = build_random_flights(
            n_flights=n_flights, world=world, seed=seed,
            corridor_frac=corridor_frac, snapshot_window_s=(0.0, 300.0),
        )
        evals: list[EvaluatedOption] = []
        for flight in flights:
            evals.extend(
                build_and_evaluate_flight(flight, world, cost_weights=cost_weights)
            )
        conflicts = build_conflict_graph(evals, world)
        buckets = build_capacity_buckets(evals, world)
        return evals, conflicts, buckets

    return build


def main() -> None:
    parser = argparse.ArgumentParser(description="CP-SAT vs Pasqal vs Xanadu benchmark")
    parser.add_argument("--flights", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds (0..N-1)")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--bo-iters", type=int, default=15)
    parser.add_argument("--csv", type=str, default="", help="optional CSV output path")
    args = parser.parse_args()

    factory = default_scenario_factory(n_flights=args.flights)
    report = run_benchmark(
        factory,
        seeds=range(args.seeds),
        n_shots=args.shots,
        bo_iters=args.bo_iters,
    )
    print(report.format_table())
    if args.csv:
        report.to_csv(args.csv)
        print(f"\nraw rows written to {args.csv}")


if __name__ == "__main__":
    main()
