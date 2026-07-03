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
from typing import cast

import numpy as np

from .analysis import feasible_cost_landscape, ground_state_degeneracy
from .classical_baselines import solve_random_repair, solve_simulated_annealing
from .flight import EvaluatedOption
from .metrics import success_probability, time_to_solution
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

SOLVER_NAMES = ("cpsat", "random-repair", "sa", "pasqal-analog", "xanadu-gbs")


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
    # Per-shot statistics (§S2). success_prob = fraction of repaired shots that
    # hit E*_exact; tts_sample_s uses only the sampling wall clock; tts_total_s
    # amortizes the full pipeline (the gap is the BO / tuning overhead).
    success_prob: float = math.nan
    tts_sample_s: float = math.nan
    tts_total_s: float = math.nan
    # Constraint-violation fingerprint means (§S5), from the RAW pre-repair
    # batch. NaN for CP-SAT (no raw samples) and for FAILED/SKIPPED cells.
    onehot_deficit_mean: float = math.nan
    onehot_excess_mean: float = math.nan
    conflict_viol_mean: float = math.nan
    capacity_overflow_mean: float = math.nan
    repair_dist_mean: float = math.nan
    # Solver-specific parameters (QuantumResult.meta, copied verbatim). The GUI
    # Dynamics tab reads T_ns/omega_max_rad_us/delta_init_rad_us/delta_final_rad_us
    # off a pasqal-analog run's meta for its "Load BO-best" button.
    meta: dict[str, object] = field(default_factory=dict)


@dataclass
class InstanceResult:
    """All solver runs on one seeded instance."""

    seed: int
    optimum: float
    n_options: int
    n_conflicts: int
    n_buckets: int
    runs: list[SolverRun] = field(default_factory=list)
    # Exact optimum (sum of c_i over the CP-SAT choice), NOT the cent-quantized
    # objective — this is the success threshold for p_s (§S2). math.nan until
    # CP-SAT has solved.
    optimum_exact: float = math.nan
    # Exact ground-state degeneracy (§S5), from feasible_cost_landscape; None
    # when the instance is too large to enumerate (ValueError, small-instance
    # tool by construction).
    n_ground_states: int | None = None

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
    # §S2 aggregates. TTS is heavy-tailed (a single p_s = 0 seed makes it
    # infinite), so it is summarized with the median + IQR across seeds, never
    # a bootstrap mean. tts_* use the sampling-only TTS (tts_sample_s).
    success_mean: float = math.nan
    tts_median_s: float = math.nan
    tts_iqr_s: float = math.nan


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
            succ: list[float] = []
            tts: list[float] = []
            for inst in self.instances:
                run = inst.run_for(solver)
                if run is not None and run.status == "OK":
                    ratios.append(run.approx_ratio)
                    feas.append(run.feasibility_rate)
                    walls.append(run.wall_clock_s)
                    if not math.isnan(run.success_prob):
                        succ.append(run.success_prob)
                    if not math.isnan(run.tts_sample_s):
                        tts.append(run.tts_sample_s)
            if not ratios:
                continue
            lo, hi = bootstrap_ci(ratios, rng=np.random.default_rng(rng_seed))
            tts_median, tts_iqr = _tts_median_iqr(tts)
            stats[solver] = SolverStats(
                solver=solver,
                n_ok=len(ratios),
                ratio_mean=float(np.mean(ratios)),
                ratio_ci_low=lo,
                ratio_ci_high=hi,
                feasibility_mean=float(np.mean(feas)),
                wall_clock_mean_s=float(np.mean(walls)),
                success_mean=float(np.mean(succ)) if succ else math.nan,
                tts_median_s=tts_median,
                tts_iqr_s=tts_iqr,
            )
        return stats

    def format_table(self) -> str:
        """Plain-text summary table (one row per solver)."""
        lines = [
            f"{'solver':<14} {'n':>3} {'ratio':>7} {'95% CI':>17} "
            f"{'feas%':>7} {'succ%':>7} {'medTTS':>9} {'wall':>9}",
        ]
        for s in self.aggregate().values():
            succ = "—" if math.isnan(s.success_mean) else f"{100 * s.success_mean:.1f}"
            lines.append(
                f"{s.solver:<14} {s.n_ok:>3} {s.ratio_mean:>7.4f} "
                f"[{s.ratio_ci_low:.4f}, {s.ratio_ci_high:.4f}] "
                f"{100 * s.feasibility_mean:>6.1f} {succ:>7} "
                f"{_fmt_tts(s.tts_median_s):>9} {1000 * s.wall_clock_mean_s:>7.0f}ms"
            )
        return "\n".join(lines)

    def to_csv(self, path: str) -> None:
        """One row per (seed, solver) — the raw data behind the paper plots."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "seed", "solver", "backend", "status", "optimum", "optimum_exact",
                "best_cost", "approx_ratio", "feasibility_rate",
                "success_prob", "tts_sample_s", "tts_total_s", "wall_clock_s",
                "n_samples", "n_options", "n_conflicts", "n_ground_states",
                "onehot_deficit_mean", "onehot_excess_mean", "conflict_viol_mean",
                "capacity_overflow_mean", "repair_dist_mean", "note",
            ])
            for inst in self.instances:
                for run in inst.runs:
                    writer.writerow([
                        inst.seed, run.solver, run.backend, run.status,
                        f"{inst.optimum:.2f}", f"{inst.optimum_exact:.6f}",
                        f"{run.best_cost:.2f}",
                        f"{run.approx_ratio:.6f}", f"{run.feasibility_rate:.4f}",
                        f"{run.success_prob:.6f}", f"{run.tts_sample_s:.6g}",
                        f"{run.tts_total_s:.6g}",
                        f"{run.wall_clock_s:.4f}", run.n_samples,
                        inst.n_options, inst.n_conflicts,
                        "" if inst.n_ground_states is None else inst.n_ground_states,
                        f"{run.onehot_deficit_mean:.6g}", f"{run.onehot_excess_mean:.6g}",
                        f"{run.conflict_viol_mean:.6g}", f"{run.capacity_overflow_mean:.6g}",
                        f"{run.repair_dist_mean:.6g}", run.note,
                    ])


def _fmt_tts(tts_s: float) -> str:
    """Render a TTS for the text table: '∞' for infinite, else seconds."""
    if math.isnan(tts_s):
        return "—"
    if math.isinf(tts_s):
        return "∞"
    return f"{tts_s:.3g}s"


def _tts_median_iqr(values: Sequence[float]) -> tuple[float, float]:
    """Median + inter-quartile range of a heavy-tailed TTS sample.

    inf values (p_s = 0 seeds) are kept: numpy's median/percentile propagate
    them faithfully, so a solver that fails on most seeds gets an infinite
    median — which is the honest summary, not a silently dropped tail.
    """
    if not values:
        return math.nan, math.nan
    arr = np.asarray(values, dtype=float)
    median = float(np.median(arr))
    # np.percentile interpolates between neighbours, and that lerp evaluates
    # inf - inf = nan (a spurious "invalid value in subtract" RuntimeWarning,
    # and a nan IQR) when a quartile lands on the infinite tail (p_s = 0 seeds).
    # Silence just that check and report a non-finite spread as an infinite IQR
    # — the honest summary of a heavy tail, not a dropped value.
    with np.errstate(invalid="ignore"):
        q25, q75 = np.percentile(arr, [25, 75])
    spread = q75 - q25
    return median, float(spread) if np.isfinite(spread) else math.inf


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

def _approx_ratio(optimum: float, best_cost: float) -> float:
    """optimum / best_cost, with the degenerate zero-cost case defined.

    Under alpha_fuel = 0 an instance can have E* = 0 with a solver's best
    repaired cost also 0, making the raw ratio 0/0. Define it: both ~0 -> 1.0
    (the solver matched a free optimum); optimum ~0 but best > 0 -> 0.0.
    """
    if abs(best_cost) < 1e-12:
        return 1.0 if abs(optimum) < 1e-12 else optimum / best_cost
    return optimum / best_cost


def _quantum_to_run(
    result: QuantumResult, optimum: float, optimum_exact: float, wall_clock_s: float
) -> SolverRun:
    """Convert a QuantumResult into a benchmark cell, with §S2 per-shot stats."""
    # p_s and both TTS variants (sampling-only vs full-pipeline amortized).
    success = math.nan
    tts_sample = math.nan
    tts_total = math.nan
    if result.evaluation is not None and not math.isnan(optimum_exact):
        success = success_probability(result.evaluation, optimum_exact)
        n = max(1, result.n_samples)
        t_sample_wall = cast(float, result.meta.get("final_sampling_wall_clock_s", wall_clock_s))
        tts_sample = time_to_solution(float(t_sample_wall) / n, success)
        tts_total = time_to_solution(wall_clock_s / n, success)

    # §S5 constraint-violation fingerprint, from the RAW pre-repair batch.
    deficit = excess = conflict_v = overflow = repair_dist = math.nan
    if result.evaluation is not None and result.evaluation.fingerprint is not None:
        fp = result.evaluation.fingerprint
        deficit = fp.onehot_deficit.mean
        excess = fp.onehot_excess.mean
        conflict_v = fp.conflict.mean
        overflow = fp.capacity_overflow.mean
        repair_dist = fp.repair_distance.mean

    return SolverRun(
        solver=result.solver,
        backend=result.backend,
        status="OK",
        best_cost=result.best_cost,
        approx_ratio=_approx_ratio(optimum, result.best_cost),
        feasibility_rate=result.feasibility_rate,
        wall_clock_s=wall_clock_s,
        n_samples=result.n_samples,
        history=list(result.history),
        success_prob=success,
        tts_sample_s=tts_sample,
        tts_total_s=tts_total,
        onehot_deficit_mean=deficit,
        onehot_excess_mean=excess,
        conflict_viol_mean=conflict_v,
        capacity_overflow_mean=overflow,
        repair_dist_mean=repair_dist,
        meta=dict(result.meta),
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
    solvers: Sequence[str] | None,
    on_progress: ProgressCallback | None,
    on_result: ResultCallback | None,
    on_phase: PhaseCallback | None,
) -> InstanceResult:
    """Steps 1-5 of the §10.1 protocol on one instance.

    `solvers` filters the non-CP-SAT solvers (CP-SAT always runs — it defines
    the exact optimum E*). None runs every solver in SOLVER_NAMES.
    """
    def enabled(name: str) -> bool:
        return solvers is None or name in solvers
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
    # Exact optimum on the same float path as repaired sample costs (NOT the
    # cent-quantized objective) — this is the success threshold for p_s (§S2).
    instance.optimum_exact = float(
        sum(evals[i].cost_combined for i in cpsat.chosen_eval_indices)
    )
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

    # Exact ground-state degeneracy (§S5) — a small-instance tool by
    # construction; the enumerator refuses instances it can't afford.
    try:
        costs, _opt = feasible_cost_landscape(evals, conflicts, buckets)
        instance.n_ground_states = ground_state_degeneracy(costs, instance.optimum_exact)
    except ValueError:
        instance.n_ground_states = None

    # --- 2. Classical samplers (null control + SA) ------------------------
    # Same output budget (n_shots samples) and the same repair as the quantum
    # pipelines, so they compete on identical footing.
    def run_baseline(name: str, solver_fn: Callable[..., QuantumResult]) -> None:
        def baseline_progress(step: int, cost: float) -> None:
            if on_progress is not None:
                on_progress(name, step, cost)
            if on_phase is not None:
                on_phase(
                    f"seed {seed}: {name} sampling ({step}/{n_shots})",
                    min(1.0, step / max(1, n_shots)),
                )

        try:
            t0 = time.perf_counter()
            result = solver_fn(
                evals, conflicts, buckets,
                n_samples=n_shots, seed=seed, on_progress=baseline_progress,
            )
            emit(_quantum_to_run(
                result, instance.optimum, instance.optimum_exact, time.perf_counter() - t0,
            ))
        except Exception as exc:  # samplers have no budget cap, but stay defensive
            emit(SolverRun(solver=name, backend="-", status="FAILED", note=str(exc)))

    if enabled("random-repair"):
        run_baseline("random-repair", solve_random_repair)
    if enabled("sa"):
        run_baseline("sa", solve_simulated_annealing)

    # --- 3. Pasqal analog-QAOA + BO ---------------------------------------
    if enabled("pasqal-analog"):
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
            emit(_quantum_to_run(
                pasqal, instance.optimum, instance.optimum_exact, time.perf_counter() - t0,
            ))
        except BackendBudgetError as exc:
            emit(SolverRun(solver="pasqal-analog", backend="-", status="SKIPPED", note=str(exc)))
        except Exception as exc:  # surface, don't kill the sweep
            emit(SolverRun(solver="pasqal-analog", backend="-", status="FAILED", note=str(exc)))

    # --- 4. Xanadu GBS ------------------------------------------------------
    if enabled("xanadu-gbs"):
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
            emit(_quantum_to_run(
                xanadu, instance.optimum, instance.optimum_exact, time.perf_counter() - t0,
            ))
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
    solvers: Sequence[str] | None = None,
    on_progress: ProgressCallback | None = None,
    on_result: ResultCallback | None = None,
    on_phase: PhaseCallback | None = None,
) -> BenchmarkReport:
    """Run the full §10.1 protocol over the given seeds.

    `solvers` filters the non-CP-SAT solvers (CP-SAT always runs); None = all.

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
            solvers=solvers,
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
    parser.add_argument(
        "--solvers", type=str, default="",
        help="comma list filtering the non-CP-SAT solvers "
             "(e.g. sa,pasqal-analog); CP-SAT always runs",
    )
    args = parser.parse_args()

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()] or None
    factory = default_scenario_factory(n_flights=args.flights)
    report = run_benchmark(
        factory,
        seeds=range(args.seeds),
        n_shots=args.shots,
        bo_iters=args.bo_iters,
        solvers=solvers,
    )
    print(report.format_table())
    if args.csv:
        report.to_csv(args.csv)
        print(f"\nraw rows written to {args.csv}")


if __name__ == "__main__":
    main()
