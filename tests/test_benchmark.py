"""Benchmark protocol: one tiny sweep end-to-end, plus the statistics."""

import math

from contrail_env import (
    bootstrap_ci,
    default_scenario_factory,
    run_benchmark,
)
from contrail_env.benchmark import SOLVER_NAMES


def test_benchmark_one_seed_all_solvers(tmp_path):
    factory = default_scenario_factory(n_flights=3)
    events: list[tuple[str, int, float]] = []
    cells: list[tuple[int, str]] = []

    report = run_benchmark(
        factory,
        seeds=[1],
        n_shots=200,
        bo_iters=5,
        on_progress=lambda solver, step, cost: events.append((solver, step, cost)),
        on_result=lambda seed, run: cells.append((seed, run.solver)),
    )

    assert len(report.instances) == 1
    inst = report.instances[0]
    assert {run.solver for run in inst.runs} == set(SOLVER_NAMES)
    assert cells == [(1, name) for name in SOLVER_NAMES]
    assert any(solver == "pasqal-analog" for solver, _s, _c in events)
    assert any(solver == "xanadu-gbs" for solver, _s, _c in events)

    cpsat = inst.run_for("cpsat")
    assert cpsat is not None and cpsat.status == "OK"
    assert cpsat.approx_ratio == 1.0

    for name in ("pasqal-analog", "xanadu-gbs"):
        run = inst.run_for(name)
        assert run is not None and run.status == "OK"
        # E* is the proven optimum, so r = E*/E <= 1 (cent-rounding slack).
        assert run.approx_ratio <= 1.0 + 1e-4
        assert 0.0 <= run.feasibility_rate <= 1.0
        assert run.wall_clock_s > 0
        assert run.history, "convergence curve must not be empty"

    # Aggregation + exports.
    stats = report.aggregate()
    assert set(stats) == set(SOLVER_NAMES)
    for s in stats.values():
        assert not math.isnan(s.ratio_mean)
        assert s.ratio_ci_low <= s.ratio_mean <= s.ratio_ci_high

    table = report.format_table()
    for name in SOLVER_NAMES:
        assert name in table

    csv_path = tmp_path / "bench.csv"
    report.to_csv(str(csv_path))
    content = csv_path.read_text(encoding="utf-8")
    assert content.count("\n") == 1 + len(SOLVER_NAMES)  # header + 3 rows


def test_bootstrap_ci_brackets_mean():
    import numpy as np

    rng = np.random.default_rng(0)
    values = list(rng.normal(loc=0.95, scale=0.01, size=20))
    lo, hi = bootstrap_ci(values, rng=np.random.default_rng(1))
    mean = float(np.mean(values))
    assert lo <= mean <= hi
    assert hi - lo < 0.02  # tight at n=20, sigma=0.01

    # Degenerate single observation: CI collapses onto the value.
    lo1, hi1 = bootstrap_ci([0.5])
    assert lo1 == hi1 == 0.5
