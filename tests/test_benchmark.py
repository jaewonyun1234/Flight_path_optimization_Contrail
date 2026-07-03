"""Benchmark protocol: one tiny sweep end-to-end, plus the statistics."""

import csv
import math
import warnings

import pytest

from contrail_env import (
    bootstrap_ci,
    default_scenario_factory,
    run_benchmark,
)
from contrail_env.benchmark import SOLVER_NAMES, _approx_ratio, _tts_median_iqr

# Columns that legitimately vary run-to-run: TTS ∝ t_shot is a wall-clock-
# derived quantity (Rønnow et al. 2014), so it tracks machine load, not physics.
_TIMING_COLUMNS = {"wall_clock_s", "tts_sample_s", "tts_total_s"}


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


def test_determinism_science_columns(tmp_path):
    """Two identical-seed runs → byte-identical science columns in the CSV.

    Only the three timing-derived columns may differ (see _TIMING_COLUMNS).
    Micro scale so both runs finish well under a minute.
    """
    factory = default_scenario_factory(n_flights=3)
    kw = dict(seeds=[0], n_shots=120, bo_iters=3, cpsat_time_limit_s=2.0)

    rows_per_run: list[list[dict[str, str]]] = []
    for i in range(2):
        report = run_benchmark(factory, **kw)
        path = tmp_path / f"run{i}.csv"
        report.to_csv(str(path))
        rows_per_run.append(
            list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        )

    rows_a, rows_b = rows_per_run
    assert rows_a and len(rows_a) == len(rows_b)
    for ra, rb in zip(rows_a, rows_b, strict=True):
        assert ra.keys() == rb.keys()
        for col in ra:
            if col in _TIMING_COLUMNS:
                continue
            assert ra[col] == rb[col], f"non-deterministic science column {col!r}"


def test_lazy_benchmark_reexport():
    """PEP 562: benchmark symbols re-export lazily; unknown attrs still raise.

    The eager import was removed to stop runpy's RuntimeWarning on
    `python -m contrail_env.benchmark`; the public API must be unchanged.
    """
    import contrail_env

    assert contrail_env.run_benchmark is run_benchmark
    assert contrail_env.BenchmarkReport.__name__ == "BenchmarkReport"
    with pytest.raises(AttributeError):
        contrail_env.no_such_symbol  # noqa: B018


def test_tts_median_iqr_infinite_tail_is_warning_free():
    """A p_s = 0 seed makes TTS inf; the IQR is inf, computed without warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning becomes a failure
        median, iqr = _tts_median_iqr([math.inf])
    assert math.isinf(median) and math.isinf(iqr)
    # A finite sample still gets a finite IQR.
    _median, iqr2 = _tts_median_iqr([1.0, 3.0, 3.0, 5.0])
    assert math.isfinite(iqr2) and iqr2 > 0.0


def test_approx_ratio_zero_cost_edge_cases():
    """0/0 is a matched free optimum (1.0), not a nan; the usual case unchanged."""
    assert _approx_ratio(0.0, 0.0) == 1.0        # both ~0 → matched a free optimum
    assert _approx_ratio(0.0, 5.0) == 0.0        # optimum 0, best > 0
    assert _approx_ratio(10.0, 12.5) == pytest.approx(0.8)


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
