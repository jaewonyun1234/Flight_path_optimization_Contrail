"""GUI Benchmark tab (tab 7): the 9-column table + per-solver bars + exports.

Follows the headless pattern of test_gui_map_panel: offscreen Qt platform set
before any Qt import, then importorskip so the test skips cleanly without the
[gui] extra. No worker threads — a synthetic BenchmarkReport is fed straight to
the finished slot, so the render path is exercised without solving anything.
"""

import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from contrail_env.benchmark import (  # noqa: E402
    SOLVER_NAMES,
    BenchmarkReport,
    InstanceResult,
    SolverRun,
)
from gui.app import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _synthetic_report() -> BenchmarkReport:
    """One instance, all five solvers OK, with per-shot stats filled in."""
    inst = InstanceResult(
        seed=0, optimum=100.0, n_options=9, n_conflicts=3, n_buckets=1,
        optimum_exact=100.0,
    )
    specs = {
        "cpsat": (100.0, 1.0, 1.0, math.nan, math.nan),
        "random-repair": (110.0, 100.0 / 110.0, 0.10, 0.12, 3.0e-4),
        "sa": (100.0, 1.0, 1.0, 0.20, 5.0e-4),
        "pasqal-analog": (101.0, 100.0 / 101.0, 0.90, 0.30, 2.0e-3),
        "xanadu-gbs": (105.0, 100.0 / 105.0, 0.02, 0.05, 1.5e-3),
    }
    for name in SOLVER_NAMES:
        best, ratio, feas, succ, tts = specs[name]
        inst.runs.append(SolverRun(
            solver=name, backend="test", status="OK",
            best_cost=best, approx_ratio=ratio, feasibility_rate=feas,
            wall_clock_s=0.01, n_samples=100, history=[(0, best)],
            success_prob=succ, tts_sample_s=tts, tts_total_s=tts,
        ))
    report = BenchmarkReport()
    report.instances.append(inst)
    return report


def test_benchmark_table_is_9_columns_by_5_rows(qapp):
    win = MainWindow()
    assert win.bench_table.columnCount() == 9
    assert win.bench_table.rowCount() == len(SOLVER_NAMES) == 5


def test_finished_slot_fills_table_and_enables_exports(qapp):
    win = MainWindow()
    win._on_bench_done(_synthetic_report())

    # Succ % column (index 6) of a sampler row parses as a float (sa p_s=0.20).
    sa_row = SOLVER_NAMES.index("sa")
    succ_text = win.bench_table.item(sa_row, 6).text()
    assert float(succ_text) == pytest.approx(20.0)

    # TTS column (index 7) is present and non-empty.
    assert win.bench_table.item(sa_row, 7).text()

    # Exports light up once a report exists, and the report is stored for G3.
    assert win.bench_export_csv_btn.isEnabled()
    assert win.bench_export_png_btn.isEnabled()
    assert win._bench_report is not None


def test_export_bench_csv_writes_file(qapp, tmp_path, monkeypatch):
    win = MainWindow()
    win._on_bench_done(_synthetic_report())
    out = tmp_path / "bench.csv"
    monkeypatch.setattr(
        "gui.app.QFileDialog.getSaveFileName", lambda *a, **k: (str(out), "")
    )
    win._on_export_bench_csv()
    assert out.exists()
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert "success_prob" in header and "tts_sample_s" in header
