"""GUI Constraint-fingerprint tab (tab 10): render from a synthetic report.

Headless pattern. The tab does no computation of its own — it aggregates the
per-instance SolverRun fingerprint means the benchmark produces — so a hand-
filled BenchmarkReport is enough to drive the grouped-bar render and the CSV
export. Also checks the placeholder before any report exists.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6")
pyqtgraph = pytest.importorskip("pyqtgraph")

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


def _report_with_fingerprints() -> BenchmarkReport:
    inst = InstanceResult(
        seed=0, optimum=100.0, n_options=9, n_conflicts=3, n_buckets=1,
        optimum_exact=100.0, n_ground_states=1,
    )
    # Hand-filled fingerprint means for all five solvers (deficit, excess,
    # conflict, overflow, repair) — distinct so the bars are visibly different.
    specs = {
        "cpsat": (0.0, 0.0, 0.0, 0.0, 0.0),
        "random-repair": (1.5, 0.3, 0.2, 0.1, 4.0),
        "sa": (0.1, 0.0, 0.0, 0.0, 1.0),
        "pasqal-analog": (0.9, 0.0, 0.0, 0.0, 2.0),
        "xanadu-gbs": (0.2, 1.1, 0.4, 0.6, 3.0),
    }
    for name in SOLVER_NAMES:
        deficit, excess, conflict, overflow, repair = specs[name]
        inst.runs.append(SolverRun(
            solver=name, backend="test", status="OK",
            best_cost=100.0, approx_ratio=1.0, feasibility_rate=0.5,
            onehot_deficit_mean=deficit, onehot_excess_mean=excess,
            conflict_viol_mean=conflict, capacity_overflow_mean=overflow,
            repair_dist_mean=repair,
        ))
    report = BenchmarkReport()
    report.instances.append(inst)
    return report


def test_canvas_holds_two_panels(qapp):
    # P1b: grouped violation bars + repair-distance bars share one canvas, so a
    # single Export PNG captures both panels.
    win = MainWindow()
    panels = [it for it in win.fp_glw.ci.items if isinstance(it, pyqtgraph.PlotItem)]
    assert len(panels) == 2
    assert win.fp_repair in panels


def test_placeholder_before_any_report(qapp):
    win = MainWindow()
    assert "run a benchmark" in win.fp_status.text()
    assert win._fp_bar_item is None


def test_refresh_builds_grouped_bars(qapp):
    win = MainWindow()
    win._bench_report = _report_with_fingerprints()
    win._refresh_fingerprint_tab()

    # Five solvers x four families -> 20 grouped bars in one BarGraphItem.
    assert win._fp_bar_item is not None
    assert len(win._fp_bar_item.opts["x"]) == 5 * 4
    assert win._fp_repair_item is not None
    assert win.fp_export_csv_btn.isEnabled()


def test_export_csv_header(qapp, tmp_path, monkeypatch):
    win = MainWindow()
    win._bench_report = _report_with_fingerprints()
    win._refresh_fingerprint_tab()

    out = tmp_path / "fp.csv"
    monkeypatch.setattr(
        "gui.app.QFileDialog.getSaveFileName", lambda *a, **k: (str(out), ""))
    win._on_export_fp_csv()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "solver,family,mean"
    assert len(lines) == 1 + 5 * 4   # header + 20 rows
