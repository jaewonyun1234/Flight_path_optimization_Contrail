"""GUI Quantum-dynamics tab (tab 9): render path + one real micro end-to-end.

Headless pattern (offscreen before Qt import, importorskip). The render slot is
fed synthetic AnalogSpectrum / DynamicsRecord objects — no solving — and then a
tiny real 2-qubit evolution is run SYNCHRONOUSLY (calling the pure science
functions directly, not a QThread) so the test stays deterministic and fast.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6")
pyqtgraph = pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from contrail_env.dynamics import DynamicsRecord, run_with_diagnostics  # noqa: E402
from contrail_env.pasqal_analog import AnnealSchedule  # noqa: E402
from contrail_env.quantum_common import build_option_graph  # noqa: E402
from contrail_env.spectral import AnalogSpectrum, instantaneous_spectrum  # noqa: E402
from gui.app import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _synthetic():
    s = np.linspace(0.02, 0.98, 5)
    energies = np.stack([s * 0 + level for level in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)], axis=1)
    gap = energies[:, 1] - energies[:, 0]
    spectrum = AnalogSpectrum(
        s=s, energies=energies, gap=gap, delta_min=float(gap.min()),
        s_star=0.5, end_degeneracy=1, n=2, k=6,
    )
    record = DynamicsRecord(
        s=s, t_ns=s * 4000.0,
        entropies={"flights_half": np.abs(np.sin(s)), "index_half": np.abs(np.cos(s))},
        p_ground=np.linspace(1.0, 0.9, 5), gap=gap, energy_pen=np.linspace(100.0, 10.0, 5),
        schedule={"T_ns": 4000.0, "omega_max": 10.0, "delta_init": -8.0, "delta_final": 8.0},
        cuts={"flights_half": (0,), "index_half": (0,)},
    )
    return spectrum, record


def test_canvas_holds_five_panels(qapp):
    # P1a: the residual ε(T) plot was merged into the main canvas (col 2,
    # rowspan 2), so one Export PNG now captures all five panels.
    win = MainWindow()
    panels = [it for it in win.dyn_glw.ci.items if isinstance(it, pyqtgraph.PlotItem)]
    assert len(panels) == 5
    assert win.dyn_resid in panels


def test_render_populates_four_cells(qapp):
    win = MainWindow()
    spectrum, record = _synthetic()
    win._render_dynamics(spectrum, record, min_epen=5.0, n_ground=1)

    for plot in (win.dyn_levels_plot, win.dyn_entropy_plot,
                 win.dyn_pground_plot, win.dyn_energy_plot):
        assert plot.listDataItems()
    # The curves actually carry the rendered data.
    assert len(win._dyn_gap_curve.getData()[0]) == 5
    assert len(win._dyn_pground_curve.getData()[0]) == 5
    # s* line present and positioned; info label carries the gap diagnostic.
    assert win.dyn_sstar_line.value() == pytest.approx(0.5)
    assert "Δ_min" in win.dyn_info.text()
    assert win.dyn_energy_min_line.isVisible()


def test_micro_end_to_end(qapp):
    win = MainWindow()
    win.flights_spin.setValue(2)
    win._rebuild_structure(win._build_cfg())
    evals, conflicts, buckets = win._evals, win._conflicts, win._buckets

    graph = build_option_graph(evals, conflicts, buckets)
    schedule = AnnealSchedule(T_ns=4000.0, omega_max=10.0, delta_init=-8.0, delta_final=8.0)
    spectrum = instantaneous_spectrum(graph, schedule, s_grid=np.linspace(0.05, 0.95, 9))
    record = run_with_diagnostics(graph, schedule, n_steps=80, n_records=8)
    win._render_dynamics(spectrum, record, min_epen=0.0, n_ground=1)

    assert win._dyn_spectrum is spectrum
    assert len(win._dyn_pground_curve.getData()[0]) == len(record.t_ns)
    assert 0.0 <= record.p_ground[-1] <= 1.0


def test_budget_guard_disables_compute(qapp):
    win = MainWindow()
    win._evals = list(range(18))  # pretend 18 option qubits
    win._update_dynamics_budget()
    assert not win.dyn_btn.isEnabled()
    assert "> 16" in win.dyn_status.text()
