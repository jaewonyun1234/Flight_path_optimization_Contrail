"""
app.py — Researcher-focused analytical dashboard for the contrail QUBO pipeline.

This is NOT a geographic / animation view. It analyzes the *problem* and the
*classical solver*: the live CP-SAT convergence stream, the conflict-graph
topology, the assembled QUBO matrix, and the cost trade-offs of the chosen
options. Quantum backends are future work, so everything here is about the
QUBO encoding and the CP-SAT ground-truth solve.

DATA FLOW
=========
* The solve runs on the gRPC server; the GUI reaches it through SolveWorker
  (a QThread) and streams progress over ZMQ — the UI never blocks.
* The server returns only the solution, so the dashboard rebuilds the QUBO and
  conflict graph LOCALLY from the same (seeded) ScenarioConfig via
  service.scenario.build_scenario_full + contrail_env.assemble_qubo. Because the
  scenario is fully seeded, the reconstruction matches what the server solved.

PANELS
======
1. Live CP-SAT convergence  — objective vs improvement index (ZMQ stream).
2. Conflict-graph topology   — option nodes grouped by flight, conflict edges.
3. QUBO & matrix analytics   — size, sparsity, penalty constants, energy offset,
                               and a heatmap of |Q|.
4. Trade-off results table   — chosen option per flight with fuel / contrail /
                               disruption, to read off the alpha,beta,gamma trade.
"""

from __future__ import annotations

import random
import sys
import threading
import time
from collections import OrderedDict

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from contrail_env import assemble_qubo
from service.client import DEFAULT_SERVER_ADDRESS, SolverClient
from service.generated import solver_pb2
from service.progress import DEFAULT_SUB_ADDRESS, subscribe
from service.scenario import build_scenario_full

# Fixed scenario knobs not exposed as controls (sensible defaults per the brief).
CORRIDOR_FRAC = 0.05
SNAPSHOT_WINDOW_S = 300.0
# Head start (seconds) for the SUB socket to connect before the solve emits.
_SUB_HEAD_START_S = 0.25

_MONO = QFont("Consolas", 10)


# =============================================================================
# WORKER THREAD  (unchanged contract: streams progress, returns SolveResponse)
# =============================================================================

class SolveWorker(QThread):
    """Runs one solve off the UI thread, streaming progress via signals."""

    progress = pyqtSignal(int, float)
    finished_ok = pyqtSignal(object)   # solver_pb2.SolveResponse
    failed = pyqtSignal(str)

    def __init__(
        self,
        cfg: solver_pb2.ScenarioConfig,
        server_address: str = DEFAULT_SERVER_ADDRESS,
        sub_address: str = DEFAULT_SUB_ADDRESS,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._server_address = server_address
        self._sub_address = sub_address

    def run(self) -> None:
        box: dict[str, object] = {}

        def do_solve() -> None:
            # Brief delay so the subscriber below connects first (slow joiner).
            time.sleep(_SUB_HEAD_START_S)
            try:
                with SolverClient(self._server_address) as client:
                    box["resp"] = client.solve(self._cfg)
            except Exception as exc:
                box["err"] = exc

        solve_thread = threading.Thread(target=do_solve, daemon=True)
        solve_thread.start()

        try:
            for improvement, objective in subscribe(
                self._cfg.progress_topic,
                self._sub_address,
                stop=lambda: not solve_thread.is_alive(),
            ):
                self.progress.emit(improvement, objective)
        except Exception as exc:
            box.setdefault("err", exc)

        solve_thread.join()
        if "err" in box:
            self.failed.emit(str(box["err"]))
        else:
            self.finished_ok.emit(box["resp"])


# =============================================================================
# MAIN WINDOW
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Contrail QUBO Dashboard — CP-SAT ground truth")

        self._worker: SolveWorker | None = None
        self._pending_cfg: solver_pb2.ScenarioConfig | None = None

        self._xs: list[int] = []
        self._ys: list[float] = []

        # Reconstructed problem state (set by _rebuild_structure).
        self._evals: list = []
        self._conflicts: list = []
        self._groups: OrderedDict[str, list[int]] = OrderedDict()
        self._graph_texts: list[pg.TextItem] = []

        self._build_ui()
        self._rebuild_structure(self._build_cfg())

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        pg.setConfigOptions(antialias=True)

        # --- Controls -------------------------------------------------------
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(42)

        self.flights_spin = QSpinBox()
        self.flights_spin.setRange(1, 12)
        self.flights_spin.setValue(4)

        self.blobs_spin = QSpinBox()
        self.blobs_spin.setRange(0, 40)
        self.blobs_spin.setValue(6)

        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 100.0)
        self.beta_spin.setSingleStep(0.5)
        self.beta_spin.setValue(5.0)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.05, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.3)

        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(0.1, 120.0)
        self.time_spin.setSingleStep(1.0)
        self.time_spin.setValue(10.0)

        self.preview_btn = QPushButton("Randomize / preview instance")
        self.preview_btn.clicked.connect(self._on_randomize)

        self.solve_btn = QPushButton("Solve")
        self.solve_btn.clicked.connect(self._on_solve)

        form = QFormLayout()
        form.addRow("Seed", self.seed_spin)
        form.addRow("Flights (F)", self.flights_spin)
        form.addRow("ISSR blobs", self.blobs_spin)
        form.addRow("Contrail weight (beta)", self.beta_spin)
        form.addRow("Contrail threshold (RHi)", self.threshold_spin)
        form.addRow("CP-SAT time limit (s)", self.time_spin)
        form.addRow(self.preview_btn)
        form.addRow(self.solve_btn)

        controls_box = QGroupBox("Instance / solver controls")
        controls_box.setLayout(form)

        self.summary = QLabel("idle")
        self.summary.setWordWrap(True)
        self.summary.setFont(_MONO)

        left = QVBoxLayout()
        left.addWidget(controls_box)
        left.addWidget(self.summary)
        left.addStretch(1)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(320)

        # --- Tabs -----------------------------------------------------------
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_convergence_tab(), "1 · Convergence")
        self.tabs.addTab(self._build_graph_tab(), "2 · Conflict graph")
        self.tabs.addTab(self._build_qubo_tab(), "3 · QUBO analytics")
        self.tabs.addTab(self._build_results_tab(), "4 · Trade-offs")

        root = QHBoxLayout()
        root.addWidget(left_widget)
        root.addWidget(self.tabs, stretch=1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    def _build_convergence_tab(self) -> QWidget:
        self.conv_plot = pg.PlotWidget(title="CP-SAT incumbent objective (live, via ZMQ)")
        self.conv_plot.setLabel("bottom", "improvement index")
        self.conv_plot.setLabel("left", "objective (combined cost)")
        self.conv_plot.showGrid(x=True, y=True, alpha=0.3)
        self.conv_curve = self.conv_plot.plot(
            [], [], pen=pg.mkPen("c", width=2), symbol="o", symbolSize=7
        )
        return self.conv_plot

    def _build_graph_tab(self) -> QWidget:
        self.graph_plot = pg.PlotWidget(
            title="Conflict graph — columns = flights, dots = options, lines = ISSR conflicts"
        )
        self.graph_plot.setLabel("bottom", "flight index")
        self.graph_plot.setLabel("left", "option index (k)")
        self.graph_plot.showGrid(x=True, y=True, alpha=0.15)
        self.edge_item = self.graph_plot.plot([], [])
        self.node_item = pg.ScatterPlotItem()
        self.graph_plot.addItem(self.node_item)
        return self.graph_plot

    def _build_qubo_tab(self) -> QWidget:
        self.qubo_stats = QLabel("—")
        self.qubo_stats.setFont(_MONO)
        self.qubo_stats.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        stats_box = QGroupBox("QUBOInstance")
        stats_layout = QVBoxLayout()
        stats_layout.addWidget(self.qubo_stats)
        stats_layout.addStretch(1)
        stats_box.setLayout(stats_layout)
        stats_box.setFixedWidth(330)

        self.qubo_view = pg.PlotWidget(title="|Q|  (upper-triangular QUBO matrix)")
        self.qubo_view.setLabel("bottom", "variable j")
        self.qubo_view.setLabel("left", "variable i")
        self.qubo_view.invertY(True)
        self.qubo_view.setAspectLocked(True)
        self.qubo_img = pg.ImageItem()
        try:
            self.qubo_img.setColorMap(pg.colormap.get("inferno"))
        except Exception:
            pass
        self.qubo_view.addItem(self.qubo_img)

        row = QHBoxLayout()
        row.addWidget(stats_box)
        row.addWidget(self.qubo_view, stretch=1)
        wrap = QWidget()
        wrap.setLayout(row)
        return wrap

    def _build_results_tab(self) -> QWidget:
        self.results = QTableWidget(0, 5)
        self.results.setHorizontalHeaderLabels(
            ["Flight", "Chosen opt#", "Fuel (kg)", "Contrail cells", "Disruption (FL-min)"]
        )
        self.results.horizontalHeader().setStretchLastSection(True)
        return self.results

    # ---------------------------------------------------------------- config --
    def _build_cfg(self, topic: str = "") -> solver_pb2.ScenarioConfig:
        return solver_pb2.ScenarioConfig(
            seed=self.seed_spin.value(),
            n_flights=self.flights_spin.value(),
            n_issr_blobs=self.blobs_spin.value(),
            alpha_fuel=1.0,
            beta_contrail=self.beta_spin.value(),
            gamma_disruption=0.5,
            corridor_frac=CORRIDOR_FRAC,
            snapshot_window_s=SNAPSHOT_WINDOW_S,
            time_limit_s=self.time_spin.value(),
            issr_threshold=self.threshold_spin.value(),
            progress_topic=topic,
        )

    # ------------------------------------------------- problem reconstruction --
    def _rebuild_structure(self, cfg: solver_pb2.ScenarioConfig) -> None:
        """Rebuild the QUBO + conflict graph locally and refresh panels 2 & 3."""
        try:
            _world, _flights, evals, conflicts, buckets = build_scenario_full(cfg)
            qubo = assemble_qubo(evals, conflicts, buckets)
        except Exception as exc:
            self.summary.setText(f"instance build failed:\n{exc}")
            return

        self._evals = evals
        self._conflicts = conflicts
        groups: OrderedDict[str, list[int]] = OrderedDict()
        for idx, ev in enumerate(evals):
            groups.setdefault(ev.flight_name, []).append(idx)
        self._groups = groups

        self._render_conflict_graph(set())
        self._render_qubo(qubo, len(conflicts), len(buckets))
        self.summary.setText(
            f"instance ready\n"
            f"F={len(groups)}  options={qubo.n_options}\n"
            f"conflicts={len(conflicts)}  buckets={len(buckets)}\n"
            f"N_total={qubo.n}\n"
            f"-> click Solve"
        )

    def _render_conflict_graph(self, chosen: set[int]) -> None:
        n = len(self._evals)
        if n == 0:
            self.node_item.setData([])
            self.edge_item.setData([], [])
            return

        pos = np.zeros((n, 2), dtype=float)
        for ff, members in enumerate(self._groups.values()):
            for kk, idx in enumerate(members):
                pos[idx] = (ff, kk)

        # Conflict edges as disconnected segments (two points per edge).
        ex: list[float] = []
        ey: list[float] = []
        for e in self._conflicts:
            ex += [pos[e.i][0], pos[e.j][0]]
            ey += [pos[e.i][1], pos[e.j][1]]
        self.edge_item.setData(
            x=np.array(ex), y=np.array(ey),
            connect="pairs", pen=pg.mkPen((150, 150, 170, 120), width=1),
        )

        n_groups = max(1, len(self._groups))
        spots = []
        for ff, members in enumerate(self._groups.values()):
            color = pg.intColor(ff, hues=n_groups)
            for kk, idx in enumerate(members):
                hit = idx in chosen
                spots.append({
                    "pos": (ff, kk),
                    "brush": color,
                    "size": 22 if hit else 13,
                    "pen": pg.mkPen("w", width=2) if hit else pg.mkPen((20, 20, 20)),
                })
        self.node_item.setData(spots=spots)

        for t in self._graph_texts:
            self.graph_plot.removeItem(t)
        self._graph_texts = []
        for ff, name in enumerate(self._groups.keys()):
            label = pg.TextItem(name, color=(210, 210, 210), anchor=(0.5, 1.0))
            label.setPos(ff, -0.5)
            self.graph_plot.addItem(label)
            self._graph_texts.append(label)

    def _render_qubo(self, qubo, n_conflicts: int, n_buckets: int) -> None:
        q = qubo.Q
        self.qubo_img.setImage(np.abs(q), autoLevels=True)

        n = qubo.n
        nonzero = int(np.count_nonzero(q))
        density = 100.0 * nonzero / (n * n) if n else 0.0
        self.qubo_stats.setText(
            f"N_total        : {n}\n"
            f"  option vars  : {qubo.n_options}\n"
            f"  slack bits   : {qubo.n_slack}\n"
            f"flights (F)    : {len(self._groups)}\n"
            f"conflict edges : {n_conflicts}\n"
            f"capacity bkts  : {n_buckets}\n"
            f"\n"
            f"Q non-zeros    : {nonzero} / {n * n}\n"
            f"Q density      : {density:.1f} %\n"
            f"(Q is upper-triangular)\n"
            f"\n"
            f"penalty A (1-hot)   : {qubo.penalty_A:.1f}\n"
            f"penalty B (conflict): {qubo.penalty_B:.1f}\n"
            f"penalty C (capacity): {qubo.penalty_C:.1f}\n"
            f"energy offset       : {qubo.constant:.1f}"
        )

    # ---------------------------------------------------------------- slots ---
    def _on_randomize(self) -> None:
        # New seed re-rolls the instance; preview its structure without solving.
        self.seed_spin.setValue(random.randint(0, 9999))
        self._xs, self._ys = [], []
        self.conv_curve.setData([], [])
        self.results.setRowCount(0)
        self._rebuild_structure(self._build_cfg())

    def _on_solve(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        topic = f"solve/{self.seed_spin.value()}-{int(time.time() * 1000)}"
        cfg = self._build_cfg(topic)
        self._pending_cfg = cfg

        self._rebuild_structure(cfg)
        self._xs, self._ys = [], []
        self.conv_curve.setData([], [])
        self.results.setRowCount(0)
        self.summary.setText("solving…")
        self.solve_btn.setEnabled(False)

        self._worker = SolveWorker(cfg)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, improvement: int, objective: float) -> None:
        self._xs.append(improvement)
        self._ys.append(objective)
        self.conv_curve.setData(self._xs, self._ys)

    def _on_finished(self, resp: object) -> None:
        choices = resp.choices  # type: ignore[attr-defined]

        self.results.setRowCount(len(choices))
        for row, c in enumerate(choices):
            values = [
                c.flight_name,
                str(c.chosen_option),
                f"{c.fuel_kg:.0f}",
                str(c.contrail_cells),
                f"{c.disruption_flmin:.1f}",
            ]
            for col, val in enumerate(values):
                self.results.setItem(row, col, QTableWidgetItem(val))

        self.summary.setText(
            f"E* = {resp.objective:.2f}\n"  # type: ignore[attr-defined]
            f"status = {resp.status}\n"  # type: ignore[attr-defined]
            f"wall = {resp.wall_clock_s * 1000:.0f} ms\n"  # type: ignore[attr-defined]
            f"conflicts = {resp.n_conflicts}\n"  # type: ignore[attr-defined]
            f"options = {resp.n_options_total}"  # type: ignore[attr-defined]
        )

        # Highlight the chosen option node in the conflict graph.
        chosen_idx: set[int] = set()
        for c in choices:
            members = self._groups.get(c.flight_name)
            if members and 0 <= c.chosen_option < len(members):
                chosen_idx.add(members[c.chosen_option])
        self._render_conflict_graph(chosen_idx)

        self.solve_btn.setEnabled(True)

    def _on_failed(self, message: str) -> None:
        self.summary.setText(f"solve failed:\n{message}")
        self.solve_btn.setEnabled(True)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1280, 760)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
