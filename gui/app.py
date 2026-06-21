"""
app.py — Researcher-focused analytical dashboard for the contrail QUBO pipeline.

Tabs 1-5 analyze the *problem* and the *solvers*: the live CP-SAT convergence
stream, the conflict-graph topology, the assembled QUBO matrix, the cost
trade-offs of the chosen options, and the head-to-head quantum benchmark (CP-SAT
vs Pasqal analog vs Xanadu GBS). Tab 6 is the one geographic view: the predicted
ISSR risk over Europe with the solved routes drawn on a real basemap.

DATA FLOW
=========
* The CP-SAT solve runs on the gRPC server; the GUI reaches it through
  SolveWorker (a QThread) and streams progress over ZMQ — the UI never blocks.
* The server returns only the solution, so the dashboard rebuilds the QUBO and
  conflict graph LOCALLY from the same (seeded) ScenarioConfig via
  service.scenario.build_scenario_full + contrail_env.assemble_qubo. Because the
  scenario is fully seeded, the reconstruction matches what the server solved.
* The quantum benchmark (tab 5) runs IN-PROCESS in a worker thread: the
  quantum samplers live in contrail_env (pasqal_analog, xanadu_gbs) and the
  protocol in contrail_env.benchmark — no service round-trip needed.

PANELS
======
1. Live CP-SAT convergence  — objective vs improvement index (ZMQ stream).
2. Conflict-graph topology   — option nodes grouped by flight, conflict edges.
3. QUBO & matrix analytics   — size, sparsity, penalty constants, energy offset,
                               and a heatmap of |Q|.
4. Trade-off results table   — chosen option per flight with fuel / contrail /
                               disruption, to read off the alpha,beta,gamma trade.
5. Quantum benchmark         — plan §10 protocol over N seeds: approximation
                               ratios with bootstrap CIs, raw feasibility rates,
                               wall clocks, and live convergence curves for the
                               Pasqal BO loop and the Xanadu GBS sampler.
6. Geographic map            — Plotly/MapLibre basemap with the ISSR risk as a
                               density overlay and the chosen vs context routes
                               (embedded QtWebEngine view; needs the [gui] extra).
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# QtWebEngine backs the geographic Map tab. It MUST be imported before any
# QApplication is constructed (Qt sets up shared OpenGL contexts at import), so
# it lives here at module scope, not lazily inside the tab. Degrades to a hint
# label if the [gui] extra (PyQt6-WebEngine) isn't installed.
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView

    _HAS_WEBENGINE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    QWebEngineView = None  # type: ignore[assignment, misc]
    _HAS_WEBENGINE = False

from contrail_env import assemble_qubo, fl_to_m, waypoints_for
from contrail_env.benchmark import SOLVER_NAMES, run_benchmark
from contrail_env.pasqal_analog import MAX_STATEVECTOR_QUBITS
from service.client import DEFAULT_SERVER_ADDRESS, SolverClient
from service.generated import solver_pb2
from service.progress import DEFAULT_SUB_ADDRESS, subscribe
from service.scenario import build_scenario_full

# Fixed scenario knobs not exposed as controls (sensible defaults per the brief).
CORRIDOR_FRAC = 0.05
SNAPSHOT_WINDOW_S = 300.0
# Head start (seconds) for the SUB socket to connect before the solve emits.
_SUB_HEAD_START_S = 0.25
# Altitude (FL) of the horizontal slice the map heatmap shows. A 2-D map can
# only show one altitude of the 3-D risk field; FL360 is the middle of the band.
_MAP_SLICE_FL = 360
# MapLibre basemap. A dark canvas makes the warm inferno ISSR overlay glow; it
# needs no API token. Tiles stream from the web — offline it falls back to a
# plain dark canvas, and the overlay + routes still render correctly.
_MAP_BASEMAP = "carto-darkmatter"
# Density-overlay grid resolution (lon x lat samples) and blur radius (px).
_MAP_GRID_NX, _MAP_GRID_NY = 130, 90
_MAP_DENSITY_RADIUS = 22

_MONO = QFont("Consolas", 10)


def _mean_best_cost(instances, solver: str) -> str:
    """Mean best cost of one solver across the benchmark instances."""
    costs = [
        run.best_cost
        for inst in instances
        for run in inst.runs
        if run.solver == solver and run.status == "OK"
    ]
    return f"{np.mean(costs):.1f}" if costs else "—"


# =============================================================================
# MAP FIGURE  (pure, no Qt — unit-testable on its own)
# =============================================================================

@dataclass(frozen=True)
class RouteLine:
    """One flight's ground track in geographic coordinates for the map."""

    name: str
    lon: np.ndarray
    lat: np.ndarray
    chosen: bool


def build_map_figure(
    *,
    source: str,
    lon: np.ndarray,
    lat: np.ndarray,
    risk: np.ndarray,
    routes: list[RouteLine],
):
    """Build the geographic Map figure: ISSR density overlay + flight routes.

    Pure function of plain arrays so it carries no Qt/web dependency and can be
    unit-tested directly. `lon`/`lat`/`risk` are the (same-shape) sampled risk
    field; `routes` are per-flight ground tracks already in lon/lat. Returns a
    Plotly Figure positioned over the data's bounding box on a MapLibre basemap.
    """
    import plotly.graph_objects as go  # lazy: keep module import lean

    lon_f = np.asarray(lon, dtype=float).ravel()
    lat_f = np.asarray(lat, dtype=float).ravel()
    risk_f = np.asarray(risk, dtype=float).ravel()

    fig = go.Figure()

    # ISSR risk as a smooth density overlay. radius blurs the grid samples into
    # a continuous field; only positive risk is drawn so clear sky stays clean.
    fig.add_trace(
        go.Densitymap(
            lon=lon_f,
            lat=lat_f,
            z=risk_f,
            radius=_MAP_DENSITY_RADIUS,
            colorscale="Inferno",
            opacity=0.55,
            colorbar=dict(title="ISSR<br>risk", thickness=12, len=0.6, x=0.99),
            name="ISSR risk",
            hoverinfo="skip",
        )
    )

    # Routes: muted gray for context/baseline, bright green for the chosen plan.
    for r in routes:
        fig.add_trace(
            go.Scattermap(
                lon=np.asarray(r.lon, dtype=float),
                lat=np.asarray(r.lat, dtype=float),
                mode="lines",
                line=dict(
                    width=3 if r.chosen else 1.5,
                    color="#28dc5a" if r.chosen else "rgba(200,200,210,0.55)",
                ),
                name=f"{r.name} (chosen)" if r.chosen else r.name,
                hovertemplate=f"{r.name}<extra></extra>",
            )
        )

    center_lon = float(np.nanmean(lon_f)) if lon_f.size else 5.0
    center_lat = float(np.nanmean(lat_f)) if lat_f.size else 46.5
    fig.update_layout(
        map=dict(style=_MAP_BASEMAP, center=dict(lon=center_lon, lat=center_lat), zoom=4.0),
        margin=dict(l=0, r=0, t=34, b=0),
        title=dict(text=f"ISSR risk + routes — source: {source} (slice FL{_MAP_SLICE_FL})", x=0.5),
        paper_bgcolor="#0e0e10",
        font=dict(color="#dddde2"),
        legend=dict(bgcolor="rgba(20,20,24,0.6)", x=0.01, y=0.99),
    )
    return fig


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
# BENCHMARK WORKER — runs the plan §10 protocol locally, off the UI thread
# =============================================================================

class BenchmarkWorker(QThread):
    """One full benchmark sweep: CP-SAT vs Pasqal vs Xanadu over N seeds.

    The scenario factory reuses the SAME seeded construction as the rest of
    the dashboard (service.scenario.build_scenario_full), so the instances
    benchmarked here are exactly the ones the other tabs display.
    """

    progress = pyqtSignal(str, int, float)   # solver, step, best cost so far
    cell_done = pyqtSignal(int, object)      # seed, contrail_env.benchmark.SolverRun
    phase = pyqtSignal(str, float)           # heartbeat: message, cell fraction [0,1]
    finished_ok = pyqtSignal(object)         # contrail_env.benchmark.BenchmarkReport
    failed = pyqtSignal(str)

    def __init__(
        self,
        cfg: solver_pb2.ScenarioConfig,
        seeds: list[int],
        n_shots: int,
        bo_iters: int,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._seeds = seeds
        self._n_shots = n_shots
        self._bo_iters = bo_iters

    def run(self) -> None:
        def factory(seed: int):
            cfg = solver_pb2.ScenarioConfig()
            cfg.CopyFrom(self._cfg)
            cfg.seed = seed
            _world, _flights, evals, conflicts, buckets = build_scenario_full(cfg)
            return evals, conflicts, buckets

        try:
            report = run_benchmark(
                factory,
                seeds=self._seeds,
                n_shots=self._n_shots,
                bo_iters=self._bo_iters,
                on_progress=lambda solver, step, cost: self.progress.emit(solver, step, cost),
                on_result=lambda seed, cell: self.cell_done.emit(seed, cell),
                on_phase=lambda message, frac: self.phase.emit(message, float(frac)),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished_ok.emit(report)


# =============================================================================
# MAIN WINDOW
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Contrail QUBO Dashboard — CP-SAT ground truth")

        self._worker: SolveWorker | None = None
        self._bench_worker: BenchmarkWorker | None = None
        self._pending_cfg: solver_pb2.ScenarioConfig | None = None

        # Benchmark progress state (see _on_bench_phase / _refresh_bench_status).
        self._bench_t0 = 0.0
        self._bench_cells_done = 0
        self._bench_msg = ""

        self._xs: list[int] = []
        self._ys: list[float] = []

        # Reconstructed problem state (set by _rebuild_structure).
        self._evals: list = []
        self._conflicts: list = []
        self._groups: OrderedDict[str, list[int]] = OrderedDict()
        self._graph_texts: list[pg.TextItem] = []
        # World + flights kept for the geographic map panel (tab 6).
        self._world: object | None = None
        self._flights: list = []

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
        self.tabs.addTab(self._build_benchmark_tab(), "5 · Quantum benchmark")
        self.tabs.addTab(self._build_map_tab(), "6 · Map")

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

    def _build_benchmark_tab(self) -> QWidget:
        # --- controls row ----------------------------------------------------
        self.bench_seeds_spin = QSpinBox()
        self.bench_seeds_spin.setRange(1, 10)
        self.bench_seeds_spin.setValue(3)

        self.bench_shots_spin = QSpinBox()
        self.bench_shots_spin.setRange(100, 5000)
        self.bench_shots_spin.setSingleStep(100)
        self.bench_shots_spin.setValue(1000)

        self.bench_bo_spin = QSpinBox()
        self.bench_bo_spin.setRange(5, 40)
        self.bench_bo_spin.setValue(12)

        self.bench_btn = QPushButton("Run benchmark")
        self.bench_btn.clicked.connect(self._on_run_benchmark)

        self.bench_status = QLabel(
            "idle — runs CP-SAT, Pasqal analog-QAOA, and Xanadu GBS on the "
            "current instance over N seeds (seed, seed+1, …)"
        )
        self.bench_status.setFont(_MONO)

        # Overall sweep progress: 100 units per (seed, solver) cell, with the
        # heartbeat filling the current cell fractionally — so the bar moves
        # even during a single multi-minute Schrödinger integration.
        self.bench_progress = QProgressBar()
        self.bench_progress.setRange(0, 1)
        self.bench_progress.setValue(0)
        self.bench_progress.setFormat("%p%")
        self.bench_progress.setFixedWidth(220)

        self._bench_timer = QTimer(self)
        self._bench_timer.setInterval(1000)
        self._bench_timer.timeout.connect(self._refresh_bench_status)

        controls = QHBoxLayout()
        for label, widget in (
            ("Seeds", self.bench_seeds_spin),
            ("Samples / shots", self.bench_shots_spin),
            ("BO iterations", self.bench_bo_spin),
        ):
            controls.addWidget(QLabel(label))
            controls.addWidget(widget)
        controls.addWidget(self.bench_btn)
        controls.addWidget(self.bench_progress)
        controls.addWidget(self.bench_status, stretch=1)

        # --- aggregate table ---------------------------------------------------
        self.bench_table = QTableWidget(len(SOLVER_NAMES), 7)
        self.bench_table.setHorizontalHeaderLabels([
            "Solver", "Backend", "E_best (mean)", "Approx ratio r",
            "r 95% CI", "Raw feas %", "Wall (ms)",
        ])
        self.bench_table.horizontalHeader().setStretchLastSection(True)
        self.bench_table.verticalHeader().setVisible(False)
        for row, name in enumerate(SOLVER_NAMES):
            self.bench_table.setItem(row, 0, QTableWidgetItem(name))

        # --- approximation-ratio bar chart -------------------------------------
        self.bench_bars = pg.PlotWidget(
            title="Approximation ratio r = E*/E (mean, bootstrap 95% CI)"
        )
        self.bench_bars.setLabel("left", "r (1.0 = proven optimum)")
        self.bench_bars.showGrid(y=True, alpha=0.3)
        axis = self.bench_bars.getAxis("bottom")
        axis.setTicks([[(i, name) for i, name in enumerate(SOLVER_NAMES)]])
        self._bench_bar_item: pg.BarGraphItem | None = None
        self._bench_err_item: pg.ErrorBarItem | None = None

        # --- convergence plots ---------------------------------------------------
        self.pasqal_plot = pg.PlotWidget(
            title="Pasqal analog-QAOA — best repaired cost vs BO evaluation"
        )
        self.pasqal_plot.setLabel("bottom", "BO evaluation")
        self.pasqal_plot.setLabel("left", "best combined cost")
        self.pasqal_plot.showGrid(x=True, y=True, alpha=0.3)
        self.pasqal_curve = self.pasqal_plot.plot(
            [], [], pen=pg.mkPen("m", width=2), symbol="o", symbolSize=6
        )
        self.pasqal_opt_line = pg.InfiniteLine(
            angle=0, pen=pg.mkPen("g", style=Qt.PenStyle.DashLine)
        )
        self.pasqal_plot.addItem(self.pasqal_opt_line)

        self.xanadu_plot = pg.PlotWidget(
            title="Xanadu GBS — best repaired cost vs samples"
        )
        self.xanadu_plot.setLabel("bottom", "samples drawn")
        self.xanadu_plot.setLabel("left", "best combined cost")
        self.xanadu_plot.showGrid(x=True, y=True, alpha=0.3)
        self.xanadu_curve = self.xanadu_plot.plot(
            [], [], pen=pg.mkPen("y", width=2), symbol="o", symbolSize=6
        )
        self.xanadu_opt_line = pg.InfiniteLine(
            angle=0, pen=pg.mkPen("g", style=Qt.PenStyle.DashLine)
        )
        self.xanadu_plot.addItem(self.xanadu_opt_line)

        self._bench_curves: dict[str, tuple[list[int], list[float]]] = {
            "pasqal-analog": ([], []),
            "xanadu-gbs": ([], []),
        }

        # --- layout ---------------------------------------------------------
        mid = QHBoxLayout()
        mid.addWidget(self.bench_table, stretch=3)
        mid.addWidget(self.bench_bars, stretch=2)
        mid_w = QWidget()
        mid_w.setLayout(mid)

        bottom = QHBoxLayout()
        bottom.addWidget(self.pasqal_plot)
        bottom.addWidget(self.xanadu_plot)
        bottom_w = QWidget()
        bottom_w.setLayout(bottom)

        root = QVBoxLayout()
        root.addLayout(controls)
        root.addWidget(mid_w, stretch=2)
        root.addWidget(bottom_w, stretch=3)
        wrap = QWidget()
        wrap.setLayout(root)
        return wrap

    def _build_map_tab(self) -> QWidget:
        # A presentation-grade geographic panel: a real MapLibre basemap (drawn
        # by Plotly) with the predicted ISSR risk as a density overlay and the
        # solved routes on top, shown inside an embedded Chromium view. The view
        # is backed by a temp HTML file that _render_map rewrites on each solve.
        fd, self._map_html_path = tempfile.mkstemp(prefix="contrail_map_", suffix=".html")
        os.close(fd)

        # QtWebEngine can't initialise under the offscreen Qt platform (no GPU /
        # display) and would crash the process, so headless runs (CI, tests) get
        # a placeholder while _render_map still writes the HTML. A real desktop
        # session gets the live map.
        headless = os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        if not _HAS_WEBENGINE or headless:
            self.map_view = None
            why = (
                "headless session (offscreen) — open the GUI on a desktop to see the map"
                if _HAS_WEBENGINE
                else 'Map needs the web view: pip install -e ".[gui]" (plotly + PyQt6-WebEngine)'
            )
            msg = QLabel(why)
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return msg

        self.map_view = QWebEngineView()
        self.map_view.setHtml(
            "<body style='margin:0;background:#0e0e10;color:#888;"
            "font-family:sans-serif'>"
            "<p style='padding:1rem'>Build a scenario to render the map…</p></body>"
        )
        return self.map_view

    def _active_anchor(self, field):
        """The GeoAnchor placing this ISSR field on the map.

        The MLIssrField carries its own anchor; the synthetic field has none,
        so we place it with the SAME canonical anchor the model/features use,
        so synthetic and ml render over the same European box.
        """
        anchor = getattr(field, "anchor", None)
        if anchor is not None:
            return anchor
        from contrail_ml.config import DEFAULT_CONFIG

        return DEFAULT_CONFIG.anchor

    def _map_routes(self, chosen_by_flight: dict[str, int] | None, anchor) -> list[RouteLine]:
        """Per-flight ground tracks in lon/lat (chosen profile, else baseline)."""
        routes: list[RouteLine] = []
        for flight in self._flights:
            profile = flight.baseline
            is_chosen = False
            if chosen_by_flight and flight.name in chosen_by_flight:
                members = self._groups.get(flight.name, [])
                k = chosen_by_flight[flight.name]
                if 0 <= k < len(members):
                    profile = self._evals[members[k]].profile
                    is_chosen = True
            wps = waypoints_for(flight, profile)
            wx = np.array([w[0] for w in wps], dtype=float)
            wy = np.array([w[1] for w in wps], dtype=float)
            lon, lat = anchor.local_to_geo(wx, wy)
            routes.append(
                RouteLine(
                    name=flight.name,
                    lon=np.asarray(lon, dtype=float),
                    lat=np.asarray(lat, dtype=float),
                    chosen=is_chosen,
                )
            )
        return routes

    def _render_map(self, chosen_by_flight: dict[str, int] | None) -> None:
        """Render the ISSR-risk overlay + flight routes onto the geographic map.

        Works for BOTH the synthetic ISSRField and the MLIssrField, since they
        share the rhi_excess_grid interface. On a 2-D map the altitude dimension
        collapses, so each flight is one ground track; once solved, chosen routes
        are drawn green over the muted context. Builds a Plotly figure and loads
        its self-contained HTML into the embedded web view.
        """
        if self._world is None:
            return
        field = self._world.issr
        anchor = self._active_anchor(field)
        g = self._world.grid

        # Sample the risk field on a local (x_km, y_km) slice at a fixed altitude,
        # then map every sample point to lon/lat (exact transform, not a bbox).
        xs = np.linspace(g.x_min_km, g.x_max_km, _MAP_GRID_NX)
        ys = np.linspace(g.y_min_km, g.y_max_km, _MAP_GRID_NY)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        zz = np.full_like(xx, fl_to_m(_MAP_SLICE_FL))
        risk = np.asarray(field.rhi_excess_grid(xx, yy, zz), dtype=float)
        lons, lats = anchor.local_to_geo(xx, yy)

        routes = self._map_routes(chosen_by_flight, anchor)
        source = getattr(field, "source", "synthetic")
        fig = build_map_figure(source=source, lon=lons, lat=lats, risk=risk, routes=routes)

        # Write a self-contained page (plotly.js inlined) and load it from disk —
        # setHtml caps payloads at ~2 MB, which the inlined library exceeds. A
        # cache-busting query forces the view to re-read after each solve.
        html = fig.to_html(include_plotlyjs="inline", full_html=True,
                           config={"displayModeBar": True})
        with open(self._map_html_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        if self.map_view is not None:
            url = QUrl.fromLocalFile(self._map_html_path)
            url.setQuery(f"v={time.time()}")
            self.map_view.load(url)

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
            world, flights, evals, conflicts, buckets = build_scenario_full(cfg)
            qubo = assemble_qubo(evals, conflicts, buckets)
        except Exception as exc:
            self.summary.setText(f"instance build failed:\n{exc}")
            return

        self._world = world
        self._flights = flights
        self._evals = evals
        self._conflicts = conflicts
        groups: OrderedDict[str, list[int]] = OrderedDict()
        for idx, ev in enumerate(evals):
            groups.setdefault(ev.flight_name, []).append(idx)
        self._groups = groups

        self._render_conflict_graph(set())
        self._render_qubo(qubo, len(conflicts), len(buckets))
        self._render_map(None)
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

        # Draw the chosen routes (green) over the risk map.
        self._render_map({c.flight_name: c.chosen_option for c in choices})

        self.solve_btn.setEnabled(True)

    def _on_failed(self, message: str) -> None:
        self.summary.setText(f"solve failed:\n{message}")
        self.solve_btn.setEnabled(True)

    # ----------------------------------------------------------- benchmark ---
    def _on_run_benchmark(self) -> None:
        if self._bench_worker is not None and self._bench_worker.isRunning():
            return

        base_seed = self.seed_spin.value()
        seeds = [base_seed + k for k in range(self.bench_seeds_spin.value())]

        for xs, ys in self._bench_curves.values():
            xs.clear()
            ys.clear()
        self.pasqal_curve.setData([], [])
        self.xanadu_curve.setData([], [])
        for row in range(self.bench_table.rowCount()):
            for col in range(1, self.bench_table.columnCount()):
                self.bench_table.setItem(row, col, QTableWidgetItem("…"))

        # Progress bar: 100 units per (seed, solver) cell.
        self.bench_progress.setRange(0, len(seeds) * len(SOLVER_NAMES) * 100)
        self.bench_progress.setValue(0)
        self._bench_cells_done = 0
        self._bench_t0 = time.monotonic()

        # Honest expectations up front: the statevector cost is 2^n.
        n_q = len(self._evals)
        if n_q > MAX_STATEVECTOR_QUBITS:
            hint = f" — {n_q} qubits > {MAX_STATEVECTOR_QUBITS}-qubit cap: Pasqal row will be SKIPPED"
        elif n_q >= 17:
            hint = f" — {n_q} qubits: each Pasqal BO eval integrates 2^{n_q} amplitudes (minutes!)"
        else:
            hint = ""
        self._bench_msg = f"running seeds {seeds}{hint}"
        self._refresh_bench_status()
        self._bench_timer.start()
        self.bench_btn.setEnabled(False)

        self._bench_worker = BenchmarkWorker(
            cfg=self._build_cfg(),
            seeds=seeds,
            n_shots=self.bench_shots_spin.value(),
            bo_iters=self.bench_bo_spin.value(),
        )
        self._bench_worker.progress.connect(self._on_bench_progress)
        self._bench_worker.cell_done.connect(self._on_bench_cell)
        self._bench_worker.phase.connect(self._on_bench_phase)
        self._bench_worker.finished_ok.connect(self._on_bench_done)
        self._bench_worker.failed.connect(self._on_bench_failed)
        self._bench_worker.start()

    def _refresh_bench_status(self) -> None:
        elapsed = int(time.monotonic() - self._bench_t0)
        self.bench_status.setText(
            f"{self._bench_msg}   [{elapsed // 60}:{elapsed % 60:02d} elapsed]"
        )

    def _on_bench_phase(self, message: str, frac: float) -> None:
        self._bench_msg = message
        self.bench_progress.setValue(
            self._bench_cells_done * 100 + int(100 * min(1.0, max(0.0, frac)))
        )
        self._refresh_bench_status()

    def _on_bench_progress(self, solver: str, step: int, cost: float) -> None:
        if solver not in self._bench_curves:
            return
        xs, ys = self._bench_curves[solver]
        xs.append(step)
        ys.append(cost)
        curve = self.pasqal_curve if solver == "pasqal-analog" else self.xanadu_curve
        curve.setData(xs, ys)
        self._bench_msg = f"{solver}  step {step}  best {cost:.1f}"
        self._refresh_bench_status()

    def _on_bench_cell(self, seed: int, cell: object) -> None:
        self._bench_cells_done += 1
        self.bench_progress.setValue(self._bench_cells_done * 100)
        # CP-SAT finishing marks the start of a new instance: reset the
        # quantum convergence curves and pin the optimum reference lines.
        if cell.solver == "cpsat":  # type: ignore[attr-defined]
            for xs, ys in self._bench_curves.values():
                xs.clear()
                ys.clear()
            self.pasqal_curve.setData([], [])
            self.xanadu_curve.setData([], [])
            optimum = cell.best_cost  # type: ignore[attr-defined]
            self.pasqal_opt_line.setValue(optimum)
            self.xanadu_opt_line.setValue(optimum)
            self._bench_msg = f"seed {seed}:  E* = {optimum:.1f}  (CP-SAT)"
            self._refresh_bench_status()

    def _on_bench_done(self, report: object) -> None:
        stats = report.aggregate()  # type: ignore[attr-defined]
        instances = report.instances  # type: ignore[attr-defined]

        # Backend / note per solver from the most recent instance.
        latest: dict[str, object] = {}
        for inst in instances:
            for run in inst.runs:
                latest[run.solver] = run

        means: list[float] = []
        err_low: list[float] = []
        err_high: list[float] = []
        for row, name in enumerate(SOLVER_NAMES):
            s = stats.get(name)
            run = latest.get(name)
            backend = getattr(run, "backend", "-")
            if s is not None:
                values = [
                    backend,
                    _mean_best_cost(instances, name),
                    f"{s.ratio_mean:.4f}",
                    f"[{s.ratio_ci_low:.4f}, {s.ratio_ci_high:.4f}]",
                    f"{100 * s.feasibility_mean:.1f}",
                    f"{1000 * s.wall_clock_mean_s:.0f}",
                ]
                means.append(s.ratio_mean)
                err_low.append(s.ratio_mean - s.ratio_ci_low)
                err_high.append(s.ratio_ci_high - s.ratio_mean)
            else:
                note = getattr(run, "note", "no data")
                status = getattr(run, "status", "—")
                values = [backend, "—", "—", "—", "—", f"{status}: {note}"]
                means.append(0.0)
                err_low.append(0.0)
                err_high.append(0.0)
            for col, val in enumerate(values, start=1):
                self.bench_table.setItem(row, col, QTableWidgetItem(str(val)))

        # Ratio bars with bootstrap CIs.
        if self._bench_bar_item is not None:
            self.bench_bars.removeItem(self._bench_bar_item)
        if self._bench_err_item is not None:
            self.bench_bars.removeItem(self._bench_err_item)
        x = np.arange(len(SOLVER_NAMES))
        self._bench_bar_item = pg.BarGraphItem(
            x=x, height=means, width=0.55, brush=(80, 160, 220, 180)
        )
        self._bench_err_item = pg.ErrorBarItem(
            x=x, y=np.array(means),
            top=np.array(err_high), bottom=np.array(err_low),
            beam=0.18, pen=pg.mkPen("w", width=2),
        )
        self.bench_bars.addItem(self._bench_bar_item)
        self.bench_bars.addItem(self._bench_err_item)
        self.bench_bars.setYRange(min(0.95, min(m for m in means if m > 0) - 0.02), 1.01)

        self._bench_timer.stop()
        self.bench_progress.setValue(self.bench_progress.maximum())
        elapsed = int(time.monotonic() - self._bench_t0)
        n_inst = len(instances)
        self.bench_status.setText(
            f"done in {elapsed // 60}:{elapsed % 60:02d} — {n_inst} seed(s) × "
            f"{len(SOLVER_NAMES)} solvers; green dashed line = CP-SAT optimum E* "
            f"of the last seed"
        )
        self.bench_btn.setEnabled(True)

    def _on_bench_failed(self, message: str) -> None:
        self._bench_timer.stop()
        self.bench_status.setText(f"benchmark failed: {message}")
        self.bench_btn.setEnabled(True)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1280, 760)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
