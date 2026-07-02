"""
app.py — Researcher-focused analytical dashboard for the contrail QUBO pipeline.

Tabs 1-5 analyze the *problem* and the *solvers*: the live CP-SAT convergence
stream, the conflict-graph topology, the assembled QUBO matrix, the cost
trade-offs of the chosen options, and the head-to-head quantum benchmark (CP-SAT
vs Pasqal analog vs Xanadu GBS). Tab 6 is the one geographic view: the predicted
ISSR risk over Europe with the solved routes drawn on a real (offline) basemap.

DATA FLOW
=========
* Everything runs IN-PROCESS — no server, no broker. The CP-SAT solve runs off
  the UI thread in SolveWorker (a QThread); its progress callback fires the Qt
  `progress` signal so the convergence curve updates live without blocking.
* The dashboard builds the QUBO and conflict graph from the same (seeded)
  ScenarioConfig via contrail_env.scenario.build_scenario_full +
  contrail_env.assemble_qubo, so every panel shows exactly what was solved.
* The quantum benchmark (tab 5) likewise runs in a worker thread: the samplers
  live in contrail_env (pasqal_analog, xanadu_gbs) and the protocol in
  contrail_env.benchmark.

PANELS
======
1. Live CP-SAT convergence  — objective vs improvement index (live, in-process).
2. Conflict-graph topology   — option nodes grouped by flight, conflict edges.
3. QUBO & matrix analytics   — size, sparsity, penalty constants, energy offset,
                               and a heatmap of |Q|.
4. Trade-off results table   — chosen option per flight with fuel / contrail /
                               disruption, to read off the alpha,beta,gamma trade.
5. Quantum benchmark         — plan §10 protocol over N seeds: approximation
                               ratios with bootstrap CIs, raw feasibility rates,
                               wall clocks, and live convergence curves for the
                               Pasqal BO loop and the Xanadu GBS sampler.
6. Geographic map            — Plotly `geo` basemap (real country borders /
                               coastlines, SVG so no WebGL, offline vectors) with
                               the ISSR risk as an Inferno marker overlay and the
                               chosen vs context routes; embedded QtWebEngine view
                               (needs the [gui] extra).
"""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401  (registers pg.exporters.ImageExporter)
from PyQt6.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
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

from contrail_env import World, assemble_qubo, fl_to_m, waypoints_for
from contrail_env.benchmark import SOLVER_NAMES, BenchmarkReport, run_benchmark
from contrail_env.dynamics import DynamicsRecord, ResidualEnergyResult
from contrail_env.geo import EUROPEAN_ANCHOR
from contrail_env.pasqal_analog import MAX_STATEVECTOR_QUBITS, OMEGA_MAX_HW, T_MAX_NS
from contrail_env.scenario import ScenarioConfig, build_scenario_full, solve_scenario
from contrail_env.spectral import AnalogSpectrum

# One pen colour per solver, used consistently across every tab so a solver is
# recognisable by colour alone (benchmark bars, convergence curves, fingerprint
# bars). Keys are the SOLVER_NAMES of contrail_env.benchmark.
SOLVER_PENS: dict[str, str] = {
    "cpsat": "g",
    "random-repair": "w",
    "sa": "c",
    "pasqal-analog": "m",
    "xanadu-gbs": "y",
}

# Pens for the entropy-cut curves (Dynamics tab) — order-stable, one per cut.
_CUT_PENS = ["c", "y", "m", "w"]

# §0.5 honesty sentence — entropy and the spectral gap locate the sweep's
# critical window; they are not performance validation. Verbatim per spec.
_DYN_HONESTY = (
    "entanglement entropy and the spectral gap are dynamics diagnostics that "
    "locate the critical window of the analog sweep; they are not evidence of "
    "computational usefulness."
)


def _export_rows_csv(parent: QWidget, header: list[str], rows: list[list[object]]) -> None:
    """Prompt for a path and write `header` + `rows` as CSV (the raw plot data)."""
    import csv

    path, _ = QFileDialog.getSaveFileName(parent, "Export CSV", "", "CSV files (*.csv)")
    if not path:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _export_plot_png(parent: QWidget, plot: pg.PlotWidget | pg.GraphicsLayoutWidget) -> None:
    """Prompt for a path and export a pyqtgraph plot/scene as a PNG image."""
    path, _ = QFileDialog.getSaveFileName(parent, "Export PNG", "", "PNG image (*.png)")
    if not path:
        return
    item = plot.plotItem if isinstance(plot, pg.PlotWidget) else plot.ci
    exporter = pg.exporters.ImageExporter(item)
    exporter.export(path)


def _fmt_tts_s(tts_s: float) -> str:
    """Render a TTS for a table cell: '∞' for infinite, '—' for undefined."""
    if math.isnan(tts_s):
        return "—"
    if math.isinf(tts_s):
        return "∞"
    return f"{tts_s:.3g}s"


def _stretch_last(table: QTableWidget, *, hide_vertical: bool = False) -> None:
    """Stretch a table's last column (the Qt stub types the header as optional)."""
    header = table.horizontalHeader()
    if header is not None:
        header.setStretchLastSection(True)
    if hide_vertical:
        v_header = table.verticalHeader()
        if v_header is not None:
            v_header.setVisible(False)

# Fixed scenario knobs not exposed as controls (sensible defaults per the brief).
CORRIDOR_FRAC = 0.05
SNAPSHOT_WINDOW_S = 300.0
# Altitude (FL) of the horizontal slice the map heatmap shows. A 2-D map can
# only show one altitude of the 3-D risk field; FL360 is the middle of the band.
_MAP_SLICE_FL = 360
# The map uses Plotly's `geo` subplot: real country borders + coastlines drawn
# with SVG (D3), NOT WebGL. This renders everywhere — including locked-down /
# headless Chromium where WebGL (MapLibre/Mapbox tile maps) is unavailable — and
# needs no map tiles or internet (Natural Earth vectors ship with plotly.js).
# Risk grid resolution (lon x lat samples) for the marker-cloud overlay, the
# marker size (px), and the fraction-of-max below which a cell is "clear sky"
# and skipped so the basemap stays visible.
_MAP_GRID_NX, _MAP_GRID_NY = 100, 68
_MAP_RISK_FLOOR_FRAC = 0.04
# Marker size for the small inset risk cloud, and the flight-level axis range
# for the main altitude-profile panel.
_MAP_INSET_MARKER = 5
_FL_AXIS_LO, _FL_AXIS_HI = 330, 410
# Per-flight colours: a flight's filed (dotted) and chosen (solid) profiles
# share its colour, so colour = which flight, line style = filed vs chosen.
_FLIGHT_COLORS = [
    "#4ea1ff", "#ffb648", "#28dc5a", "#ff6ec7",
    "#b388ff", "#5ad8e6", "#ff7a5c", "#c0e84a",
]
# Bundled Natural Earth vectors for the geo basemap (see _geo_assets_script).
_GEO_TOPOJSON_NAME = "world_50m"
_GEO_ASSET_PATH = Path(__file__).resolve().parent / "assets" / f"{_GEO_TOPOJSON_NAME}.json"
_GEO_ASSET_CACHE: str | None = None

_MONO = QFont("Consolas", 10)


def _geo_assets_script() -> str:
    """Inline the basemap topojson so the geo map needs no CDN / network.

    Plotly's geo subplot otherwise fetches its country/coastline vectors from
    cdn.plot.ly at render time. Injecting them into window.PlotlyGeoAssets (which
    plotly.js reads on load) makes the map fully offline — important on locked-
    down networks. Returns "" if the bundled asset is absent, leaving plotly to
    fall back to the CDN.
    """
    global _GEO_ASSET_CACHE
    if _GEO_ASSET_CACHE is None:
        try:
            _GEO_ASSET_CACHE = _GEO_ASSET_PATH.read_text(encoding="utf-8")
        except OSError:
            _GEO_ASSET_CACHE = ""
    if not _GEO_ASSET_CACHE:
        return ""
    return (
        "<script>window.PlotlyGeoAssets=window.PlotlyGeoAssets||{topojson:{}};"
        f'window.PlotlyGeoAssets.topojson["{_GEO_TOPOJSON_NAME}"]={_GEO_ASSET_CACHE};</script>'
    )


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
    """One flight's ground track in geographic coordinates (the inset map)."""

    name: str
    lon: np.ndarray
    lat: np.ndarray
    chosen: bool


@dataclass(frozen=True)
class ProfileSeries:
    """One flight's altitude profile along its track: flight level vs distance.

    `chosen` marks the solver's pick (solid line); otherwise it's the filed
    baseline (dotted). Both of a flight's lines share `color`, so colour says
    *which flight* and line style says *filed vs chosen*. `issr_mask[i]` is True
    where the aircraft — at that point AND that altitude — sits inside a contrail
    region; those points get a red dot, so you see exactly where a route plows
    through (or climbs over) the ISSR.
    """

    name: str
    s_km: np.ndarray
    fl: np.ndarray
    issr_mask: np.ndarray
    color: str
    chosen: bool


def _profile_trace(go, ps: ProfileSeries):
    """A flight-level-vs-distance line with red dots at ISSR crossings."""
    issr = np.asarray(ps.issr_mask, dtype=bool)
    return go.Scatter(
        x=np.asarray(ps.s_km, dtype=float),
        y=np.asarray(ps.fl, dtype=float),
        mode="lines+markers",
        line=dict(
            width=3 if ps.chosen else 1.6,
            color=ps.color,
            dash="solid" if ps.chosen else "dot",
        ),
        marker=dict(size=np.where(issr, 7.0, 0.0), color="#ff4d4d", line=dict(width=0)),
        name=f"{ps.name} · chosen" if ps.chosen else f"{ps.name} · filed",
        hovertemplate=f"{ps.name}<br>FL%{{y:.0f}} @ %{{x:.0f}} km<extra></extra>",
    )


def _route_trace(go, r: RouteLine):
    """A static ground track for the inset map (green if chosen, else muted)."""
    return go.Scattergeo(
        lon=np.asarray(r.lon, dtype=float),
        lat=np.asarray(r.lat, dtype=float),
        mode="lines",
        line=dict(width=2.5 if r.chosen else 1.0,
                  color="#28dc5a" if r.chosen else "rgba(170,172,184,0.5)"),
        showlegend=False,
        hovertemplate=f"{r.name}<extra></extra>",
    )


def build_map_figure(
    *,
    source: str,
    profiles: list[ProfileSeries],
    lon: np.ndarray,
    lat: np.ndarray,
    risk: np.ndarray,
    routes: list[RouteLine],
):
    """Build the Map tab figure: a big altitude-profile panel + a small inset map.

    The optimiser only changes ALTITUDE, so a top-down map can't tell the filed
    route from the chosen one — their ground tracks are identical. The main panel
    therefore plots flight level vs along-track distance: the filed baseline
    (dotted) against the chosen profile (solid), with a red dot wherever the
    aircraft, at that altitude, is inside a contrail region. The familiar
    geographic ISSR map rides along as a small inset (top-right) for context.

    Pure function of plain arrays, so it unit-tests directly. Drawn with Plotly:
    the profiles on xy axes, the inset on a `geo` subplot (SVG Natural Earth
    vectors — renders with no WebGL / CDN).
    """
    import plotly.graph_objects as go  # lazy: keep module import lean

    fig = go.Figure()

    # --- main panel: altitude profiles (xy axes) ---
    for ps in profiles:
        fig.add_trace(_profile_trace(go, ps))

    # --- inset map: ISSR risk cloud + ground tracks (geo subplot, top-right) ---
    lon_f = np.asarray(lon, dtype=float).ravel()
    lat_f = np.asarray(lat, dtype=float).ravel()
    risk_f = np.asarray(risk, dtype=float).ravel()
    rmax = float(np.nanmax(risk_f)) if risk_f.size else 0.0
    keep = risk_f > (_MAP_RISK_FLOOR_FRAC * rmax) if rmax > 0 else np.zeros_like(risk_f, bool)
    kr = risk_f[keep]
    sizes = _MAP_INSET_MARKER * (0.5 + 1.0 * (kr / rmax)) if rmax > 0 else _MAP_INSET_MARKER
    fig.add_trace(
        go.Scattergeo(
            lon=lon_f[keep], lat=lat_f[keep], mode="markers",
            marker=dict(size=sizes, color=kr, colorscale="Inferno",
                        cmin=0.0, cmax=rmax if rmax > 0 else 1.0,
                        opacity=0.6, line=dict(width=0), showscale=False),
            name="ISSR risk", showlegend=False, hoverinfo="skip",
        )
    )
    for r in routes:
        fig.add_trace(_route_trace(go, r))

    # Frame the inset on the data box.
    margin = 1.0
    lon_lo = float(np.nanmin(lon_f)) - margin if lon_f.size else -6.0
    lon_hi = float(np.nanmax(lon_f)) + margin if lon_f.size else 16.0
    lat_lo = float(np.nanmin(lat_f)) - margin if lat_f.size else 42.0
    lat_hi = float(np.nanmax(lat_f)) + margin if lat_f.size else 51.0
    fig.update_geos(
        domain=dict(x=[0.66, 0.995], y=[0.58, 0.99]),   # top-right inset
        projection_type="mercator", resolution=50,
        lonaxis_range=[lon_lo, lon_hi], lataxis_range=[lat_lo, lat_hi],
        showland=True, landcolor="#1a1b20",
        showocean=True, oceancolor="#0b0c10",
        showcountries=True, countrycolor="rgba(255,255,255,0.22)",
        showcoastlines=True, coastlinecolor="rgba(255,255,255,0.3)",
        showlakes=False, bgcolor="#0e0e10",
    )
    fig.update_layout(
        margin=dict(l=64, r=12, t=42, b=48),
        title=dict(text=f"Altitude profiles — filed (dotted) vs chosen (solid)   ·   "
                        f"inset: ISSR map ({source})", x=0.5),
        xaxis=dict(title="along-track distance (km)", gridcolor="rgba(255,255,255,0.08)",
                   zeroline=False, color="#cfcfd6"),
        yaxis=dict(title="flight level (FL)", range=[_FL_AXIS_LO, _FL_AXIS_HI],
                   gridcolor="rgba(255,255,255,0.08)", zeroline=False, color="#cfcfd6"),
        paper_bgcolor="#0e0e10", plot_bgcolor="#0e0e10",
        font=dict(color="#dddde2"),
        legend=dict(bgcolor="rgba(20,20,24,0.6)", x=0.01, y=0.99, font=dict(size=10)),
    )
    return fig


# =============================================================================
# WORKER THREAD  (unchanged contract: streams progress, returns SolveResponse)
# =============================================================================

class SolveWorker(QThread):
    """Runs one CP-SAT solve off the UI thread, streaming progress via signals.

    The solve runs in-process (contrail_env.scenario.solve_scenario); the
    progress callback fires on this worker thread and the Qt signal hands each
    incumbent to the UI thread, so the dashboard stays responsive with no
    network service involved.
    """

    progress = pyqtSignal(int, float)
    finished_ok = pyqtSignal(object)   # contrail_env.scenario.SolveResult
    failed = pyqtSignal(str)

    def __init__(self, cfg: ScenarioConfig) -> None:
        super().__init__()
        self._cfg = cfg

    def run(self) -> None:
        try:
            result = solve_scenario(
                self._cfg,
                on_progress=lambda improvement, objective: self.progress.emit(
                    improvement, objective
                ),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished_ok.emit(result)


class _AnalysisWorker(QThread):
    """Runs a (possibly slow) analysis off the UI thread.

    `fn(emit)` receives a callback to stream intermediate results; whatever it
    returns is delivered via `done`. Used by the research tabs (landscape /
    quantum convergence / hardness sweep) so the dashboard never blocks.
    """

    done = pyqtSignal(object)
    progress = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn(self.progress.emit)
        except Exception as exc:  # noqa: BLE001 - surfaced to the status label
            self.failed.emit(str(exc))
        else:
            self.done.emit(result)


# =============================================================================
# BENCHMARK WORKER — runs the plan §10 protocol locally, off the UI thread
# =============================================================================

class BenchmarkWorker(QThread):
    """One full benchmark sweep: CP-SAT vs Pasqal vs Xanadu over N seeds.

    The scenario factory reuses the SAME seeded construction as the rest of
    the dashboard (contrail_env.scenario.build_scenario_full), so the instances
    benchmarked here are exactly the ones the other tabs display.
    """

    progress = pyqtSignal(str, int, float)   # solver, step, best cost so far
    cell_done = pyqtSignal(int, object)      # seed, contrail_env.benchmark.SolverRun
    phase = pyqtSignal(str, float)           # heartbeat: message, cell fraction [0,1]
    finished_ok = pyqtSignal(object)         # contrail_env.benchmark.BenchmarkReport
    failed = pyqtSignal(str)

    def __init__(
        self,
        cfg: ScenarioConfig,
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
            cfg = replace(self._cfg, seed=seed)
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
        self._pending_cfg: ScenarioConfig | None = None
        # Last completed benchmark report — read by the exports and by the
        # constraint-fingerprint tab (G3).
        self._bench_report: BenchmarkReport | None = None

        # Research-tab workers (landscape / quantum convergence / hardness sweep).
        self._land_worker: _AnalysisWorker | None = None
        self._qconv_worker: _AnalysisWorker | None = None
        self._hard_worker: _AnalysisWorker | None = None
        self._qconv_data: dict[str, tuple[list, list]] = {"pasqal": ([], []), "gbs": ([], [])}

        # Quantum-dynamics tab workers + last results (G2).
        self._dyn_worker: _AnalysisWorker | None = None
        self._dyn_residual_worker: _AnalysisWorker | None = None
        self._dyn_spectrum: AnalogSpectrum | None = None
        self._dyn_record: DynamicsRecord | None = None
        self._dyn_residual: ResidualEnergyResult | None = None

        # Benchmark progress state (see _on_bench_phase / _refresh_bench_status).
        self._bench_t0 = 0.0
        self._bench_cells_done = 0
        self._bench_msg = ""

        # Reconstructed problem state (set by _rebuild_structure).
        self._evals: list = []
        self._conflicts: list = []
        self._buckets: list = []
        self._groups: OrderedDict[str, list[int]] = OrderedDict()
        self._graph_texts: list[pg.TextItem] = []
        # World + flights kept for the geographic map panel.
        self._world: World | None = None
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
        self.tabs.addTab(self._build_landscape_tab(), "1 · Energy landscape")
        self.tabs.addTab(self._build_quantum_conv_tab(), "2 · Quantum convergence")
        self.tabs.addTab(self._build_hardness_tab(), "3 · Hardness sweep")
        self.tabs.addTab(self._build_graph_tab(), "4 · Conflict graph")
        self.tabs.addTab(self._build_qubo_tab(), "5 · QUBO analytics")
        self.tabs.addTab(self._build_results_tab(), "6 · Trade-offs")
        self.tabs.addTab(self._build_benchmark_tab(), "7 · Quantum benchmark")
        self.tabs.addTab(self._build_map_tab(), "8 · Map")
        self.tabs.addTab(self._build_dynamics_tab(), "9 · Quantum dynamics")

        root = QHBoxLayout()
        root.addWidget(left_widget)
        root.addWidget(self.tabs, stretch=1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    # ----------------------------------------------------- tab 1: landscape ---
    def _build_landscape_tab(self) -> QWidget:
        self.land_btn = QPushButton("Compute landscape (+ GBS / random samples)")
        self.land_btn.clicked.connect(self._on_landscape)
        self.land_status = QLabel(
            "idle — enumerates every feasible solution's cost and overlays where "
            "the GBS sampler vs random actually land. Click after building an instance."
        )
        self.land_status.setFont(_MONO)

        self.land_plot = pg.PlotWidget(
            title="Solution-cost landscape — where the optimum sits, and where samplers land"
        )
        self.land_plot.setLabel("bottom", "combined cost (lower = better)")
        self.land_plot.setLabel("left", "frequency")
        self.land_plot.showGrid(x=True, y=True, alpha=0.3)
        self.land_plot.addLegend()
        self.land_all = self.land_plot.plot([], [], pen=pg.mkPen((90, 140, 220)), name="all feasible")
        self.land_rnd = self.land_plot.plot([], [], pen=pg.mkPen((180, 180, 180)), name="random")
        self.land_gbs = self.land_plot.plot([], [], pen=pg.mkPen((40, 220, 120)), name="GBS")
        self.land_opt = pg.InfiniteLine(angle=90, pen=pg.mkPen("g", width=2, style=Qt.PenStyle.DashLine))
        self.land_plot.addItem(self.land_opt)

        controls = QHBoxLayout()
        controls.addWidget(self.land_btn)
        controls.addWidget(self.land_status, stretch=1)
        lay = QVBoxLayout()
        lay.addLayout(controls)
        lay.addWidget(self.land_plot, stretch=1)
        wrap = QWidget()
        wrap.setLayout(lay)
        return wrap

    def _on_landscape(self) -> None:
        if self._land_worker is not None and self._land_worker.isRunning():
            return
        if not self._evals:
            self.land_status.setText("build an instance first (Randomize)")
            return
        evals, conflicts, buckets = self._evals, self._conflicts, self._buckets
        seed = self.seed_spin.value()

        def job(_emit):
            from contrail_env.analysis import (
                feasible_cost_landscape,
                gbs_sample_costs,
                random_sample_costs,
            )
            costs, optimum = feasible_cost_landscape(evals, conflicts, buckets)
            gbs = gbs_sample_costs(evals, conflicts, buckets, n_samples=500, seed=seed)
            rnd = random_sample_costs(evals, conflicts, buckets, n_samples=500, seed=seed)
            return costs, gbs, rnd, optimum

        self.land_btn.setEnabled(False)
        self.land_status.setText("computing…")
        self._land_worker = _AnalysisWorker(job)
        self._land_worker.done.connect(self._on_landscape_done)
        self._land_worker.failed.connect(self._on_landscape_failed)
        self._land_worker.start()

    def _on_landscape_done(self, result) -> None:
        costs, gbs, rnd, optimum = result
        allv = np.concatenate([costs, gbs, rnd])
        lo, hi = float(allv.min()), float(allv.max())
        edges = np.linspace(lo, hi if hi > lo else lo + 1.0, 31)

        def hist(curve, vals, brush):
            y, _ = np.histogram(np.asarray(vals, dtype=float), bins=edges)
            curve.setData(edges, y, stepMode="center", fillLevel=0, brush=brush)

        hist(self.land_all, costs, (90, 140, 220, 110))
        hist(self.land_rnd, rnd, (180, 180, 180, 110))
        hist(self.land_gbs, gbs, (40, 220, 120, 130))
        self.land_opt.setValue(optimum)
        self.land_status.setText(
            f"{costs.size} feasible solutions · optimum = {optimum:.1f} · "
            f"GBS best = {float(gbs.min()):.1f} · random best = {float(rnd.min()):.1f}  "
            f"(does GBS pile up nearer the optimum than random?)"
        )
        self.land_btn.setEnabled(True)

    def _on_landscape_failed(self, msg: str) -> None:
        self.land_status.setText(f"failed: {msg}")
        self.land_btn.setEnabled(True)

    # ------------------------------------------ tab 2: quantum convergence ---
    def _build_quantum_conv_tab(self) -> QWidget:
        self.qconv_btn = QPushButton("Run quantum solvers on current instance")
        self.qconv_btn.clicked.connect(self._on_quantum_conv)
        self.qconv_status = QLabel(
            "idle — runs Pasqal QAOA + Xanadu GBS on the current instance and tracks "
            "best-cost-so-far vs work (a convergence curve that actually moves)."
        )
        self.qconv_status.setFont(_MONO)

        self.qconv_plot = pg.PlotWidget(
            title="Quantum convergence — best repaired cost vs work (lower = better)"
        )
        self.qconv_plot.setLabel("bottom", "progress (BO eval / sample batch)")
        self.qconv_plot.setLabel("left", "best combined cost so far")
        self.qconv_plot.showGrid(x=True, y=True, alpha=0.3)
        self.qconv_plot.addLegend()
        self.qconv_pasqal = self.qconv_plot.plot(
            [], [], pen=pg.mkPen("m", width=2), symbol="o", symbolSize=6, name="Pasqal QAOA")
        self.qconv_gbs = self.qconv_plot.plot(
            [], [], pen=pg.mkPen("y", width=2), symbol="o", symbolSize=6, name="Xanadu GBS")
        self.qconv_opt = pg.InfiniteLine(angle=0, pen=pg.mkPen("g", style=Qt.PenStyle.DashLine))
        self.qconv_plot.addItem(self.qconv_opt)

        controls = QHBoxLayout()
        controls.addWidget(self.qconv_btn)
        controls.addWidget(self.qconv_status, stretch=1)
        lay = QVBoxLayout()
        lay.addLayout(controls)
        lay.addWidget(self.qconv_plot, stretch=1)
        wrap = QWidget()
        wrap.setLayout(lay)
        return wrap

    def _on_quantum_conv(self) -> None:
        if self._qconv_worker is not None and self._qconv_worker.isRunning():
            return
        if not self._evals:
            self.qconv_status.setText("build an instance first (Randomize)")
            return
        evals, conflicts, buckets = self._evals, self._conflicts, self._buckets
        seed = self.seed_spin.value()

        def job(emit):
            from contrail_env.pasqal_analog import solve_pasqal_analog
            from contrail_env.solver_cpsat import solve_cpsat
            from contrail_env.xanadu_gbs import solve_xanadu_gbs
            emit(("optimum", solve_cpsat(evals, conflicts, buckets, time_limit_s=5.0).objective))
            solve_pasqal_analog(evals, conflicts, buckets, bo_iters=12, seed=seed,
                                on_progress=lambda i, c: emit(("pasqal", i, c)))
            solve_xanadu_gbs(evals, conflicts, buckets, n_samples=800, seed=seed,
                             on_progress=lambda s, c: emit(("gbs", s, c)))
            return None

        self._qconv_data = {"pasqal": ([], []), "gbs": ([], [])}
        self.qconv_pasqal.setData([], [])
        self.qconv_gbs.setData([], [])
        self.qconv_btn.setEnabled(False)
        self.qconv_status.setText("running… (Pasqal QAOA can take a few seconds)")
        self._qconv_worker = _AnalysisWorker(job)
        self._qconv_worker.progress.connect(self._on_qconv_progress)
        self._qconv_worker.done.connect(self._on_qconv_done)
        self._qconv_worker.failed.connect(self._on_qconv_failed)
        self._qconv_worker.start()

    def _on_qconv_progress(self, msg) -> None:
        if msg[0] == "optimum":
            self.qconv_opt.setValue(float(msg[1]))
            return
        key = "pasqal" if msg[0] == "pasqal" else "gbs"
        xs, ys = self._qconv_data[key]
        xs.append(msg[1])
        ys.append(msg[2])
        (self.qconv_pasqal if key == "pasqal" else self.qconv_gbs).setData(xs, ys)

    def _on_qconv_done(self, _result) -> None:
        self.qconv_btn.setEnabled(True)
        self.qconv_status.setText("done — curves are best-cost-so-far; the dashed line is the CP-SAT optimum")

    def _on_qconv_failed(self, msg: str) -> None:
        self.qconv_btn.setEnabled(True)
        self.qconv_status.setText(f"failed: {msg}")

    # ------------------------------------------------ tab 3: hardness sweep ---
    def _build_hardness_tab(self) -> QWidget:
        self.hard_max = QSpinBox()
        self.hard_max.setRange(4, 16)
        self.hard_max.setValue(10)
        self.hard_btn = QPushButton("Run sweep")
        self.hard_btn.clicked.connect(self._on_hardness)
        self.hard_status = QLabel(
            "idle — solves a fresh instance at each size 2..N and records CP-SAT "
            "effort. A flat incumbents=1 line means the exact solver one-shots it."
        )
        self.hard_status.setFont(_MONO)

        self.hard_time_plot = pg.PlotWidget(title="CP-SAT wall time vs problem size")
        self.hard_time_plot.setLabel("bottom", "flights")
        self.hard_time_plot.setLabel("left", "wall time (ms)")
        self.hard_time_plot.showGrid(x=True, y=True, alpha=0.3)
        self.hard_time_curve = self.hard_time_plot.plot(
            [], [], pen=pg.mkPen("c", width=2), symbol="o", symbolSize=7)

        self.hard_inc_plot = pg.PlotWidget(
            title="CP-SAT incumbents found vs size (flat = trivially easy)")
        self.hard_inc_plot.setLabel("bottom", "flights")
        self.hard_inc_plot.setLabel("left", "# incumbents")
        self.hard_inc_plot.showGrid(x=True, y=True, alpha=0.3)
        self.hard_inc_curve = self.hard_inc_plot.plot(
            [], [], pen=pg.mkPen("m", width=2), symbol="o", symbolSize=7)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("max flights"))
        controls.addWidget(self.hard_max)
        controls.addWidget(self.hard_btn)
        controls.addWidget(self.hard_status, stretch=1)
        lay = QVBoxLayout()
        lay.addLayout(controls)
        lay.addWidget(self.hard_time_plot, stretch=1)
        lay.addWidget(self.hard_inc_plot, stretch=1)
        wrap = QWidget()
        wrap.setLayout(lay)
        return wrap

    def _on_hardness(self) -> None:
        if self._hard_worker is not None and self._hard_worker.isRunning():
            return
        seed = self.seed_spin.value()
        sizes = list(range(2, self.hard_max.value() + 1))

        def job(_emit):
            from contrail_env.analysis import hardness_sweep
            return hardness_sweep(sizes, seed=seed, time_limit_s=3.0)

        self.hard_btn.setEnabled(False)
        self.hard_status.setText("running sweep… (solves one instance per size)")
        self._hard_worker = _AnalysisWorker(job)
        self._hard_worker.done.connect(self._on_hardness_done)
        self._hard_worker.failed.connect(self._on_hardness_failed)
        self._hard_worker.start()

    def _on_hardness_done(self, out) -> None:
        self.hard_time_curve.setData(out["sizes"], out["cpsat_ms"])
        self.hard_inc_curve.setData(out["sizes"], out["incumbents"])
        self.hard_status.setText(
            f"done · {int(out['sizes'].min())}–{int(out['sizes'].max())} flights · "
            f"max {float(out['cpsat_ms'].max()):.0f} ms · "
            f"incumbents {int(out['incumbents'].min())}–{int(out['incumbents'].max())}"
        )
        self.hard_btn.setEnabled(True)

    def _on_hardness_failed(self, msg: str) -> None:
        self.hard_status.setText(f"failed: {msg}")
        self.hard_btn.setEnabled(True)

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
        _stretch_last(self.results)
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
            "idle — CP-SAT is the exact verifier at this size; the quantum + "
            "classical samplers are compared at matched output budget."
        )
        self.bench_status.setFont(_MONO)

        # Export buttons (enabled once a report exists — G3 reads it too).
        self.bench_export_csv_btn = QPushButton("Export CSV…")
        self.bench_export_csv_btn.clicked.connect(self._on_export_bench_csv)
        self.bench_export_png_btn = QPushButton("Export table PNG…")
        self.bench_export_png_btn.clicked.connect(self._on_export_bench_png)
        self.bench_export_csv_btn.setEnabled(False)
        self.bench_export_png_btn.setEnabled(False)

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
        controls.addWidget(self.bench_export_csv_btn)
        controls.addWidget(self.bench_export_png_btn)
        controls.addWidget(self.bench_status, stretch=1)

        # --- aggregate table ---------------------------------------------------
        self.bench_columns = [
            "Solver", "Backend", "E_best (mean)", "Approx ratio r", "r 95% CI",
            "Raw feas %", "Succ %", "TTS99 (med)", "Wall (ms)",
        ]
        self.bench_table = QTableWidget(len(SOLVER_NAMES), len(self.bench_columns))
        self.bench_table.setHorizontalHeaderLabels(self.bench_columns)
        _stretch_last(self.bench_table, hide_vertical=True)
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

        # Xanadu GBS + the classical SA baseline share this "cost vs samples"
        # plot (both stream best-cost-so-far against sample count); colour tells
        # them apart via SOLVER_PENS.
        self.xanadu_plot = pg.PlotWidget(
            title="Samplers — best repaired cost vs samples (Xanadu GBS + SA baseline)"
        )
        self.xanadu_plot.setLabel("bottom", "samples drawn")
        self.xanadu_plot.setLabel("left", "best combined cost")
        self.xanadu_plot.showGrid(x=True, y=True, alpha=0.3)
        self.xanadu_plot.addLegend()
        self.xanadu_curve = self.xanadu_plot.plot(
            [], [], pen=pg.mkPen(SOLVER_PENS["xanadu-gbs"], width=2),
            symbol="o", symbolSize=6, name="Xanadu GBS",
        )
        self.sa_curve = self.xanadu_plot.plot(
            [], [], pen=pg.mkPen(SOLVER_PENS["sa"], width=2),
            symbol="t", symbolSize=6, name="SA baseline",
        )
        self.xanadu_opt_line = pg.InfiniteLine(
            angle=0, pen=pg.mkPen("g", style=Qt.PenStyle.DashLine)
        )
        self.xanadu_plot.addItem(self.xanadu_opt_line)

        self._bench_curves: dict[str, tuple[list[int], list[float]]] = {
            "pasqal-analog": ([], []),
            "xanadu-gbs": ([], []),
            "sa": ([], []),
        }
        self._bench_curve_items = {
            "pasqal-analog": self.pasqal_curve,
            "xanadu-gbs": self.xanadu_curve,
            "sa": self.sa_curve,
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

    # ------------------------------------------------ tab 9: quantum dynamics ---
    def _build_dynamics_tab(self) -> QWidget:
        # Schedule spinboxes, clamped to the imported hardware envelope / BO box.
        self.dyn_T = QDoubleSpinBox()
        self.dyn_T.setRange(1500.0, T_MAX_NS)
        self.dyn_T.setSingleStep(100.0)
        self.dyn_T.setValue(4000.0)
        self.dyn_omega = QDoubleSpinBox()
        self.dyn_omega.setRange(0.1, OMEGA_MAX_HW)
        self.dyn_omega.setSingleStep(0.5)
        self.dyn_omega.setValue(10.0)
        self.dyn_di = QDoubleSpinBox()
        self.dyn_di.setRange(-14.0, -2.0)
        self.dyn_di.setSingleStep(0.5)
        self.dyn_di.setValue(-8.0)
        self.dyn_df = QDoubleSpinBox()
        self.dyn_df.setRange(2.0, 14.0)
        self.dyn_df.setSingleStep(0.5)
        self.dyn_df.setValue(8.0)
        self.dyn_steps = QSpinBox()
        self.dyn_steps.setRange(200, 1000)
        self.dyn_steps.setSingleStep(50)
        self.dyn_steps.setValue(500)
        self.dyn_records = QSpinBox()
        self.dyn_records.setRange(10, 50)
        self.dyn_records.setValue(25)

        self.dyn_load_btn = QPushButton("Load BO-best")
        self.dyn_load_btn.setEnabled(False)
        self.dyn_load_btn.clicked.connect(self._on_load_bo_best)
        self.dyn_btn = QPushButton("Compute dynamics")
        self.dyn_btn.clicked.connect(self._on_compute_dynamics)
        self.dyn_progress = QProgressBar()
        self.dyn_progress.setRange(0, 100)
        self.dyn_progress.setValue(0)
        self.dyn_progress.setFixedWidth(180)

        controls = QHBoxLayout()
        for label, widget in (
            ("T (ns)", self.dyn_T), ("Ω_max", self.dyn_omega),
            ("δ_init", self.dyn_di), ("δ_final", self.dyn_df),
            ("n_steps", self.dyn_steps), ("n_records", self.dyn_records),
        ):
            controls.addWidget(QLabel(label))
            controls.addWidget(widget)
        controls.addWidget(self.dyn_load_btn)
        controls.addWidget(self.dyn_btn)
        controls.addWidget(self.dyn_progress)
        controls.addStretch(1)

        # Residual-energy sweep controls + exports.
        self.dyn_nT = QSpinBox()
        self.dyn_nT.setRange(4, 12)
        self.dyn_nT.setValue(8)
        self.dyn_residual_btn = QPushButton("Run ε(T) sweep")
        self.dyn_residual_btn.clicked.connect(self._on_run_residual)
        self.dyn_export_csv_btn = QPushButton("Export CSV…")
        self.dyn_export_csv_btn.clicked.connect(self._on_export_dynamics_csv)
        self.dyn_export_png_btn = QPushButton("Export PNG…")
        self.dyn_export_png_btn.clicked.connect(self._on_export_dynamics_png)
        controls2 = QHBoxLayout()
        controls2.addWidget(QLabel("ε(T) points"))
        controls2.addWidget(self.dyn_nT)
        controls2.addWidget(self.dyn_residual_btn)
        controls2.addStretch(1)
        controls2.addWidget(self.dyn_export_csv_btn)
        controls2.addWidget(self.dyn_export_png_btn)

        # 2x2 diagnostics grid.
        self.dyn_glw = pg.GraphicsLayoutWidget()
        self.dyn_levels_plot = self.dyn_glw.addPlot(row=0, col=0, title="Levels & gap")
        self.dyn_levels_plot.setLabel("bottom", "s = t/T")
        self.dyn_levels_plot.setLabel("left", "energy (rad/µs)")
        self.dyn_levels_plot.showGrid(x=True, y=True, alpha=0.3)
        self._dyn_level_curves = [
            self.dyn_levels_plot.plot([], [], pen=pg.mkPen((130, 130, 140), width=1))
            for _ in range(6)
        ]
        self._dyn_gap_curve = self.dyn_levels_plot.plot(
            [], [], pen=pg.mkPen("w", width=2), name="Δ(s)")
        self.dyn_sstar_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen("r", style=Qt.PenStyle.DashLine))
        self.dyn_levels_plot.addItem(self.dyn_sstar_line)

        self.dyn_entropy_plot = self.dyn_glw.addPlot(row=0, col=1, title="Entanglement entropy")
        self.dyn_entropy_plot.setLabel("bottom", "t (ns)")
        self.dyn_entropy_plot.setLabel("left", "S_A (nats)")
        self.dyn_entropy_plot.showGrid(x=True, y=True, alpha=0.3)
        self.dyn_entropy_plot.addLegend()
        self._dyn_entropy_curves = {
            "flights_half": self.dyn_entropy_plot.plot(
                [], [], pen=pg.mkPen(_CUT_PENS[0], width=2), name="flights_half"),
            "index_half": self.dyn_entropy_plot.plot(
                [], [], pen=pg.mkPen(_CUT_PENS[1], width=2), name="index_half"),
        }

        self.dyn_pground_plot = self.dyn_glw.addPlot(row=1, col=0, title="Ground-space population")
        self.dyn_pground_plot.setLabel("bottom", "t (ns)")
        self.dyn_pground_plot.setLabel("left", "P₀(t)")
        self.dyn_pground_plot.setYRange(0.0, 1.0)
        self.dyn_pground_plot.showGrid(x=True, y=True, alpha=0.3)
        self._dyn_pground_curve = self.dyn_pground_plot.plot(
            [], [], pen=pg.mkPen("c", width=2))

        self.dyn_energy_plot = self.dyn_glw.addPlot(row=1, col=1, title="⟨E_pen⟩(t)")
        self.dyn_energy_plot.setLabel("bottom", "t (ns)")
        self.dyn_energy_plot.setLabel("left", "⟨E_pen⟩")
        self.dyn_energy_plot.showGrid(x=True, y=True, alpha=0.3)
        self._dyn_energy_curve = self.dyn_energy_plot.plot(
            [], [], pen=pg.mkPen("m", width=2))
        self.dyn_energy_min_line = pg.InfiniteLine(
            angle=0, pen=pg.mkPen("g", style=Qt.PenStyle.DashLine))
        self.dyn_energy_min_line.setVisible(False)
        self.dyn_energy_plot.addItem(self.dyn_energy_min_line)

        # Residual-energy sweep plot (log-log).
        self.dyn_residual_plot = pg.PlotWidget(title="Residual energy ε(T) — run the sweep")
        self.dyn_residual_plot.setLabel("bottom", "T (ns)")
        self.dyn_residual_plot.setLabel("left", "ε(T)")
        self.dyn_residual_plot.setLogMode(x=True, y=True)
        self.dyn_residual_plot.showGrid(x=True, y=True, alpha=0.3)
        self.dyn_residual_curve = self.dyn_residual_plot.plot(
            [], [], pen=pg.mkPen("y", width=2), symbol="o", symbolSize=6)

        self.dyn_info = QLabel("Δ_min: —   s*: —   end_degeneracy: —   |P₀| exact: —")
        self.dyn_info.setFont(_MONO)
        self.dyn_status = QLabel(_DYN_HONESTY)
        self.dyn_status.setFont(_MONO)
        self.dyn_status.setWordWrap(True)

        plots_row = QHBoxLayout()
        plots_row.addWidget(self.dyn_glw, stretch=3)
        plots_row.addWidget(self.dyn_residual_plot, stretch=1)
        plots_w = QWidget()
        plots_w.setLayout(plots_row)

        root = QVBoxLayout()
        root.addLayout(controls)
        root.addLayout(controls2)
        root.addWidget(plots_w, stretch=1)
        root.addWidget(self.dyn_info)
        root.addWidget(self.dyn_status)
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
        """The GeoAnchor placing the synthetic ISSR field on the map.

        The field carries no anchor of its own (geography only matters for the
        map), so we place it with the canonical European anchor that matches the
        default world geometry.
        """
        return getattr(field, "anchor", None) or EUROPEAN_ANCHOR

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

    def _one_series(self, flight, profile, color: str, chosen: bool) -> ProfileSeries:
        """An altitude profile (FL vs along-track distance) + per-point ISSR mask."""
        wps = waypoints_for(flight, profile)
        x = np.array([w[0] for w in wps], dtype=float)
        y = np.array([w[1] for w in wps], dtype=float)
        z = np.array([w[2] for w in wps], dtype=float)
        s_km = np.hypot(x - x[0], y - y[0])          # distance flown from origin
        fl = z / 0.3048 / 100.0                      # metres -> flight level
        assert self._world is not None               # callers render only after build
        issr = np.asarray(self._world.issr.mask_grid(x, y, z), dtype=np.bool_)
        return ProfileSeries(name=flight.name, s_km=s_km, fl=fl, issr_mask=issr,
                             color=color, chosen=chosen)

    def _profile_series(self, chosen_by_flight: dict[str, int] | None) -> list[ProfileSeries]:
        """Per flight: its filed baseline, plus the chosen profile once solved."""
        series: list[ProfileSeries] = []
        for idx, flight in enumerate(self._flights):
            color = _FLIGHT_COLORS[idx % len(_FLIGHT_COLORS)]
            series.append(self._one_series(flight, flight.baseline, color, chosen=False))
            if chosen_by_flight and flight.name in chosen_by_flight:
                members = self._groups.get(flight.name, [])
                k = chosen_by_flight[flight.name]
                if 0 <= k < len(members):
                    prof = self._evals[members[k]].profile
                    series.append(self._one_series(flight, prof, color, chosen=True))
        return series

    def _render_map(self, chosen_by_flight: dict[str, int] | None) -> None:
        """Render the altitude-profile panel + the inset ISSR map.

        The main panel shows flight level vs along-track distance (filed vs
        chosen, with ISSR crossings marked) so the optimiser's altitude decision
        is visible; the inset gives geographic context. The risk field for the
        inset is sampled at a fixed altitude slice and mapped to lon/lat.
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
        profiles = self._profile_series(chosen_by_flight)
        source = getattr(field, "source", "synthetic")
        fig = build_map_figure(source=source, profiles=profiles,
                               lon=lons, lat=lats, risk=risk, routes=routes)

        # Write a self-contained page (plotly.js inlined) and load it from disk —
        # setHtml caps payloads at ~2 MB, which the inlined library exceeds. A
        # cache-busting query forces the view to re-read after each solve. The
        # basemap vectors are inlined into <head> so the geo map needs no CDN.
        html = fig.to_html(include_plotlyjs="inline", full_html=True,
                           config={"displayModeBar": True})
        assets = _geo_assets_script()
        if assets:
            html = html.replace("</head>", assets + "</head>", 1)
        with open(self._map_html_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        if self.map_view is not None:
            url = QUrl.fromLocalFile(self._map_html_path)
            url.setQuery(f"v={time.time()}")
            self.map_view.load(url)

    # ---------------------------------------------------------------- config --
    def _build_cfg(self) -> ScenarioConfig:
        return ScenarioConfig(
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
        )

    # ------------------------------------------------- problem reconstruction --
    def _rebuild_structure(self, cfg: ScenarioConfig) -> None:
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
        self._buckets = buckets
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
        self._update_dynamics_budget()  # the instance changed size

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
        self.results.setRowCount(0)
        self._rebuild_structure(self._build_cfg())

    def _on_solve(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        cfg = self._build_cfg()
        self._pending_cfg = cfg

        self._rebuild_structure(cfg)
        self.results.setRowCount(0)
        self.summary.setText("solving…")
        self.solve_btn.setEnabled(False)

        self._worker = SolveWorker(cfg)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

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

        for name, (xs, ys) in self._bench_curves.items():
            xs.clear()
            ys.clear()
            self._bench_curve_items[name].setData([], [])
        for row in range(self.bench_table.rowCount()):
            for col in range(1, self.bench_table.columnCount()):
                self.bench_table.setItem(row, col, QTableWidgetItem("…"))
        self.bench_export_csv_btn.setEnabled(False)
        self.bench_export_png_btn.setEnabled(False)

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
        self._bench_curve_items[solver].setData(xs, ys)
        self._bench_msg = f"{solver}  step {step}  best {cost:.1f}"
        self._refresh_bench_status()

    def _on_bench_cell(self, seed: int, cell: object) -> None:
        self._bench_cells_done += 1
        self.bench_progress.setValue(self._bench_cells_done * 100)
        # CP-SAT finishing marks the start of a new instance: reset the
        # sampler convergence curves and pin the optimum reference lines.
        if cell.solver == "cpsat":  # type: ignore[attr-defined]
            for name, (xs, ys) in self._bench_curves.items():
                xs.clear()
                ys.clear()
                self._bench_curve_items[name].setData([], [])
            optimum = cell.best_cost  # type: ignore[attr-defined]
            self.pasqal_opt_line.setValue(optimum)
            self.xanadu_opt_line.setValue(optimum)
            self._bench_msg = f"seed {seed}:  E* = {optimum:.1f}  (CP-SAT)"
            self._refresh_bench_status()

    def _on_bench_done(self, report: BenchmarkReport) -> None:
        self._bench_report = report
        stats = report.aggregate()
        instances = report.instances

        # Backend / note per solver from the most recent instance.
        latest: dict[str, object] = {}
        for inst in instances:
            for cell_run in inst.runs:
                latest[cell_run.solver] = cell_run

        means: list[float] = []
        err_low: list[float] = []
        err_high: list[float] = []
        for row, name in enumerate(SOLVER_NAMES):
            s = stats.get(name)
            run = latest.get(name)
            backend = getattr(run, "backend", "-")
            if s is not None:
                succ = "—" if math.isnan(s.success_mean) else f"{100 * s.success_mean:.1f}"
                values = [
                    backend,
                    _mean_best_cost(instances, name),
                    f"{s.ratio_mean:.4f}",
                    f"[{s.ratio_ci_low:.4f}, {s.ratio_ci_high:.4f}]",
                    f"{100 * s.feasibility_mean:.1f}",
                    succ,
                    _fmt_tts_s(s.tts_median_s),
                    f"{1000 * s.wall_clock_mean_s:.0f}",
                ]
                means.append(s.ratio_mean)
                err_low.append(s.ratio_mean - s.ratio_ci_low)
                err_high.append(s.ratio_ci_high - s.ratio_mean)
            else:
                note = getattr(run, "note", "no data")
                status = getattr(run, "status", "—")
                values = [backend, "—", "—", "—", "—", "—", "—", f"{status}: {note}"]
                means.append(0.0)
                err_low.append(0.0)
                err_high.append(0.0)
            for col, val in enumerate(values, start=1):
                self.bench_table.setItem(row, col, QTableWidgetItem(str(val)))

        # Ratio bars, one SOLVER_PENS colour per solver, with bootstrap CIs.
        if self._bench_bar_item is not None:
            self.bench_bars.removeItem(self._bench_bar_item)
        if self._bench_err_item is not None:
            self.bench_bars.removeItem(self._bench_err_item)
        x = np.arange(len(SOLVER_NAMES))
        self._bench_bar_item = pg.BarGraphItem(
            x=x, height=means, width=0.55,
            brushes=[pg.mkBrush(SOLVER_PENS[name]) for name in SOLVER_NAMES],
        )
        self._bench_err_item = pg.ErrorBarItem(
            x=x, y=np.array(means),
            top=np.array(err_high), bottom=np.array(err_low),
            beam=0.18, pen=pg.mkPen("w", width=2),
        )
        self.bench_bars.addItem(self._bench_bar_item)
        self.bench_bars.addItem(self._bench_err_item)
        positive = [m for m in means if m > 0]
        self.bench_bars.setYRange(min(0.95, min(positive) - 0.02) if positive else 0.0, 1.01)

        self._bench_timer.stop()
        self.bench_progress.setValue(self.bench_progress.maximum())
        self.bench_export_csv_btn.setEnabled(True)
        self.bench_export_png_btn.setEnabled(True)
        self._update_load_bo_best_state()  # G2 Load BO-best button
        self._refresh_fingerprint_tab()  # G3 reads self._bench_report
        elapsed = int(time.monotonic() - self._bench_t0)
        n_inst = len(instances)
        self.bench_status.setText(
            f"done in {elapsed // 60}:{elapsed % 60:02d} — {n_inst} seed(s) × "
            f"{len(SOLVER_NAMES)} solvers; CP-SAT is the exact verifier, the rest "
            f"are compared at matched output budget"
        )
        self.bench_btn.setEnabled(True)

    def _on_export_bench_csv(self) -> None:
        if self._bench_report is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export benchmark CSV", "", "CSV files (*.csv)")
        if path:
            self._bench_report.to_csv(path)

    def _on_export_bench_png(self) -> None:
        _export_plot_png(self, self.bench_bars)

    # ------------------------------------------------ tab 9: quantum dynamics ---
    def _latest_pasqal_run(self):
        """The most recent OK pasqal-analog run with a schedule in its meta."""
        if self._bench_report is None:
            return None
        latest = None
        for inst in self._bench_report.instances:
            run = inst.run_for("pasqal-analog")
            if run is not None and run.status == "OK" and run.meta:
                latest = run
        return latest

    def _update_load_bo_best_state(self) -> None:
        self.dyn_load_btn.setEnabled(self._latest_pasqal_run() is not None)

    def _update_dynamics_budget(self) -> None:
        """Mirror the science budget guard: disable compute above 16 qubits."""
        n = len(self._evals)
        over = n > MAX_STATEVECTOR_QUBITS - 4  # spectral/dynamics cap is 16
        self.dyn_btn.setEnabled(not over)
        self.dyn_residual_btn.setEnabled(not over)
        if over:
            self.dyn_status.setText(f"n = {n} option qubits > 16 — reduce flights")
        else:
            self.dyn_status.setText(_DYN_HONESTY)

    def _on_load_bo_best(self) -> None:
        run = self._latest_pasqal_run()
        if run is None:
            return
        meta = run.meta
        self.dyn_T.setValue(float(meta.get("T_ns", self.dyn_T.value())))  # type: ignore[arg-type]
        self.dyn_omega.setValue(
            float(meta.get("omega_max_rad_us", self.dyn_omega.value())))  # type: ignore[arg-type]
        self.dyn_di.setValue(
            float(meta.get("delta_init_rad_us", self.dyn_di.value())))  # type: ignore[arg-type]
        self.dyn_df.setValue(
            float(meta.get("delta_final_rad_us", self.dyn_df.value())))  # type: ignore[arg-type]
        self.dyn_status.setText("loaded the BO-best schedule from the last Pasqal run")

    def _on_compute_dynamics(self) -> None:
        if self._dyn_worker is not None and self._dyn_worker.isRunning():
            return
        if not self._evals:
            self.dyn_status.setText("build an instance first (Randomize)")
            return
        n = len(self._evals)
        if n > MAX_STATEVECTOR_QUBITS - 4:
            self.dyn_status.setText(f"n = {n} option qubits > 16 — reduce flights")
            return
        evals, conflicts, buckets = self._evals, self._conflicts, self._buckets
        t_ns, omega = self.dyn_T.value(), self.dyn_omega.value()
        d_init, d_final = self.dyn_di.value(), self.dyn_df.value()
        n_steps, n_records = self.dyn_steps.value(), self.dyn_records.value()

        def job(emit):
            from contrail_env.analysis import feasible_cost_landscape, ground_state_degeneracy
            from contrail_env.dynamics import run_with_diagnostics
            from contrail_env.pasqal_analog import AnnealSchedule, penalized_energy_vector
            from contrail_env.quantum_common import build_option_graph
            from contrail_env.spectral import instantaneous_spectrum

            graph = build_option_graph(evals, conflicts, buckets)
            schedule = AnnealSchedule(
                T_ns=t_ns, omega_max=omega, delta_init=d_init, delta_final=d_final)

            def on_phase(msg: str, frac: float) -> None:
                emit(("phase", msg, frac))

            spectrum = instantaneous_spectrum(graph, schedule, on_phase=on_phase)
            record = run_with_diagnostics(
                graph, schedule, n_steps=n_steps, n_records=n_records, on_phase=on_phase)
            min_epen = float(penalized_energy_vector(graph).min())
            try:
                costs, _opt = feasible_cost_landscape(evals, conflicts, buckets)
                n_ground = ground_state_degeneracy(costs, float(costs.min()))
            except ValueError:
                n_ground = None
            return {
                "spectrum": spectrum, "record": record,
                "min_epen": min_epen, "n_ground": n_ground,
            }

        self.dyn_btn.setEnabled(False)
        self.dyn_progress.setValue(0)
        self.dyn_status.setText("computing spectrum + dynamics…")
        self._dyn_worker = _AnalysisWorker(job)
        self._dyn_worker.progress.connect(self._on_dyn_progress)
        self._dyn_worker.done.connect(self._on_dynamics_done)
        self._dyn_worker.failed.connect(self._on_dynamics_failed)
        self._dyn_worker.start()

    def _on_dyn_progress(self, payload) -> None:
        if isinstance(payload, tuple) and payload and payload[0] == "phase":
            _tag, msg, frac = payload
            self.dyn_progress.setValue(int(100 * min(1.0, max(0.0, frac))))
            self.dyn_status.setText(msg)

    def _on_dynamics_done(self, payload) -> None:
        self._render_dynamics(
            payload["spectrum"], payload["record"],
            payload.get("min_epen"), payload.get("n_ground"))
        self.dyn_progress.setValue(100)
        self.dyn_btn.setEnabled(True)
        self.dyn_status.setText(_DYN_HONESTY)

    def _on_dynamics_failed(self, msg: str) -> None:
        self.dyn_btn.setEnabled(True)
        self.dyn_status.setText(f"failed: {msg}")

    def _render_dynamics(self, spectrum, record, min_epen=None, n_ground=None) -> None:
        """Draw the 2x2 diagnostics grid + info row from plain data (no Qt in)."""
        self._dyn_spectrum = spectrum
        self._dyn_record = record

        s = spectrum.s
        k = spectrum.energies.shape[1]
        for i, curve in enumerate(self._dyn_level_curves):
            curve.setData(s, spectrum.energies[:, i]) if i < k else curve.setData([], [])
        self._dyn_gap_curve.setData(s, spectrum.gap)
        self.dyn_sstar_line.setValue(spectrum.s_star)
        dmin = spectrum.delta_min
        t_adiab = (self.dyn_omega.value() / (dmin * dmin)) * 1000.0 if dmin > 0 else float("inf")
        self.dyn_levels_plot.setTitle(
            f"Levels & gap — Δ_min={dmin:.3f} rad/µs @ s*={spectrum.s_star:.2f}; "
            f"T_adiab~{t_adiab:.0f} ns vs chosen T={self.dyn_T.value():.0f} ns")

        for name, curve in self._dyn_entropy_curves.items():
            if name in record.entropies:
                curve.setData(record.t_ns, record.entropies[name])
            else:
                curve.setData([], [])

        self._dyn_pground_curve.setData(record.t_ns, record.p_ground)
        self._dyn_energy_curve.setData(record.t_ns, record.energy_pen)
        if min_epen is not None:
            self.dyn_energy_min_line.setValue(min_epen)
            self.dyn_energy_min_line.setVisible(True)
        else:
            self.dyn_energy_min_line.setVisible(False)

        parts = [
            f"Δ_min = {spectrum.delta_min:.4f}",
            f"s* = {spectrum.s_star:.3f}",
            f"end_degeneracy = {spectrum.end_degeneracy}",
        ]
        if n_ground is not None:
            agree = "agree" if n_ground == spectrum.end_degeneracy else "differ"
            parts.append(f"|P₀| exact = {n_ground} ({agree})")
        else:
            parts.append("|P₀| exact = —")
        self.dyn_info.setText("   ".join(parts))

    def _on_run_residual(self) -> None:
        if self._dyn_residual_worker is not None and self._dyn_residual_worker.isRunning():
            return
        if not self._evals:
            self.dyn_status.setText("build an instance first (Randomize)")
            return
        n = len(self._evals)
        if n > MAX_STATEVECTOR_QUBITS - 4:
            self.dyn_status.setText(f"n = {n} option qubits > 16 — reduce flights")
            return
        evals, conflicts, buckets = self._evals, self._conflicts, self._buckets
        omega, d_init, d_final = self.dyn_omega.value(), self.dyn_di.value(), self.dyn_df.value()
        n_T, n_steps = self.dyn_nT.value(), self.dyn_steps.value()

        def job(emit):
            from contrail_env.dynamics import residual_energy_vs_T
            from contrail_env.pasqal_analog import AnnealSchedule
            from contrail_env.quantum_common import build_option_graph

            graph = build_option_graph(evals, conflicts, buckets)
            base = AnnealSchedule(
                T_ns=4000.0, omega_max=omega, delta_init=d_init, delta_final=d_final)
            t_grid = np.geomspace(600.0, T_MAX_NS, n_T)

            def on_phase(msg: str, frac: float) -> None:
                emit(("phase", msg, frac))

            return residual_energy_vs_T(graph, base, t_grid, n_steps=n_steps, on_phase=on_phase)

        self.dyn_residual_btn.setEnabled(False)
        self.dyn_status.setText("running ε(T) sweep…")
        self._dyn_residual_worker = _AnalysisWorker(job)
        self._dyn_residual_worker.progress.connect(self._on_dyn_progress)
        self._dyn_residual_worker.done.connect(self._on_residual_done)
        self._dyn_residual_worker.failed.connect(self._on_residual_failed)
        self._dyn_residual_worker.start()

    def _on_residual_done(self, result) -> None:
        self._dyn_residual = result
        self.dyn_residual_curve.setData(result.T_ns, result.residual)
        self.dyn_residual_plot.setTitle(
            f"Residual energy ε(T) — μ={result.mu:.3f}, R²={result.r2:.3f}")
        self.dyn_residual_btn.setEnabled(True)
        self.dyn_status.setText(_DYN_HONESTY)

    def _on_residual_failed(self, msg: str) -> None:
        self.dyn_residual_btn.setEnabled(True)
        self.dyn_status.setText(f"residual sweep failed: {msg}")

    def _on_export_dynamics_csv(self) -> None:
        if self._dyn_record is None or self._dyn_spectrum is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export dynamics CSV", "", "CSV files (*.csv)")
        if not path:
            return
        import csv

        rec, spec = self._dyn_record, self._dyn_spectrum
        cut_names = list(rec.entropies.keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["s", "t_ns", "gap", "p_ground", "energy_pen"]
                            + [f"S_{c}" for c in cut_names])
            for i in range(len(rec.s)):
                writer.writerow(
                    [rec.s[i], rec.t_ns[i], rec.gap[i], rec.p_ground[i], rec.energy_pen[i]]
                    + [rec.entropies[c][i] for c in cut_names])
        # Sibling levels file: s, E_0..E_{k-1}.
        levels_path = (path[:-4] if path.endswith(".csv") else path) + "_levels.csv"
        with open(levels_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["s"] + [f"E_{j}" for j in range(spec.energies.shape[1])])
            for i in range(len(spec.s)):
                writer.writerow([spec.s[i], *list(spec.energies[i])])
        # Optional residual sweep file.
        if self._dyn_residual is not None:
            res_path = (path[:-4] if path.endswith(".csv") else path) + "_residual.csv"
            with open(res_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["T_ns", "residual"])
                for t_ns, res in zip(self._dyn_residual.T_ns, self._dyn_residual.residual,
                                     strict=True):
                    writer.writerow([t_ns, res])
        self.dyn_status.setText(f"exported dynamics arrays to {path}")

    def _on_export_dynamics_png(self) -> None:
        _export_plot_png(self, self.dyn_glw)

    def _refresh_fingerprint_tab(self) -> None:
        """Populated by the constraint-fingerprint tab (G3); no-op until then."""

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
