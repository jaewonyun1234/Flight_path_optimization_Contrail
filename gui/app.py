r"""
app.py — PyQt6 desktop client for the contrail CP-SAT solver service.

The GUI never solves anything itself: it sends a ScenarioConfig to the gRPC
server, live-plots the convergence curve as ZMQ progress arrives, draws the
problem (2D ISSR heatmap + interactive 3D trajectories), and tabulates the
chosen option per flight. That client/server split is the whole point.

The scenario is rebuilt locally from the same (seeded) config the server used,
via service.scenario — so what you SEE is exactly what the server SOLVED.

THREADING MODEL
===============
All network + solve work happens on a QThread worker (`SolveWorker`). The
worker talks to the UI ONLY through Qt signals (progress / finished / failed);
it never touches a widget directly. Inside the worker, the blocking gRPC call
runs on a small helper thread so the worker can read the ZMQ progress stream
at the same time.

RUN
===
    pip install -e .
    bash scripts/gen_proto.sh        # Windows: .\scripts\gen_proto.ps1
    python -m service.server         # in one terminal
    python gui/app.py                # in another
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
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

from contrail_env import enumerate_option_specs, fl_to_m, m_to_fl, waypoints_for
from service.client import DEFAULT_SERVER_ADDRESS, SolverClient
from service.generated import solver_pb2
from service.progress import DEFAULT_SUB_ADDRESS, subscribe
from service.scenario import build_world_and_flights

# Fixed scenario knobs not exposed as controls (sensible defaults per the brief).
CORRIDOR_FRAC = 0.05
SNAPSHOT_WINDOW_S = 300.0
# Head start (seconds) for the SUB socket to connect before the solve emits.
_SUB_HEAD_START_S = 0.25

# 3D display: altitude (flight level) is exaggerated against the horizontal km
# so climbs/descents are visible. FL_CENTER maps to z=0 in the scene.
_VSCALE = 6.0
_FL_CENTER = 370.0


def _flight_color_gl(i: int) -> tuple[float, float, float, float]:
    """A distinct RGBA (0-1) color per flight for the 3D lines."""
    c = pg.intColor(i, hues=9, alpha=255)
    return (c.red() / 255.0, c.green() / 255.0, c.blue() / 255.0, 1.0)


def _flight_pen_2d(i: int):
    return pg.mkPen(pg.intColor(i, hues=9), width=2)


# =============================================================================
# WORKER THREAD
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
        self.setWindowTitle("Contrail Optimizer — CP-SAT over gRPC/ZMQ")

        self._worker: SolveWorker | None = None
        self._xs: list[int] = []
        self._ys: list[float] = []

        # Scene state, kept in sync with the last-built config.
        self._world = None
        self._flights: list = []
        self._last_chosen: dict[str, int] | None = None
        self._pending_cfg: solver_pb2.ScenarioConfig | None = None

        self._route_items: list[pg.PlotDataItem] = []
        self._gl_items: list = []

        self._build_ui()
        self._refresh_scene(self._build_cfg(), chosen=None)

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
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

        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(0.1, 120.0)
        self.time_spin.setSingleStep(1.0)
        self.time_spin.setValue(10.0)

        self.fl_combo = QComboBox()
        self.fl_combo.addItems(["FL340", "FL360", "FL380", "FL400"])
        self.fl_combo.setCurrentText("FL360")
        self.fl_combo.currentTextChanged.connect(self._on_fl_changed)

        self.solve_btn = QPushButton("Solve")
        self.solve_btn.clicked.connect(self._on_solve)

        # Rebuild the scene live when the scenario shape changes.
        for spin in (self.seed_spin, self.flights_spin, self.blobs_spin):
            spin.valueChanged.connect(self._on_scenario_changed)

        form = QFormLayout()
        form.addRow("Seed", self.seed_spin)
        form.addRow("Flights", self.flights_spin)
        form.addRow("ISSR blobs", self.blobs_spin)
        form.addRow("Contrail weight (beta)", self.beta_spin)
        form.addRow("Time limit (s)", self.time_spin)
        form.addRow("2D map flight level", self.fl_combo)
        form.addRow(self.solve_btn)

        controls_box = QGroupBox("Scenario")
        controls_box.setLayout(form)

        self.status_label = QLabel("idle")
        self.status_label.setWordWrap(True)

        # --- Results table --------------------------------------------------
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Flight", "Option", "Fuel (kg)", "Contrail cells", "Disruption (FL-min)"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)

        left = QVBoxLayout()
        left.addWidget(controls_box)
        left.addWidget(self.status_label)
        left.addWidget(QLabel("Chosen options"))
        left.addWidget(self.table, stretch=1)
        left_widget = QWidget()
        left_widget.setLayout(left)

        # --- Live objective plot -------------------------------------------
        self.plot = pg.PlotWidget(title="Best objective vs improvement")
        self.plot.setLabel("bottom", "improvement index")
        self.plot.setLabel("left", "objective")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot([], [], pen=pg.mkPen("c", width=2), symbol="o", symbolSize=6)

        # --- 3D trajectory view --------------------------------------------
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=1700, elevation=22, azimuth=-60)
        gl_note = QLabel(
            "Drag to rotate · scroll to zoom.  Vertical axis = flight level "
            "(exaggerated).  Orange cloud = ISSR contrail zones."
        )
        gl_note.setWordWrap(True)
        gl_panel = QVBoxLayout()
        gl_panel.addWidget(self.gl_view, stretch=1)
        gl_panel.addWidget(gl_note)
        gl_widget = QWidget()
        gl_widget.setLayout(gl_panel)

        # --- 2D ISSR map + routes ------------------------------------------
        self.map_widget = pg.PlotWidget(title="ISSR field + flight routes (top-down)")
        self.map_widget.setLabel("bottom", "x (km)")
        self.map_widget.setLabel("left", "y (km)")
        self.img = pg.ImageItem()
        try:
            self.img.setColorMap(pg.colormap.get("viridis"))
        except Exception:
            pass  # colormap is cosmetic; never block the UI on it
        self.map_widget.addItem(self.img)

        # Tabs: show the 3D view first (the headline), 2D map second.
        self.view_tabs = QTabWidget()
        self.view_tabs.addTab(gl_widget, "3D trajectories")
        self.view_tabs.addTab(self.map_widget, "2D ISSR map")

        right = QVBoxLayout()
        right.addWidget(self.plot, stretch=1)
        right.addWidget(self.view_tabs, stretch=2)
        right_widget = QWidget()
        right_widget.setLayout(right)

        root = QHBoxLayout()
        root.addWidget(left_widget, stretch=0)
        root.addWidget(right_widget, stretch=1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

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
            progress_topic=topic,
        )

    # -------------------------------------------------------------- rendering --
    def _refresh_scene(
        self, cfg: solver_pb2.ScenarioConfig, chosen: dict[str, int] | None
    ) -> None:
        """Rebuild world + flights from cfg, then redraw both views."""
        try:
            self._world, self._flights = build_world_and_flights(cfg)
        except Exception as exc:
            self.status_label.setText(f"scene build failed: {exc}")
            return
        self._last_chosen = chosen
        self._render_map()
        self._render_3d(chosen)

    def _selected_fl(self) -> int:
        return int(self.fl_combo.currentText().replace("FL", ""))

    def _render_map(self) -> None:
        if self._world is None:
            return
        g = self._world.grid
        xx, yy = g.meshgrid_xy()
        zz = np.full_like(xx, fl_to_m(self._selected_fl()))
        field = self._world.issr.rhi_excess_grid(xx, yy, zz)

        self.img.setImage(field, autoLevels=True)
        self.img.setRect(
            g.x_min_km, g.y_min_km, g.x_max_km - g.x_min_km, g.y_max_km - g.y_min_km
        )

        for item in self._route_items:
            self.map_widget.removeItem(item)
        self._route_items = []
        for i, flight in enumerate(self._flights):
            ox, oy = flight.origin_km
            dx, dy = flight.destination_km
            line = self.map_widget.plot([ox, dx], [oy, dy], pen=_flight_pen_2d(i))
            self._route_items.append(line)

    def _render_3d(self, chosen: dict[str, int] | None) -> None:
        if self._world is None:
            return
        try:
            self._render_3d_inner(chosen)
        except Exception as exc:
            self.status_label.setText(f"3D render failed: {exc}")

    def _render_3d_inner(self, chosen: dict[str, int] | None) -> None:
        g = self._world.grid
        cx = (g.x_min_km + g.x_max_km) / 2.0
        cy = (g.y_min_km + g.y_max_km) / 2.0

        def to_scene(x_km: float, y_km: float, fl: float) -> tuple[float, float, float]:
            return (x_km - cx, y_km - cy, (fl - _FL_CENTER) * _VSCALE)

        for item in self._gl_items:
            self.gl_view.removeItem(item)
        self._gl_items = []

        # Reference grid at the FL340 plane.
        grid = gl.GLGridItem()
        grid.setSize(g.x_max_km - g.x_min_km, g.y_max_km - g.y_min_km)
        grid.setSpacing(100, 100)
        grid.translate(0, 0, (340 - _FL_CENTER) * _VSCALE)
        self.gl_view.addItem(grid)
        self._gl_items.append(grid)

        # ISSR zones as a translucent orange point cloud.
        rng = np.random.default_rng(0)
        pts: list[tuple[float, float, float]] = []
        for blob in self._world.issr.blobs:
            n = 120
            xs = rng.normal(blob.cx_km, blob.sigma_h_km, n)
            ys = rng.normal(blob.cy_km, blob.sigma_h_km, n)
            zs = rng.normal(blob.cz_m, blob.sigma_v_m, n)
            for x, y, zm in zip(xs, ys, zs, strict=True):
                fl = max(336, min(404, m_to_fl(zm)))
                pts.append(to_scene(float(x), float(y), fl))
        if pts:
            cloud = gl.GLScatterPlotItem(
                pos=np.array(pts, dtype=float), color=(1.0, 0.5, 0.15, 0.18), size=4.0
            )
            self.gl_view.addItem(cloud)
            self._gl_items.append(cloud)

        # One 3D polyline per flight (chosen option if known, else baseline).
        for i, flight in enumerate(self._flights):
            specs = enumerate_option_specs(flight)
            idx = chosen.get(flight.name, 0) if chosen else 0
            idx = max(0, min(idx, len(specs) - 1))
            waypoints = waypoints_for(flight, specs[idx].profile)
            line = np.array(
                [to_scene(x, y, m_to_fl(z)) for (x, y, z, _t) in waypoints],
                dtype=float,
            )
            traj = gl.GLLinePlotItem(
                pos=line, color=_flight_color_gl(i), width=3.0, antialias=True
            )
            self.gl_view.addItem(traj)
            self._gl_items.append(traj)

    # ---------------------------------------------------------------- slots ---
    def _on_scenario_changed(self, _value: int) -> None:
        self._refresh_scene(self._build_cfg(), chosen=None)

    def _on_fl_changed(self, _text: str) -> None:
        self._render_map()  # only the 2D heatmap layer depends on the chosen FL

    def _on_solve(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        topic = f"solve/{self.seed_spin.value()}-{int(time.time() * 1000)}"
        cfg = self._build_cfg(topic)
        self._pending_cfg = cfg

        self._refresh_scene(cfg, chosen=None)  # show baseline routes immediately
        self._xs, self._ys = [], []
        self.curve.setData([], [])
        self.status_label.setText("solving…")
        self.solve_btn.setEnabled(False)

        self._worker = SolveWorker(cfg)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, improvement: int, objective: float) -> None:
        self._xs.append(improvement)
        self._ys.append(objective)
        self.curve.setData(self._xs, self._ys)

    def _on_finished(self, resp: object) -> None:
        choices = resp.choices  # type: ignore[attr-defined]
        self.table.setRowCount(len(choices))
        for row, c in enumerate(choices):
            self.table.setItem(row, 0, QTableWidgetItem(c.flight_name))
            self.table.setItem(row, 1, QTableWidgetItem(str(c.chosen_option)))
            self.table.setItem(row, 2, QTableWidgetItem(f"{c.fuel_kg:.0f}"))
            self.table.setItem(row, 3, QTableWidgetItem(str(c.contrail_cells)))
            self.table.setItem(row, 4, QTableWidgetItem(f"{c.disruption_flmin:.1f}"))

        self.status_label.setText(
            f"objective = {resp.objective:.2f}   "  # type: ignore[attr-defined]
            f"status = {resp.status}   "  # type: ignore[attr-defined]
            f"wall = {resp.wall_clock_s * 1000:.0f} ms   "  # type: ignore[attr-defined]
            f"conflicts = {resp.n_conflicts}   "  # type: ignore[attr-defined]
            f"options = {resp.n_options_total}"  # type: ignore[attr-defined]
        )

        # Redraw the chosen trajectories on top of the solved scenario.
        chosen = {c.flight_name: c.chosen_option for c in choices}
        if self._pending_cfg is not None:
            self._refresh_scene(self._pending_cfg, chosen=chosen)
        self.solve_btn.setEnabled(True)

    def _on_failed(self, message: str) -> None:
        self.status_label.setText(f"solve failed: {message}")
        self.solve_btn.setEnabled(True)


def main() -> None:
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
