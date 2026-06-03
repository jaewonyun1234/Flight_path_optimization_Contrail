r"""
app.py — PyQt6 desktop client for the contrail CP-SAT solver service.

The GUI never solves anything itself: it sends a ScenarioConfig to the gRPC
server, live-plots the convergence curve as ZMQ progress arrives, draws the
problem (2D ISSR heatmap + interactive 3D trajectories), and tabulates the
chosen option per flight. That client/server split is the whole point.

The scenario is rebuilt locally from the same (seeded) config the server used,
via service.scenario — so what you SEE is exactly what the server SOLVED.

3D VIEW
=======
Per flight it draws BOTH paths so the optimizer's effect is visible:
    * baseline (the original plan)  — faint gray
    * chosen   (the optimized plan) — bright, in the flight's color
Options only change altitude, so the two share a ground track but sit at
different heights. Segments that pass through an ISSR (forming a contrail) turn
red on both. ISSR zones render as a glowing cloud (denser where RHi is higher).
A small arrowhead + an "Airplane N" label mark each flight's heading.

THREADING MODEL
===============
All network + solve work happens on a QThread worker (`SolveWorker`). The
worker talks to the UI ONLY through Qt signals; it never touches a widget
directly. Inside the worker, the blocking gRPC call runs on a small helper
thread so the worker can read the ZMQ progress stream at the same time.

RUN
===
    pip install -e .
    bash scripts/gen_proto.sh        # Windows: .\scripts\gen_proto.ps1
    python -m service.server         # in one terminal
    python gui/app.py                # in another
"""

from __future__ import annotations

import math
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

# Trajectory colors.
_BASELINE_RGBA = (0.55, 0.55, 0.6, 0.5)      # faint gray = original plan
_CONTRAIL_BASELINE_RGBA = (0.8, 0.2, 0.2, 0.8)
_CONTRAIL_CHOSEN_RGBA = (1.0, 0.15, 0.15, 1.0)  # bright red = forming a contrail


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
            "Drag to rotate, scroll to zoom.  Vertical = flight level (exaggerated).  "
            "Faint gray = original plan, bright = optimized.  "
            "Red = passing through a contrail zone.  Orange cloud = ISSR (denser = stronger)."
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

    # --- 3D helpers ----------------------------------------------------------
    def _add_gl(self, item) -> None:
        self.gl_view.addItem(item)
        self._gl_items.append(item)

    def _scene_xform(self):
        g = self._world.grid
        cx = (g.x_min_km + g.x_max_km) / 2.0
        cy = (g.y_min_km + g.y_max_km) / 2.0

        def to_scene(x_km: float, y_km: float, fl: float) -> tuple[float, float, float]:
            return (x_km - cx, y_km - cy, (fl - _FL_CENTER) * _VSCALE)

        return to_scene

    def _line_points(self, to_scene, waypoints) -> np.ndarray:
        return np.array(
            [to_scene(x, y, m_to_fl(z)) for (x, y, z, _t) in waypoints], dtype=float
        )

    def _vertex_colors(self, waypoints, normal_rgba, contrail_rgba) -> np.ndarray:
        issr = self._world.issr
        return np.array(
            [contrail_rgba if issr.is_inside(x, y, z) else normal_rgba
             for (x, y, z, _t) in waypoints],
            dtype=float,
        )

    def _arrowhead(self, head: np.ndarray, ox: float, oy: float,
                   dx: float, dy: float, size: float = 45.0) -> np.ndarray:
        theta = math.atan2(dy - oy, dx - ox)
        hx, hy, hz = float(head[0]), float(head[1]), float(head[2])
        pts = []
        for back in (math.radians(150.0), math.radians(-150.0)):
            a = theta + back
            pts.append((hx, hy, hz))
            pts.append((hx + size * math.cos(a), hy + size * math.sin(a), hz))
        return np.array(pts, dtype=float)

    def _add_label(self, pos, text: str, i: int) -> None:
        try:
            label = gl.GLTextItem(
                pos=np.array(pos, dtype=float), text=text, color=pg.intColor(i, hues=9)
            )
            self._add_gl(label)
        except Exception:
            pass  # GLTextItem may be unavailable on some pyqtgraph builds

    def _render_3d_inner(self, chosen: dict[str, int] | None) -> None:
        for item in self._gl_items:
            self.gl_view.removeItem(item)
        self._gl_items = []

        to_scene = self._scene_xform()
        g = self._world.grid

        # Reference grid at the FL340 plane.
        grid = gl.GLGridItem()
        grid.setSize(g.x_max_km - g.x_min_km, g.y_max_km - g.y_min_km)
        grid.setSpacing(100, 100)
        grid.translate(0, 0, (340 - _FL_CENTER) * _VSCALE)
        self._add_gl(grid)

        self._add_issr_cloud(to_scene)

        for i, flight in enumerate(self._flights):
            specs = enumerate_option_specs(flight)
            chosen_idx = chosen.get(flight.name, 0) if chosen else 0
            chosen_idx = max(0, min(chosen_idx, len(specs) - 1))

            # Original (baseline) path — only drawn once there's an optimized
            # path to compare it against.
            if chosen is not None and chosen_idx != 0:
                base_wps = waypoints_for(flight, specs[0].profile, sample_dt_s=30.0)
                base_pts = self._line_points(to_scene, base_wps)
                base_cols = self._vertex_colors(
                    base_wps, _BASELINE_RGBA, _CONTRAIL_BASELINE_RGBA
                )
                self._add_gl(gl.GLLinePlotItem(
                    pos=base_pts, color=base_cols, width=1.5, antialias=True
                ))

            # Optimized (chosen) path in the flight's color.
            wps = waypoints_for(flight, specs[chosen_idx].profile, sample_dt_s=30.0)
            pts = self._line_points(to_scene, wps)
            cols = self._vertex_colors(
                wps, _flight_color_gl(i), _CONTRAIL_CHOSEN_RGBA
            )
            self._add_gl(gl.GLLinePlotItem(pos=pts, color=cols, width=3.0, antialias=True))

            # Direction arrowhead + label at the destination end.
            ox, oy = flight.origin_km
            dx, dy = flight.destination_km
            arrow = self._arrowhead(pts[-1], ox, oy, dx, dy)
            self._add_gl(gl.GLLinePlotItem(
                pos=arrow, color=_flight_color_gl(i), width=2.5, mode="lines"
            ))
            label_pos = (float(pts[-1][0]), float(pts[-1][1]), float(pts[-1][2]) + 30.0)
            self._add_label(label_pos, f"Airplane {i + 1}", i)

        self.gl_view.setCameraPosition(
            distance=max(g.x_max_km - g.x_min_km, 1200), elevation=22, azimuth=-60
        )

    def _add_issr_cloud(self, to_scene) -> None:
        """ISSR zones as a glowing point cloud, denser/brighter where RHi is high."""
        issr = self._world.issr
        rng = np.random.default_rng(0)
        pts: list[tuple[float, float, float]] = []
        cols: list[tuple[float, float, float, float]] = []
        for blob in issr.blobs:
            n = 300
            xs = rng.normal(blob.cx_km, blob.sigma_h_km * 1.2, n)
            ys = rng.normal(blob.cy_km, blob.sigma_h_km * 1.2, n)
            zs = rng.normal(blob.cz_m, blob.sigma_v_m * 1.2, n)
            for x, y, zm in zip(xs, ys, zs, strict=True):
                rhi = issr.rhi_excess(float(x), float(y), float(zm))
                if rhi < 0.08:
                    continue  # skip faint wisps
                w = min(1.0, rhi)
                alpha = min(0.55, 0.1 + rhi * 0.5)
                fl = max(330, min(410, m_to_fl(zm)))
                pts.append(to_scene(float(x), float(y), fl))
                cols.append((1.0, 0.45 + 0.45 * w, 0.15 + 0.5 * w, alpha))
        if pts:
            cloud = gl.GLScatterPlotItem(
                pos=np.array(pts, dtype=float), color=np.array(cols, dtype=float), size=7.0
            )
            cloud.setGLOptions("additive")  # overlapping points glow = denser cores
            self._add_gl(cloud)

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

        # Redraw with the chosen options (original vs optimized) on the same scene.
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
