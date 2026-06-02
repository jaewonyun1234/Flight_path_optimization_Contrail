"""
app.py — PyQt6 desktop client for the contrail CP-SAT solver service.

The GUI never solves anything itself: it sends a ScenarioConfig to the gRPC
server, live-plots the convergence curve as ZMQ progress arrives, draws the
ISSR field with the flight routes overlaid, and tabulates the chosen option
per flight. That client/server split is the whole point of this layer.

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
    bash scripts/gen_proto.sh
    python -m service.server      # in one terminal
    python gui/app.py             # in another
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pyqtgraph as pg
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
    QVBoxLayout,
    QWidget,
)

from contrail_env import build_random_flights, default_european_world, fl_to_m
from service.client import DEFAULT_SERVER_ADDRESS, SolverClient
from service.generated import solver_pb2
from service.progress import DEFAULT_SUB_ADDRESS, subscribe

# Fixed scenario knobs not exposed as controls (sensible defaults per the brief).
CORRIDOR_FRAC = 0.05
SNAPSHOT_WINDOW_S = 300.0
# Head start (seconds) for the SUB socket to connect before the solve emits.
_SUB_HEAD_START_S = 0.25


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
        self._route_items: list[pg.PlotDataItem] = []

        self._build_ui()

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        # --- Controls -------------------------------------------------------
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(42)

        self.flights_spin = QSpinBox()
        self.flights_spin.setRange(1, 12)
        self.flights_spin.setValue(3)

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

        form = QFormLayout()
        form.addRow("Seed", self.seed_spin)
        form.addRow("Flights", self.flights_spin)
        form.addRow("ISSR blobs", self.blobs_spin)
        form.addRow("Contrail weight (beta)", self.beta_spin)
        form.addRow("Time limit (s)", self.time_spin)
        form.addRow("Map flight level", self.fl_combo)
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

        # --- ISSR map + routes ---------------------------------------------
        self.map_widget = pg.PlotWidget(title="ISSR field + flight routes")
        self.map_widget.setLabel("bottom", "x (km)")
        self.map_widget.setLabel("left", "y (km)")
        self.img = pg.ImageItem()
        try:
            self.img.setColorMap(pg.colormap.get("viridis"))
        except Exception:
            pass  # colormap is cosmetic; never block the UI on it
        self.map_widget.addItem(self.img)

        right = QVBoxLayout()
        right.addWidget(self.plot, stretch=1)
        right.addWidget(self.map_widget, stretch=1)
        right_widget = QWidget()
        right_widget.setLayout(right)

        root = QHBoxLayout()
        root.addWidget(left_widget, stretch=0)
        root.addWidget(right_widget, stretch=1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        # Draw the initial map for the default scenario.
        self._render_map()

    # -------------------------------------------------------------- helpers ---
    def _selected_fl(self) -> int:
        return int(self.fl_combo.currentText().replace("FL", ""))

    def _render_map(self) -> None:
        """Render the ISSR heatmap at the chosen FL with flight routes on top."""
        seed = self.seed_spin.value()
        n_blobs = self.blobs_spin.value()
        n_flights = self.flights_spin.value()

        world = default_european_world(seed=seed, n_issr_blobs=n_blobs)
        flights = build_random_flights(
            n_flights=n_flights,
            world=world,
            seed=seed,
            corridor_frac=CORRIDOR_FRAC,
            snapshot_window_s=(0.0, SNAPSHOT_WINDOW_S),
        )

        g = world.grid
        xx, yy = g.meshgrid_xy()
        zz = np.full_like(xx, fl_to_m(self._selected_fl()))
        field = world.issr.rhi_excess_grid(xx, yy, zz)

        self.img.setImage(field, autoLevels=True)
        self.img.setRect(
            g.x_min_km, g.y_min_km, g.x_max_km - g.x_min_km, g.y_max_km - g.y_min_km
        )

        for item in self._route_items:
            self.map_widget.removeItem(item)
        self._route_items = []
        for flight in flights:
            ox, oy = flight.origin_km
            dx, dy = flight.destination_km
            line = self.map_widget.plot(
                [ox, dx], [oy, dy], pen=pg.mkPen("r", width=2)
            )
            self._route_items.append(line)

    # ---------------------------------------------------------------- slots ---
    def _on_fl_changed(self, _text: str) -> None:
        # Re-render only the heatmap layer for the newly selected FL.
        self._render_map()

    def _on_solve(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        self._render_map()
        self._xs, self._ys = [], []
        self.curve.setData([], [])
        self.status_label.setText("solving…")
        self.solve_btn.setEnabled(False)

        topic = f"solve/{self.seed_spin.value()}-{int(time.time() * 1000)}"
        cfg = solver_pb2.ScenarioConfig(
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
        self.solve_btn.setEnabled(True)

    def _on_failed(self, message: str) -> None:
        self.status_label.setText(f"solve failed: {message}")
        self.solve_btn.setEnabled(True)


def main() -> None:
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1150, 740)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
