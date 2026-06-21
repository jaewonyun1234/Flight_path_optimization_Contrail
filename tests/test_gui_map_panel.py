"""GUI Map tab (tab 6): the Plotly figure builder + the headless render path.

The map draws a MapLibre basemap with an ISSR density overlay and the flight
routes, shown in an embedded QtWebEngine view. QtWebEngine can't initialise
under the offscreen Qt platform, so headless runs build a placeholder instead
of the live view — but _render_map still writes the self-contained HTML, so the
figure/serialisation logic stays fully covered here.

Skips entirely when the [gui] extra (PyQt6 / pyqtgraph / plotly) isn't
installed, so it runs locally with the dashboard deps but never blocks the lean
CI gates. Run locally with:  pytest tests/test_gui_map_panel.py
"""

import os

import numpy as np
import pytest

# Must be set before any Qt import so CI / headless boxes need no display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("plotly")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from contrail_env import build_random_flights, default_european_world  # noqa: E402
from gui.app import MainWindow, RouteLine, build_map_figure  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    # One QApplication per process; reuse if another test already made one.
    app = QApplication.instance() or QApplication([])
    yield app


# --------------------------------------------------------------------------- #
# Pure figure builder — no Qt, no web view.                                    #
# --------------------------------------------------------------------------- #

def test_build_map_figure_has_density_and_route_traces():
    lon, lat = np.meshgrid(np.linspace(-2, 18, 20), np.linspace(43, 50, 15), indexing="ij")
    risk = np.abs(np.sin(lon) * np.cos(lat))
    routes = [
        RouteLine("AB123", np.array([0.0, 5.0, 10.0]), np.array([45.0, 46.0, 47.0]), chosen=True),
        RouteLine("CD456", np.array([1.0, 6.0]), np.array([44.0, 48.0]), chosen=False),
    ]
    fig = build_map_figure(source="synthetic", lon=lon, lat=lat, risk=risk, routes=routes)

    types = [t.type for t in fig.data]
    assert types.count("densitymap") == 1      # the ISSR overlay
    assert types.count("scattermap") == 2      # one trace per route
    # The figure must serialise to a self-contained page (this is what the
    # web view loads); inlined plotly.js makes it well over the setHtml limit.
    html = fig.to_html(include_plotlyjs="inline", full_html=True)
    assert "<html" in html.lower() and len(html) > 2_000_000


def test_build_map_figure_handles_empty_routes():
    lon, lat = np.meshgrid(np.linspace(0, 10, 8), np.linspace(44, 49, 6), indexing="ij")
    fig = build_map_figure(source="synthetic", lon=lon, lat=lat,
                           risk=np.zeros_like(lon), routes=[])
    assert [t.type for t in fig.data] == ["densitymap"]


# --------------------------------------------------------------------------- #
# Window integration — tab registration + the headless render/write path.      #
# --------------------------------------------------------------------------- #

def test_map_tab_is_registered(qapp):
    win = MainWindow()
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert any("Map" in t for t in titles), titles
    # Headless: no live web view, but the backing HTML path is always set up.
    assert win._map_html_path


def test_render_map_synthetic_writes_html(qapp):
    # __init__ already built a synthetic world + rendered a preview; re-run the
    # no-selection and a chosen-selection render to exercise both branches.
    win = MainWindow()
    win._render_map(None)
    chosen = {f.name: 0 for f in win._flights}
    win._render_map(chosen)
    assert os.path.exists(win._map_html_path)
    assert os.path.getsize(win._map_html_path) > 2_000_000


def test_render_map_ml_writes_html(qapp):
    # The ML field needs the [ml] extra (scipy interpolation + the model).
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    import warnings

    from contrail_ml.config import MLConfig
    from contrail_ml.issr_field import MLIssrField

    cfg = MLConfig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world = default_european_world(
            seed=1,
            issr_source="ml",
            issr_kwargs=dict(config=cfg, met_source="synthetic",
                             allow_synthetic=True, grid_res_deg=3.0, seed=1),
        )
    assert isinstance(world.issr, MLIssrField)
    flights = build_random_flights(n_flights=2, world=world, seed=1,
                                   corridor_frac=0.05, snapshot_window_s=(0.0, 300.0))

    win = MainWindow()
    win._world = world
    win._flights = flights
    # Renders the ML risk field + routes without raising; the active anchor
    # comes from the field itself (MLIssrField.anchor), not the synthetic default.
    win._render_map(None)
    assert win._active_anchor(world.issr) is world.issr.anchor
    assert os.path.exists(win._map_html_path)
