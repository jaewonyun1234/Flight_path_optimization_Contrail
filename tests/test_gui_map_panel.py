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
from gui.app import (  # noqa: E402
    _MAP_ANIM_FRAMES,
    MainWindow,
    RouteLine,
    _geo_assets_script,
    build_map_figure,
)


def _route(name, lon, lat, t_s, chosen):
    return RouteLine(name, np.asarray(lon, float), np.asarray(lat, float),
                     np.asarray(t_s, float), chosen)


@pytest.fixture(scope="module")
def qapp():
    # One QApplication per process; reuse if another test already made one.
    app = QApplication.instance() or QApplication([])
    yield app


# --------------------------------------------------------------------------- #
# Pure figure builder — no Qt, no web view.                                    #
# --------------------------------------------------------------------------- #

def test_build_map_figure_animates_routes_with_frames():
    lon, lat = np.meshgrid(np.linspace(-2, 18, 20), np.linspace(43, 50, 15), indexing="ij")
    risk = np.abs(np.sin(lon) * np.cos(lat))
    t = np.array([0.0, 400.0, 800.0, 1200.0])
    routes = [
        _route("AB123", [0, 5, 8, 10], [45, 46, 46.5, 47], t, chosen=True),
        _route("CD456", [1, 3, 5, 6], [44, 45, 47, 48], t, chosen=False),
    ]
    fig = build_map_figure(source="synthetic", lon=lon, lat=lat, risk=risk, routes=routes)

    # SVG `geo` traces (no WebGL). trace 0 = risk overlay; then per route a
    # trail line + a head marker, so markers = risk + 2 heads, lines = 2 trails.
    assert all(tr.type == "scattergeo" for tr in fig.data)
    assert fig.data[0].mode == "markers"
    modes = [tr.mode for tr in fig.data]
    assert modes.count("lines") == 2
    assert modes.count("markers") == 3
    # Animation machinery: one frame per time step, a play/pause control + slider.
    assert len(fig.frames) == _MAP_ANIM_FRAMES
    assert fig.layout.updatemenus and fig.layout.sliders
    # Frames update only the 2N route traces, never the risk overlay (trace 0).
    assert list(fig.frames[0].traces) == [1, 2, 3, 4]
    # Self-contained page (what the web view loads); inlined plotly.js is large.
    html = fig.to_html(include_plotlyjs="inline", full_html=True)
    assert "<html" in html.lower() and len(html) > 2_000_000


def test_build_map_figure_static_when_not_animated():
    lon, lat = np.meshgrid(np.linspace(-2, 18, 10), np.linspace(43, 50, 8), indexing="ij")
    risk = np.abs(np.sin(lon))
    t = np.array([0.0, 600.0, 1200.0])
    routes = [_route("AB123", [0, 5, 10], [45, 46, 47], t, chosen=True)]
    fig = build_map_figure(source="synthetic", lon=lon, lat=lat, risk=risk,
                           routes=routes, animate=False)
    # No frames; the route is drawn in full (one trail line + one head marker).
    assert not fig.frames
    modes = [tr.mode for tr in fig.data]
    assert modes.count("lines") == 1
    assert modes.count("markers") == 2  # risk overlay + head marker


def test_build_map_figure_handles_empty_routes():
    lon, lat = np.meshgrid(np.linspace(0, 10, 8), np.linspace(44, 49, 6), indexing="ij")
    fig = build_map_figure(source="synthetic", lon=lon, lat=lat,
                           risk=np.zeros_like(lon), routes=[])
    # Only the (empty, all-clear-sky) risk overlay; no routes, no frames.
    assert [t.mode for t in fig.data] == ["markers"]
    assert not fig.frames


def test_geo_basemap_is_bundled_offline():
    # The country/coastline vectors must ship in-repo and inline into the page,
    # so the geo map renders with no CDN / network (locked-down corporate boxes).
    script = _geo_assets_script()
    assert script and "world_50m" in script and "PlotlyGeoAssets" in script


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


def test_render_map_uses_canonical_anchor(qapp):
    # The synthetic field carries no anchor of its own, so the map falls back to
    # the canonical European anchor.
    from contrail_env.geo import EUROPEAN_ANCHOR

    world = default_european_world(seed=1)
    flights = build_random_flights(n_flights=2, world=world, seed=1,
                                   corridor_frac=0.05, snapshot_window_s=(0.0, 300.0))
    win = MainWindow()
    win._world = world
    win._flights = flights
    win._render_map(None)
    assert win._active_anchor(world.issr) is EUROPEAN_ANCHOR
    assert os.path.exists(win._map_html_path)
