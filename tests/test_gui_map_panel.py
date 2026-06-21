"""GUI map panel (tab 6): builds and renders for synthetic and ML worlds.

Headless: forces the offscreen Qt platform so it needs no display. Skips
entirely when the [gui] extra (PyQt6/pyqtgraph) isn't installed — e.g. the
core CI jobs — so it runs locally with the dashboard deps but never blocks the
lean gates. Run locally with:  pytest tests/test_gui_map_panel.py
"""

import os

import pytest

# Must be set before any Qt import so CI / headless boxes need no X server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from contrail_env import build_random_flights, default_european_world  # noqa: E402
from gui.app import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    # One QApplication per process; reuse if another test already made one.
    app = QApplication.instance() or QApplication([])
    yield app


def test_map_tab_is_registered(qapp):
    win = MainWindow()
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert any("Map" in t for t in titles), titles
    assert win.map_plot is not None


def test_render_map_synthetic_runs(qapp):
    # __init__ already built a synthetic world + rendered a preview; re-run both
    # the no-selection and a chosen-selection render to exercise both branches.
    win = MainWindow()
    win._render_map(None)
    chosen = {f.name: 0 for f in win._flights}
    win._render_map(chosen)


def test_render_map_ml_runs(qapp):
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
