"""GUI contract tests for import and CJS action wiring.

These tests are skipped when Qt bindings are not available.
"""

from pathlib import Path
import importlib

import pandas as pd
import pytest


pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # type: ignore


def _load_gui_module_or_skip():
    try:
        return importlib.import_module("pymast.gui_launcher")
    except RuntimeError as exc:
        if "Qt is required" in str(exc):
            pytest.skip("Qt bindings are not available for GUI contract tests.")
        raise


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_run_import_normalizes_receiver_type(monkeypatch, app, tmp_path):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    class DummyProject:
        def __init__(self):
            self.db = str(tmp_path / "dummy.h5")
            self.called = None

        def telem_data_import(self, **kwargs):
            self.called = kwargs

    project = DummyProject()
    window.project = project
    window.import_rec_id.setText("REC001")
    window.import_rec_type.setCurrentText("pit")
    window.import_file_dir.setText(str(tmp_path))
    window.import_db_dir.setText(project.db)
    window.import_ant_map.setPlainText("{}")

    monkeypatch.setattr(window, "_run_action_async", lambda _label, fn, success_message=None: fn())

    window.run_import()

    assert project.called is not None
    assert project.called["rec_type"] == "pit"


def test_run_cjs_writes_csv_and_inp(monkeypatch, app, tmp_path):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    class DummyProject:
        output_dir = str(tmp_path)

    class DummyCJS:
        def __init__(self):
            self.inp = "/* FISH1 */  10101     1;"
            self.cross = pd.DataFrame({"R00": [1], "R01": [0]}, index=["FISH1"])

        def input_file(self, _model_name, _output_ws):
            return None

    monkeypatch.setattr(window, "_run_action_async", lambda _label, fn, success_message=None: fn())
    monkeypatch.setattr(gui.formatter, "cjs_data_prep", lambda **_kwargs: DummyCJS())

    window.project = DummyProject()
    window.cjs_output_ws.setText(str(tmp_path))
    window.cjs_model_name.setText("gui_contract")
    window.cjs_map.setPlainText("{'R01': 'R01'}")

    window.run_cjs()

    csv_path = tmp_path / "gui_contract.csv"
    inp_path = tmp_path / "gui_contract.inp"

    assert csv_path.exists()
    assert inp_path.exists()
    assert "FISH1" in inp_path.read_text(encoding="utf-8")


def test_cancel_active_action_sets_cancel_flag(app):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    class DummyThread:
        def __init__(self):
            self.interrupted = False

        def isRunning(self):
            return True

        def requestInterruption(self):
            self.interrupted = True

    thread = DummyThread()
    window._active_thread = thread
    window._active_action_label = "Run Bout Detection"

    window.cancel_active_action()

    assert window._cancel_requested is True
    assert thread.interrupted is True


def test_check_cancel_requested_raises(app):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))
    window._cancel_requested = True

    with pytest.raises(gui.ActionCancelled):
        window._check_cancel_requested()
