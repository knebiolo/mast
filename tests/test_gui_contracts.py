"""GUI contract tests for import and CJS action wiring.

These tests are skipped when Qt bindings are not available.
"""

from pathlib import Path
import importlib
import json

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


def test_refresh_data_viewer_reads_hdf_preview(app, tmp_path):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    db_path = tmp_path / "viewer_test.h5"
    source = pd.DataFrame(
        {
            "freq_code": ["F1", "F2"],
            "rec_id": ["R01", "R02"],
            "value": [1, 2],
        }
    )
    source.to_hdf(db_path, key="recaptures", format="table", mode="w")

    window.import_db_dir.setText(str(db_path))
    window.refresh_data_viewer_keys()
    window.viewer_limit_spin.setValue(10)
    window.viewer_offset_spin.setValue(0)
    window.viewer_key_combo.setCurrentText("/recaptures")

    window.refresh_data_viewer()

    assert window.viewer_table.rowCount() == 2
    assert window.viewer_table.columnCount() >= 3
    assert "loaded 2 row(s)" in window.viewer_status_label.text().lower()


def test_gui_session_round_trip(app, tmp_path):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    db_path = tmp_path / "session_test.h5"
    pd.DataFrame({"value": [1]}).to_hdf(db_path, key="raw_data", format="table", mode="w")

    window.import_db_dir.setText(str(db_path))
    session_path = window._update_session_state_path()
    assert session_path is not None

    window.project_dir_edit.setText(r"C:\example\project")
    window.db_name_edit.setText("demo_db")
    window.class_threshold.setValue(1.7)
    window.like_power.setChecked(False)
    window.viewer_where_edit.setText("rec_id == 'R01'")

    window.save_gui_session()
    assert session_path.exists()

    saved = json.loads(session_path.read_text(encoding="utf-8"))
    assert saved["widgets"]["db_name_edit"] == "demo_db"

    window.db_name_edit.setText("changed")
    window.class_threshold.setValue(2.5)
    window.like_power.setChecked(True)
    window.viewer_where_edit.setText("")

    window.load_gui_session()

    assert window.db_name_edit.text() == "demo_db"
    assert window.class_threshold.value() == pytest.approx(1.7)
    assert window.like_power.isChecked() is False
    assert window.viewer_where_edit.text() == "rec_id == 'R01'"


def test_goto_step_updates_viewer_key(app, tmp_path):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    db_path = tmp_path / "step_viewer.h5"
    pd.DataFrame({"freq_code": ["F1"], "rec_id": ["R01"]}).to_hdf(db_path, key="raw_data", format="table", mode="w")
    pd.DataFrame({"freq_code": ["F1"], "rec_id": ["R01"], "test": [1]}).to_hdf(db_path, key="classified", format="table", mode="a")

    window.import_db_dir.setText(str(db_path))
    window.refresh_data_viewer_keys()

    window.goto_step(1)
    assert window.viewer_key_combo.currentText() == "/raw_data"

    window.goto_step(3)
    assert window.viewer_key_combo.currentText() == "/classified"


def test_filter_viewer_to_current_receiver_uses_active_step(app, tmp_path):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    db_path = tmp_path / "receiver_filter.h5"
    pd.DataFrame(
        {
            "freq_code": ["F1", "F2"],
            "rec_id": ["REC001", "REC002"],
            "test": [1, 1],
        }
    ).to_hdf(db_path, key="classified", format="table", mode="w")

    window.import_db_dir.setText(str(db_path))
    window.refresh_data_viewer_keys()
    window.goto_step(3)
    window.class_rec_id.setText("REC001")
    window.viewer_limit_spin.setValue(10)

    window.filter_viewer_to_current_receiver()

    assert window.viewer_where_edit.text() == "rec_id == 'REC001'"
    assert window.viewer_table.rowCount() == 1
