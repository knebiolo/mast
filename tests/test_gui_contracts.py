"""GUI contract tests for import and CJS action wiring.

These tests are skipped when Qt bindings are not available.
"""

from pathlib import Path
import importlib
import json

import matplotlib.pyplot as plt
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
    window.import_rec_id.clear()
    window.import_rec_id.addItems(["REC001"])
    window.import_rec_id.setCurrentText("REC001")
    window.import_rec_type.setCurrentText("pit")
    window.import_file_dir.setText(str(tmp_path))
    window.import_db_dir.setText(project.db)
    window.import_ant_map.setPlainText("{}")

    monkeypatch.setattr(window, "_run_action_async", lambda _label, fn, success_message=None: fn())

    window.run_import()

    assert project.called is not None
    assert project.called["rec_type"] == "pit"


def test_import_receiver_id_dropdown_uses_project_receivers(app):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    class DummyProject:
        def __init__(self):
            self.receivers = pd.DataFrame(
                {"rec_type": ["srx1200", "pit"]},
                index=["REC001", "REC002"],
            )

    window.project = DummyProject()
    window._refresh_import_receiver_ids()

    assert window.import_rec_id.count() == 2
    assert window.import_rec_id.itemText(0) == "REC001"
    assert window.import_rec_id.itemText(1) == "REC002"


def test_async_action_worker_emits_captured_output(app):
    gui = _load_gui_module_or_skip()
    output = []
    finished = []

    worker = gui.AsyncActionWorker(lambda: print("IMPORT SUMMARY LINE"))
    worker.output.connect(output.append)
    worker.finished.connect(lambda: finished.append(True))
    worker.run()

    assert finished == [True]
    assert any("IMPORT SUMMARY LINE" in line for line in output)


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


def test_browse_existing_hdf_autofills_setup_fields(monkeypatch, app, tmp_path):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    db_path = tmp_path / "existing_project.h5"
    pd.DataFrame({"x": [1]}).to_hdf(db_path, key="raw_data", format="table", mode="w")

    tag_csv = tmp_path / "tblMasterTag.csv"
    rec_csv = tmp_path / "tblMasterReceiver.csv"
    nodes_csv = tmp_path / "tblNodes.csv"
    tag_csv.write_text("freq_code,pulse_rate,tag_type,rel_date,rel_loc,cap_loc\n", encoding="utf-8")
    rec_csv.write_text("rec_id,rec_type,node\n", encoding="utf-8")
    nodes_csv.write_text("node\n", encoding="utf-8")

    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", lambda *_args, **_kwargs: (str(db_path), ""))

    window.project_dir_edit.setText("")
    window.db_name_combo.setCurrentText("")
    window.tag_csv_edit.setText("")
    window.receiver_csv_edit.setText("")
    window.nodes_csv_edit.setText("")

    window._browse_existing_hdf_for_setup()

    assert window.project_dir_edit.text() == str(tmp_path)
    assert window.db_name_combo.currentText() == "existing_project"
    assert window.import_db_dir.text() == str(db_path)
    assert window.tag_csv_edit.text() == str(tag_csv)
    assert window.receiver_csv_edit.text() == str(rec_csv)
    assert window.nodes_csv_edit.text() == str(nodes_csv)


def test_generate_parameter_plot_from_loaded_db_without_project(app, tmp_path):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    db_path = tmp_path / "trained_only.h5"
    trained = pd.DataFrame(
        {
            "rec_type": ["srx1200", "srx1200"],
            "detection": [1, 0],
            "cons_length": [4, 2],
            "hit_ratio": [0.8, 0.3],
            "power": [75, 30],
            "noise_ratio": [0.1, 0.5],
            "lag_diff": [12, 45],
        }
    )
    trained.to_hdf(db_path, key="trained", format="table", mode="w")

    window.project = None
    window.import_db_dir.setText(str(db_path))
    window.train_rec_type.clear()

    fig = window._generate_parameter_plot("Consecutive Hit Length")

    assert fig is not None
    assert window.train_rec_type.currentText() == "srx1200"


def test_generate_parameter_plot_uses_side_by_side_false_true_axes(app, tmp_path):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    db_path = tmp_path / "trained_side_by_side.h5"
    trained = pd.DataFrame(
        {
            "rec_type": ["srx1200", "srx1200", "srx1200", "srx1200"],
            "detection": [1, 0, 1, 0],
            "cons_length": [4, 2, 5, 3],
            "hit_ratio": [0.8, 0.3, 0.7, 0.4],
            "power": [75, 30, 80, 35],
            "noise_ratio": [0.1, 0.5, 0.2, 0.4],
            "lag_diff": [12, 45, 10, 40],
        }
    )
    trained.to_hdf(db_path, key="trained", format="table", mode="w")

    window.project = None
    window.import_db_dir.setText(str(db_path))
    window.train_rec_type.clear()
    window.train_rec_type.addItem("srx1200")
    window.train_rec_type.setCurrentText("srx1200")

    fig = window._generate_parameter_plot("Consecutive Hit Length")

    assert fig is not None
    assert len(fig.axes) == 2
    assert fig.axes[0].get_title() == "False Positive"
    assert fig.axes[1].get_title() == "True Detection"


def test_display_plot_twice_does_not_delete_placeholder_label(app):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2], [1, 2])
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2], [2, 1])

    window._display_plot(fig1)
    window._display_plot(fig2)

    pixmap = window.plot_image_label.pixmap()
    assert pixmap is not None
    assert pixmap.isNull() is False


def test_plot_controls_enabled_only_for_training_and_classification(app):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    window.goto_step(2)
    assert window.plot_selector_combo.isEnabled() is True
    assert window.plot_selector_label.text() == "Parameter:"
    assert window.plot_selector_combo.count() == 5

    window.goto_step(5)
    assert window.plot_selector_combo.isEnabled() is False
    assert window.plot_selector_label.text() == "Plot:"
    assert window.plot_selector_combo.currentText() == "No plots for this workflow step"


def test_plot_selection_reuses_cached_pixmap(monkeypatch, app):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))
    window.goto_step(2)
    window.train_rec_type.clear()
    window.train_rec_type.addItem("srx1200")
    window.train_rec_type.setCurrentText("srx1200")

    calls = {"n": 0}

    def _fake_generate(_label):
        calls["n"] += 1
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        return fig

    monkeypatch.setattr(window, "_generate_parameter_plot", _fake_generate)

    window.plot_selector_combo.setCurrentText("Hit Ratio Distribution")
    window._on_plot_selection_changed(window.plot_selector_combo.currentIndex())
    window._on_plot_selection_changed(window.plot_selector_combo.currentIndex())

    assert calls["n"] == 1


def test_preview_receiver_network_from_setup_files(monkeypatch, app, tmp_path):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    nodes_csv = tmp_path / "tblNodes.csv"
    receivers_csv = tmp_path / "tblMasterReceiver.csv"

    pd.DataFrame(
        {
            "node": ["N1", "N2"],
            "X": [0, 10],
            "Y": [0, 5],
            "parent": ["N1", ""],
            "child": ["N2", ""],
        }
    ).to_csv(nodes_csv, index=False)
    pd.DataFrame(
        {
            "rec_id": ["R01", "R02"],
            "rec_type": ["srx1200", "srx1200"],
            "node": ["N1", "N2"],
        }
    ).to_csv(receivers_csv, index=False)

    captured = {}

    def _capture_plot(fig):
        captured["title"] = fig.axes[0].get_title()
        captured["collection_count"] = len(fig.axes[0].collections)

    monkeypatch.setattr(window, "_display_plot", _capture_plot)
    window.nodes_csv_edit.setText(str(nodes_csv))
    window.receiver_csv_edit.setText(str(receivers_csv))
    window.project = None

    window.preview_receiver_network_from_setup()

    assert captured["title"] == "Receiver Network Graph"
    assert captured["collection_count"] >= 2


def test_return_to_setup_triggers_network_preview(monkeypatch, app, tmp_path):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    nodes_csv = tmp_path / "tblNodes.csv"
    pd.DataFrame(
        {
            "node": ["N1", "N2"],
            "X": [0, 10],
            "Y": [0, 5],
        }
    ).to_csv(nodes_csv, index=False)

    calls = {"n": 0}

    def _fake_preview():
        calls["n"] += 1

    monkeypatch.setattr(window, "preview_receiver_network_from_setup", _fake_preview)
    window.nodes_csv_edit.setText(str(nodes_csv))

    window.goto_step(2)
    window.goto_step(0)
    window.goto_step(3)
    window.goto_step(0)

    assert calls["n"] == 2


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
    window.db_name_combo.setCurrentText("demo_db")
    window.class_threshold.setValue(1.7)
    window.like_power.setChecked(False)
    window.viewer_where_edit.setText("rec_id == 'R01'")

    window.save_gui_session()
    assert session_path.exists()

    saved = json.loads(session_path.read_text(encoding="utf-8"))
    assert saved["widgets"]["db_name_combo"] == "demo_db"

    window.db_name_combo.setCurrentText("changed")
    window.class_threshold.setValue(2.5)
    window.like_power.setChecked(True)
    window.viewer_where_edit.setText("")

    window.load_gui_session()

    assert window.db_name_combo.currentText() == "demo_db"
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


def test_load_step_qc_summary_for_classified_data(app, tmp_path):
    gui = _load_gui_module_or_skip()
    window = gui.WorkflowWindow(Path("."))

    db_path = tmp_path / "qc_summary.h5"
    pd.DataFrame(
        {
            "freq_code": ["F1", "F2", "F3"],
            "rec_id": ["REC001", "REC001", "REC002"],
            "test": [1, 0, 1],
            "iter": [2, 2, 2],
            "posterior_T": [0.9, 0.2, 0.8],
            "time_stamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        }
    ).to_hdf(db_path, key="classified", format="table", mode="w")

    window.import_db_dir.setText(str(db_path))
    window.refresh_data_viewer_keys()
    window.goto_step(3)
    window.class_rec_id.setText("REC001")

    window.load_step_qc_summary()

    qc_text = window.viewer_qc_summary.toPlainText()
    assert "Stage QC: /classified" in qc_text
    assert "Rows: 2" in qc_text
    assert "Classified true detections: 1 (50.0%)" in qc_text
    assert "Latest iteration in view: 2" in qc_text
