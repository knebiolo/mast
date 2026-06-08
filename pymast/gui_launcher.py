"""PyMAST full GUI workflow runner (no script execution)."""

from __future__ import annotations

import ast
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from pymast import formatter
from pymast.overlap_removal import bout, overlap_reduction
from pymast.radio_project import radio_project

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QPushButton,
            QScrollArea,
            QSpinBox,
            QStackedWidget,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError("Qt is required. Install PySide6 or PyQt5.") from exc


STEP_TITLES = {
    0: "Project Setup",
    1: "Import Data",
    2: "Train Classifier",
    3: "Classify Detections",
    4: "Calculate Bouts",
    5: "Reduce Overlap",
    6: "Create Recaptures",
    7: "Time-to-Event Model",
    8: "CJS Model",
}

STEP_HELP = {
    0: (
        "Project Setup\n\n"
        "Purpose:\n"
        "- Load core metadata and initialize/reload the project database.\n\n"
        "Required:\n"
        "- project_dir: folder where Output/, Data/, and .h5 live\n"
        "- db_name: base database name (without .h5)\n"
        "- tblMasterTag.csv, tblMasterReceiver.csv, tblNodes.csv\n\n"
        "Typical workflow:\n"
        "1. Select project folder and CSV setup files\n"
        "2. Keep detection_count=5 and duration=1 unless your study requires otherwise\n"
        "3. Click Initialize / Reload Project"
    ),
    1: (
        "Step 01 - Import Data\n\n"
        "Purpose:\n"
        "- Import raw receiver files into the project database for one receiver at a time.\n\n"
        "Key inputs:\n"
        "- rec_id and rec_type must match your receiver metadata\n"
        "- file_dir should point to the raw files for that receiver\n"
        "- ant_to_rec_dict should be a valid Python dict literal\n\n"
        "Example antenna map:\n"
        "{'A0': 'REC001'}\n\n"
        "Mini workflow:\n"
        "1. Pick receiver\n"
        "2. Set raw data folder\n"
        "3. Run Import"
    ),
    2: (
        "Step 02 - Train Classifier\n\n"
        "Purpose:\n"
        "- Train Naive Bayes features for selected fish at a receiver.\n\n"
        "Typical workflow:\n"
        "1. Set rec_id\n"
        "2. Use Train All Fish for normal runs\n"
        "3. Optionally run training summary"
    ),
    3: (
        "Step 03 - Classify Detections\n\n"
        "Purpose:\n"
        "- Score detections and label likely true detections vs noise.\n\n"
        "Tips:\n"
        "- threshold_ratio=1.0 is balanced\n"
        "- Select at least one likelihood predictor\n"
        "- Keep default predictor set unless you have model-specific reasons"
    ),
    4: (
        "Step 04 - Calculate Bouts\n\n"
        "Purpose:\n"
        "- Cluster detections into temporal bouts (presence periods).\n\n"
        "Tips:\n"
        "- eps_multiplier=5 is a common default\n"
        "- Run all receivers for full pipeline consistency\n"
        "- Enable visualization when tuning parameters"
    ),
    5: (
        "Step 05 - Reduce Overlap\n\n"
        "Purpose:\n"
        "- Resolve overlapping detections across receivers.\n\n"
        "Recommended start:\n"
        "- method='posterior'\n"
        "- p_value_threshold=0.05\n"
        "- effect_size_threshold=0.3\n"
        "- min_detections=1\n\n"
        "Use Nested Doll only when you have explicit parent-child receiver geometry."
    ),
    6: (
        "Step 06 - Create Recaptures\n\n"
        "Purpose:\n"
        "- Build the recaptures table used by downstream movement models.\n\n"
        "Typical workflow:\n"
        "1. Leave Export CSV enabled\n"
        "2. Enable PIT Study only for PIT workflows\n"
        "3. Click Create Recaptures Table"
    ),
    7: (
        "Step 07 - Time-to-Event\n\n"
        "Purpose:\n"
        "- Build multi-state movement/survival modeling tables.\n\n"
        "Key inputs:\n"
        "- receiver_to_state must be a dict literal, e.g. {'R01': 1, 'R02': 2}\n"
        "- adjacency_filter should be list of illegal transitions, e.g. [(9,1), (8,2)]\n\n"
        "Mini workflow:\n"
        "1. Set mapping\n"
        "2. Set filters if needed\n"
        "3. Run TTE Data Prep"
    ),
    8: (
        "Step 08 - CJS Model\n\n"
        "Purpose:\n"
        "- Export CJS encounter histories for MARK/R workflows.\n\n"
        "Key input:\n"
        "- receiver_to_recap dict literal, e.g. {'R01': 'R01', 'R02': 'R01'}\n\n"
        "Mini workflow:\n"
        "1. Configure receiver->occasion mapping\n"
        "2. Set model name and output folder\n"
        "3. Run CJS Export"
    ),
}


class WorkflowWindow(QMainWindow):
    def __init__(self, repo_root: Path) -> None:
        super().__init__()
        self.repo_root = repo_root
        self.logo_path = repo_root / "pymast_logo.png"

        self.project: Optional[radio_project] = None
        self.tte_obj = None
        self.cjs_obj = None

        self.setWindowTitle("PyMAST Full GUI Workflow")
        self.resize(1300, 900)

        root = QWidget()
        root_layout = QVBoxLayout(root)

        self.stack = QStackedWidget()
        root_layout.addWidget(self.stack, stretch=4)

        log_label = QLabel("Workflow Log")
        log_label.setStyleSheet("font-weight: 600;")
        root_layout.addWidget(log_label)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Workflow output and errors will appear here.")
        root_layout.addWidget(self.log_output, stretch=2)

        self.setCentralWidget(root)

        self.home_page = self._build_home_page()
        self.stack.addWidget(self.home_page)

        self.step_pages: Dict[int, QWidget] = {}
        for step in range(0, 9):
            page = self._build_step_page(step)
            self.step_pages[step] = page
            self.stack.addWidget(page)

    def _build_home_page(self) -> QWidget:
        page = QWidget()
        page_layout = QHBoxLayout(page)

        # Left pane: Logo
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(20, 20, 20, 20)

        if self.logo_path.exists():
            logo_label = QLabel()
            logo_label.setPixmap(self._full_logo_pixmap(width=320))
            logo_label.setAlignment(Qt.AlignCenter)
            left_layout.addWidget(logo_label)

        title = QLabel("PyMAST")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        left_layout.addWidget(title)

        subtitle = QLabel("End-to-End Workflow")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 12px; color: #666;")
        left_layout.addWidget(subtitle)

        left_layout.addStretch(1)
        page_layout.addWidget(left_pane, stretch=1)

        # Right pane: Number-pad style button grid
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(20, 20, 20, 20)

        grid_title = QLabel("Workflow Steps")
        grid_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        right_layout.addWidget(grid_title)

        # Build 3x3 grid (Project Setup + 8 steps)
        grid = QWidget()
        grid_layout = self._make_grid_layout()

        setup_btn = QPushButton("Setup")
        setup_btn.setMinimumHeight(60)
        setup_btn.setMinimumWidth(80)
        setup_btn.setStyleSheet("font-size: 11px; font-weight: 600;")
        setup_btn.clicked.connect(lambda checked=False: self.goto_step(0))
        grid_layout.addWidget(setup_btn, 0, 0)

        for i, step in enumerate(range(1, 9), start=1):
            btn = QPushButton(f"{step}")
            btn.setMinimumHeight(60)
            btn.setMinimumWidth(80)
            btn.setStyleSheet("font-size: 13px; font-weight: 700;")
            btn.setToolTip(STEP_TITLES[step])
            btn.clicked.connect(lambda checked=False, s=step: self.goto_step(s))
            row = (i) // 3
            col = (i) % 3
            grid_layout.addWidget(btn, row, col)

        grid.setLayout(grid_layout)
        right_layout.addWidget(grid)

        info_text = QLabel("Click a step number to configure and run workflow stages.")
        info_text.setStyleSheet("font-size: 10px; color: #666;")
        info_text.setWordWrap(True)
        right_layout.addWidget(info_text)

        right_layout.addStretch(1)
        page_layout.addWidget(right_pane, stretch=1)

        return page

    def _make_grid_layout(self):
        """Helper to create a 3x3 grid layout."""
        return QGridLayout()

    def _build_step_page(self, step: int) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        if self.logo_path.exists():
            logo_label = QLabel()
            logo_label.setPixmap(self._icon_logo_pixmap(width=220))
            logo_label.setAlignment(Qt.AlignCenter)
            outer.addWidget(logo_label)

        title_row = QHBoxLayout()
        title_row.addStretch(1)

        if step == 0:
            title_text = "Project Setup"
        else:
            title_text = f"Step {step:02d}: {STEP_TITLES[step]}"

        title = QLabel(title_text)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        title_row.addWidget(title)

        help_btn = QPushButton("?")
        help_btn.setFixedWidth(30)
        help_btn.setToolTip("Open detailed help for this section")
        help_btn.clicked.connect(lambda checked=False, s=step: self.show_step_help(s))
        title_row.addWidget(help_btn)
        title_row.addStretch(1)

        outer.addLayout(title_row)

        content = QWidget()
        content_layout = QVBoxLayout(content)

        if step == 0:
            self._build_step0(content_layout)
        elif step == 1:
            self._build_step1(content_layout)
        elif step == 2:
            self._build_step2(content_layout)
        elif step == 3:
            self._build_step3(content_layout)
        elif step == 4:
            self._build_step4(content_layout)
        elif step == 5:
            self._build_step5(content_layout)
        elif step == 6:
            self._build_step6(content_layout)
        elif step == 7:
            self._build_step7(content_layout)
        elif step == 8:
            self._build_step8(content_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        nav = QHBoxLayout()
        home_btn = QPushButton("Home")
        home_btn.clicked.connect(self.goto_home)
        prev_btn = QPushButton("Previous")
        prev_btn.clicked.connect(lambda: self.goto_step(max(0, step - 1)))
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(lambda: self.goto_step(min(8, step + 1)))
        nav.addWidget(home_btn)
        nav.addWidget(prev_btn)
        nav.addStretch(1)
        nav.addWidget(next_btn)
        outer.addLayout(nav)

        return page

    def _build_step0(self, parent_layout: QVBoxLayout) -> None:
        g_project = QGroupBox("Project Setup")
        f_project = QFormLayout(g_project)

        self.project_dir_edit, project_dir_row = self._line_with_browse(dir_mode=True)
        self._tip(self.project_dir_edit, "Root project folder. Should contain setup CSV files and will hold Data/, Output/, and the HDF5 database.")
        self.db_name_edit = QLineEdit("my_telemetry_study")
        self._tip(self.db_name_edit, "Database base name (no .h5 needed). Example: my_telemetry_study")
        self.det_count_spin = QSpinBox()
        self.det_count_spin.setRange(1, 100)
        self.det_count_spin.setValue(5)
        self._tip(self.det_count_spin, "Detection history window size used in predictor calculations. Typical range: 3-7.")
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.01, 1000.0)
        self.duration_spin.setDecimals(3)
        self.duration_spin.setValue(1.0)
        self._tip(self.duration_spin, "Noise window duration used by predictor calculations. Typical range: 1-5.")

        self.tag_csv_edit, tag_row = self._line_with_browse(file_mode=True)
        self.receiver_csv_edit, rec_row = self._line_with_browse(file_mode=True)
        self.nodes_csv_edit, nodes_row = self._line_with_browse(file_mode=True)
        self._tip(self.tag_csv_edit, "Path to tblMasterTag.csv with tag metadata (freq_code, pulse_rate, tag_type, rel_date, etc.).")
        self._tip(self.receiver_csv_edit, "Path to tblMasterReceiver.csv with receiver metadata (rec_id, rec_type, node, etc.).")
        self._tip(self.nodes_csv_edit, "Path to tblNodes.csv with spatial node coordinates and node IDs.")

        f_project.addRow("Project Directory", project_dir_row)
        f_project.addRow("Database Name", self.db_name_edit)
        f_project.addRow("Detection Count", self.det_count_spin)
        f_project.addRow("Duration", self.duration_spin)
        f_project.addRow("Tag Metadata CSV", tag_row)
        f_project.addRow("Receiver Metadata CSV", rec_row)
        f_project.addRow("Nodes Metadata CSV", nodes_row)

        init_btn = QPushButton("Initialize / Reload Project")
        init_btn.clicked.connect(self.initialize_project_from_form)
        self._tip(init_btn, "Loads metadata CSVs and initializes/reloads the project database object.")
        f_project.addRow(init_btn)

        parent_layout.addWidget(g_project)

    def _build_step1(self, parent_layout: QVBoxLayout) -> None:
        g_import = QGroupBox("Data Import Parameters")
        f_import = QFormLayout(g_import)

        self.import_rec_id = QLineEdit("REC001")
        self._tip(self.import_rec_id, "Receiver ID to import. Must exist in your receiver metadata table.")
        self.import_rec_type = QComboBox()
        self.import_rec_type.addItems([
            "srx1200", "srx800", "srx600", "orion", "ares", "VR2", "vr2", "PIT", "PIT_Multiple"
        ])
        self.import_file_dir, file_dir_row = self._line_with_browse(dir_mode=True)
        self._tip(self.import_file_dir, "Directory containing raw receiver files to import for this receiver.")
        self.import_db_dir = QLineEdit()
        self._tip(self.import_db_dir, "Target .h5 database path. Usually auto-filled after Project Setup.")
        self.import_scan_time = QDoubleSpinBox()
        self.import_scan_time.setRange(0.01, 1200.0)
        self.import_scan_time.setValue(2.5)
        self._tip(self.import_scan_time, "Scan duration per channel in seconds for multi-channel receiver types.")
        self.import_channels = QSpinBox()
        self.import_channels.setRange(1, 128)
        self.import_channels.setValue(1)
        self._tip(self.import_channels, "Number of channels configured on the receiver.")
        self.import_ant_map = QPlainTextEdit("{'A0': 'REC001'}")
        self.import_ant_map.setFixedHeight(70)
        self._tip(self.import_ant_map, "Python dict literal mapping antenna IDs to receiver IDs. Example: {'A0': 'REC001'}")
        self.import_ka_format = QCheckBox("Use KA Format")
        self._tip(self.import_ka_format, "Enable Kleinschmidt Associates-specific formatting rules when supported.")

        f_import.addRow("Receiver ID", self.import_rec_id)
        f_import.addRow("Receiver Type", self.import_rec_type)
        f_import.addRow("Raw Data Directory", file_dir_row)
        f_import.addRow("Database Path", self.import_db_dir)
        f_import.addRow("Scan Time (sec)", self.import_scan_time)
        f_import.addRow("Channels", self.import_channels)
        f_import.addRow("Antenna->Receiver Dict", self.import_ant_map)
        f_import.addRow("", self.import_ka_format)

        run_import_btn = QPushButton("Run Import")
        run_import_btn.clicked.connect(self.run_import)
        self._tip(run_import_btn, "Import raw files into /raw_data for this receiver.")
        undo_import_btn = QPushButton("Undo Import")
        undo_import_btn.clicked.connect(self.undo_import)
        self._tip(undo_import_btn, "Remove imported rows for the current receiver ID.")

        row = QHBoxLayout()
        row.addWidget(run_import_btn)
        row.addWidget(undo_import_btn)
        f_import.addRow(row)

        parent_layout.addWidget(g_import)

    def _build_step2(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Training Parameters")
        form = QFormLayout(group)

        self.train_rec_id = QLineEdit("REC001")
        self._tip(self.train_rec_id, "Receiver ID to train against.")
        self.train_rec_type = QLineEdit("srx1200")
        self._tip(self.train_rec_type, "Receiver type label used for training summary output.")
        self.train_all_fish = QCheckBox("Train All Fish Detected At Receiver")
        self.train_all_fish.setChecked(True)
        self._tip(self.train_all_fish, "If checked, auto-fetch fish IDs detected at this receiver.")
        self.train_fish_codes = QLineEdit()
        self.train_fish_codes.setPlaceholderText("Optional if not training all: freq1,freq2")
        self._tip(self.train_fish_codes, "Comma-separated fish freq_code list when not training all fish.")
        self.train_summary = QCheckBox("Run Training Summary")
        self.train_summary.setChecked(True)
        self._tip(self.train_summary, "Generate classifier training summary statistics/plots.")

        form.addRow("Receiver ID", self.train_rec_id)
        form.addRow("Receiver Type", self.train_rec_type)
        form.addRow("", self.train_all_fish)
        form.addRow("Fish Codes", self.train_fish_codes)
        form.addRow("", self.train_summary)

        run_btn = QPushButton("Run Training")
        run_btn.clicked.connect(self.run_training)
        self._tip(run_btn, "Run project.train() for selected fish IDs.")
        undo_btn = QPushButton("Undo Training")
        undo_btn.clicked.connect(self.undo_training)
        self._tip(undo_btn, "Remove training records for this receiver.")
        row = QHBoxLayout()
        row.addWidget(run_btn)
        row.addWidget(undo_btn)
        form.addRow(row)

        parent_layout.addWidget(group)

    def _build_step3(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Classification Parameters")
        form = QFormLayout(group)

        self.class_rec_id = QLineEdit("REC001")
        self._tip(self.class_rec_id, "Receiver ID to classify.")
        self.class_rec_type = QLineEdit("srx1200")
        self._tip(self.class_rec_type, "Receiver type context for classification.")
        self.class_threshold = QDoubleSpinBox()
        self.class_threshold.setRange(0.01, 100.0)
        self.class_threshold.setValue(1.0)
        self._tip(self.class_threshold, "Decision threshold ratio. 1.0=balanced MAP, >1 stricter, <1 more permissive.")

        self.like_hit = QCheckBox("hit_ratio")
        self.like_cons = QCheckBox("cons_length")
        self.like_noise = QCheckBox("noise_ratio")
        self.like_power = QCheckBox("power")
        self.like_lag = QCheckBox("lag_diff")
        for cb in [self.like_hit, self.like_cons, self.like_noise, self.like_power, self.like_lag]:
            cb.setChecked(True)

        like_row = QWidget()
        like_layout = QHBoxLayout(like_row)
        like_layout.setContentsMargins(0, 0, 0, 0)
        like_layout.addWidget(self.like_hit)
        like_layout.addWidget(self.like_cons)
        like_layout.addWidget(self.like_noise)
        like_layout.addWidget(self.like_power)
        like_layout.addWidget(self.like_lag)

        self.class_rec_list = QLineEdit()
        self.class_rec_list.setPlaceholderText("Optional list literal, e.g. ['REC001','REC002']")
        self._tip(self.class_rec_list, "Optional Python list literal of receiver IDs.")

        form.addRow("Receiver ID", self.class_rec_id)
        form.addRow("Receiver Type", self.class_rec_type)
        form.addRow("Threshold Ratio", self.class_threshold)
        form.addRow("Likelihood Predictors", like_row)
        form.addRow("Optional rec_list", self.class_rec_list)

        run_btn = QPushButton("Run Classification")
        run_btn.clicked.connect(self.run_classification)
        self._tip(run_btn, "Run classifier scoring and write classified results.")
        undo_btn = QPushButton("Undo Classification")
        undo_btn.clicked.connect(self.undo_classification)
        self._tip(undo_btn, "Remove classification output for the selected receiver.")
        row = QHBoxLayout()
        row.addWidget(run_btn)
        row.addWidget(undo_btn)
        form.addRow(row)

        parent_layout.addWidget(group)

    def _build_step4(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Bout Parameters")
        form = QFormLayout(group)

        self.bout_all_receivers = QCheckBox("Run For All Receivers")
        self.bout_all_receivers.setChecked(True)
        self._tip(self.bout_all_receivers, "Process bouts for every receiver in project metadata.")
        self.bout_rec_id = QLineEdit("REC001")
        self._tip(self.bout_rec_id, "Receiver ID if running bout detection for one receiver only.")
        self.bout_eps = QSpinBox()
        self.bout_eps.setRange(1, 100)
        self.bout_eps.setValue(5)
        self._tip(self.bout_eps, "DBSCAN epsilon multiplier. Actual epsilon = pulse_rate * eps_multiplier.")
        self.bout_lag = QSpinBox()
        self.bout_lag.setRange(1, 500)
        self.bout_lag.setValue(9)
        self._tip(self.bout_lag, "Legacy lag parameter retained for compatibility with existing workflows.")
        self.bout_visualize = QCheckBox("Visualize Bout Lengths")
        self._tip(self.bout_visualize, "Generate bout-length diagnostic plots after clustering.")

        form.addRow("", self.bout_all_receivers)
        form.addRow("Receiver ID (if not all)", self.bout_rec_id)
        form.addRow("eps_multiplier", self.bout_eps)
        form.addRow("lag_window", self.bout_lag)
        form.addRow("", self.bout_visualize)

        run_btn = QPushButton("Run Bout Detection")
        run_btn.clicked.connect(self.run_bouts)
        self._tip(run_btn, "Cluster detections into bouts and write /presence table.")
        undo_btn = QPushButton("Undo Bouts")
        undo_btn.clicked.connect(self.undo_bouts)
        self._tip(undo_btn, "Delete existing bout/presence output so you can rerun with new settings.")
        row = QHBoxLayout()
        row.addWidget(run_btn)
        row.addWidget(undo_btn)
        form.addRow(row)

        parent_layout.addWidget(group)

    def _build_step5(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Overlap Parameters")
        form = QFormLayout(group)

        self.overlap_use_all_pairs = QCheckBox("Auto-build edges as all receiver pairs")
        self.overlap_use_all_pairs.setChecked(True)
        self._tip(self.overlap_use_all_pairs, "If checked, compare every receiver pair; otherwise provide custom nodes/edges.")
        self.overlap_nodes = QLineEdit()
        self.overlap_nodes.setPlaceholderText("Optional list literal, e.g. ['R01','R02']")
        self._tip(self.overlap_nodes, "Optional Python list of receiver IDs for overlap analysis scope.")
        self.overlap_edges = QLineEdit()
        self.overlap_edges.setPlaceholderText("Optional list of tuples, e.g. [('R01','R02')]")
        self._tip(self.overlap_edges, "Optional Python list of directed receiver pairs as tuples.")

        self.overlap_method = QComboBox()
        self.overlap_method.addItems(["posterior", "power"])
        self._tip(self.overlap_method, "Comparison metric: posterior classifier confidence or signal power.")
        self.overlap_p = QDoubleSpinBox()
        self.overlap_p.setRange(0.0001, 1.0)
        self.overlap_p.setDecimals(4)
        self.overlap_p.setValue(0.05)
        self._tip(self.overlap_p, "Maximum p-value for significance in posterior-based overlap decisions.")
        self.overlap_effect = QDoubleSpinBox()
        self.overlap_effect.setRange(0.0, 5.0)
        self.overlap_effect.setValue(0.3)
        self._tip(self.overlap_effect, "Minimum Cohen's d effect size for posterior-based removal decisions.")
        self.overlap_power = QDoubleSpinBox()
        self.overlap_power.setRange(0.0, 5.0)
        self.overlap_power.setValue(0.2)
        self._tip(self.overlap_power, "Relative power-difference threshold when using method='power'.")
        self.overlap_min_det = QSpinBox()
        self.overlap_min_det.setRange(1, 1000)
        self.overlap_min_det.setValue(1)
        self._tip(self.overlap_min_det, "Minimum detections required in each bout for overlap comparison.")
        self.overlap_expand = QSpinBox()
        self.overlap_expand.setRange(0, 36000)
        self.overlap_expand.setValue(0)
        self._tip(self.overlap_expand, "Seconds to expand bout windows before overlap checks.")
        self.overlap_conf = QLineEdit()
        self.overlap_conf.setPlaceholderText("Optional confidence_threshold, e.g. 0.1")
        self._tip(self.overlap_conf, "Optional posterior mean-difference tiebreak threshold.")

        form.addRow("", self.overlap_use_all_pairs)
        form.addRow("Nodes", self.overlap_nodes)
        form.addRow("Edges", self.overlap_edges)
        form.addRow("Method", self.overlap_method)
        form.addRow("p_value_threshold", self.overlap_p)
        form.addRow("effect_size_threshold", self.overlap_effect)
        form.addRow("power_threshold", self.overlap_power)
        form.addRow("min_detections", self.overlap_min_det)
        form.addRow("bout_expansion", self.overlap_expand)
        form.addRow("confidence_threshold", self.overlap_conf)

        run_btn = QPushButton("Run Unsupervised Overlap")
        run_btn.clicked.connect(self.run_overlap_unsupervised)
        self._tip(run_btn, "Run overlap_reduction.unsupervised_removal with the configured parameters.")
        nested_btn = QPushButton("Run Nested Doll")
        nested_btn.clicked.connect(self.run_overlap_nested)
        self._tip(nested_btn, "Run hierarchical nested-doll overlap logic using your nodes/edges.")
        undo_btn = QPushButton("Undo Overlap")
        undo_btn.clicked.connect(self.undo_overlap)
        self._tip(undo_btn, "Clear overlap decisions from the database.")

        row = QHBoxLayout()
        row.addWidget(run_btn)
        row.addWidget(nested_btn)
        row.addWidget(undo_btn)
        form.addRow(row)

        parent_layout.addWidget(group)

    def _build_step6(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Recaptures Parameters")
        form = QFormLayout(group)

        self.recap_export = QCheckBox("Export CSV")
        self.recap_export.setChecked(True)
        self._tip(self.recap_export, "Write recaptures.csv to Output/ when recaptures table is built.")
        self.recap_pit_study = QCheckBox("PIT Study Mode")
        self._tip(self.recap_pit_study, "Enable PIT-specific behavior in recaptures table generation.")

        form.addRow("", self.recap_export)
        form.addRow("", self.recap_pit_study)

        run_btn = QPushButton("Create Recaptures Table")
        run_btn.clicked.connect(self.run_recaptures)
        self._tip(run_btn, "Build/refresh recaptures table from classified, bout, and overlap outputs.")
        undo_btn = QPushButton("Undo Recaptures")
        undo_btn.clicked.connect(self.undo_recaptures)
        self._tip(undo_btn, "Delete existing recaptures table entries.")
        row = QHBoxLayout()
        row.addWidget(run_btn)
        row.addWidget(undo_btn)
        form.addRow(row)

        parent_layout.addWidget(group)

    def _build_step7(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Time-to-Event Parameters")
        form = QFormLayout(group)

        self.tte_node_map = QPlainTextEdit("{'REC001': 1}")
        self.tte_node_map.setFixedHeight(100)
        self._tip(self.tte_node_map, "Python dict mapping receiver IDs to model states. Example: {'R01':1, 'R02':2}")

        self.tte_initial_release = QCheckBox("initial_state_release")
        self.tte_initial_release.setChecked(True)
        self._tip(self.tte_initial_release, "Include release as state 0 to model movement from release point.")
        self.tte_last_presence = QCheckBox("last_presence_time0")
        self._tip(self.tte_last_presence, "Set time zero to last presence at initial state rather than first detection.")
        self.tte_hit_ratio_filter = QCheckBox("hit_ratio_filter")
        self._tip(self.tte_hit_ratio_filter, "Filter low hit_ratio detections before TTE table creation.")

        self.tte_cap_loc = QLineEdit()
        self.tte_rel_loc = QLineEdit()
        self.tte_species = QLineEdit()
        self._tip(self.tte_cap_loc, "Optional filter: capture location string.")
        self._tip(self.tte_rel_loc, "Optional filter: release location string.")
        self._tip(self.tte_species, "Optional filter: species name/code.")
        self.tte_rel_date = QLineEdit()
        self.tte_rel_date.setPlaceholderText("Optional date string")
        self._tip(self.tte_rel_date, "Optional release-date lower bound. Example: 2026-01-01")
        self.tte_recap_date = QLineEdit()
        self.tte_recap_date.setPlaceholderText("Optional date string")
        self._tip(self.tte_recap_date, "Optional recapture-date lower bound. Example: 2026-01-01")

        self.tte_unknown_state = QLineEdit()
        self.tte_unknown_state.setPlaceholderText("Optional int")
        self._tip(self.tte_unknown_state, "Optional numeric state ID for unknown states.")
        self.tte_bucket_min = QSpinBox()
        self.tte_bucket_min.setRange(1, 1440)
        self.tte_bucket_min.setValue(15)
        self._tip(self.tte_bucket_min, "Time bucket length in minutes for TTE aggregation.")
        self.tte_adjacency = QLineEdit()
        self.tte_adjacency.setPlaceholderText("Optional list of tuples, e.g. [(9,1),(8,2)]")
        self._tip(self.tte_adjacency, "Optional list of illegal transitions as (from_state, to_state) tuples.")

        form.addRow("receiver_to_state", self.tte_node_map)
        form.addRow("", self.tte_initial_release)
        form.addRow("", self.tte_last_presence)
        form.addRow("", self.tte_hit_ratio_filter)
        form.addRow("cap_loc", self.tte_cap_loc)
        form.addRow("rel_loc", self.tte_rel_loc)
        form.addRow("species", self.tte_species)
        form.addRow("rel_date", self.tte_rel_date)
        form.addRow("recap_date", self.tte_recap_date)
        form.addRow("unknown_state", self.tte_unknown_state)
        form.addRow("bucket_length_min", self.tte_bucket_min)
        form.addRow("adjacency_filter", self.tte_adjacency)

        run_btn = QPushButton("Run TTE Data Prep")
        run_btn.clicked.connect(self.run_tte)
        self._tip(run_btn, "Create TTE object, run data_prep, and generate summary outputs.")
        form.addRow(run_btn)

        parent_layout.addWidget(group)

    def _build_step8(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("CJS Parameters")
        form = QFormLayout(group)

        self.cjs_map = QPlainTextEdit("{'REC001': 'R01'}")
        self.cjs_map.setFixedHeight(100)
        self._tip(self.cjs_map, "Python dict mapping receiver IDs to recapture occasions. Example: {'R01':'R01','R02':'R02'}")
        self.cjs_species = QLineEdit()
        self.cjs_rel_loc = QLineEdit()
        self.cjs_cap_loc = QLineEdit()
        self._tip(self.cjs_species, "Optional species filter for CJS input generation.")
        self._tip(self.cjs_rel_loc, "Optional release-location filter for CJS generation.")
        self._tip(self.cjs_cap_loc, "Optional capture-location filter for CJS generation.")
        self.cjs_initial_release = QCheckBox("initial_recap_release")
        self._tip(self.cjs_initial_release, "Start CJS histories from first recap event instead of explicit release row.")
        self.cjs_model_name = QLineEdit("my_study_cjs")
        self._tip(self.cjs_model_name, "Output file base name for CJS artifacts.")
        self.cjs_output_ws, output_row = self._line_with_browse(dir_mode=True)
        self._tip(self.cjs_output_ws, "Output directory for CJS .csv/.inp-style export files.")

        form.addRow("receiver_to_recap", self.cjs_map)
        form.addRow("species", self.cjs_species)
        form.addRow("rel_loc", self.cjs_rel_loc)
        form.addRow("cap_loc", self.cjs_cap_loc)
        form.addRow("", self.cjs_initial_release)
        form.addRow("model_name", self.cjs_model_name)
        form.addRow("output_ws", output_row)

        run_btn = QPushButton("Run CJS Export")
        run_btn.clicked.connect(self.run_cjs)
        self._tip(run_btn, "Build CJS encounter histories and write output files.")
        form.addRow(run_btn)

        parent_layout.addWidget(group)

    def _line_with_browse(self, file_mode: bool = False, dir_mode: bool = False) -> Tuple[QLineEdit, QWidget]:
        line = QLineEdit()
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line)
        btn = QPushButton("Browse")
        if file_mode:
            self._tip(btn, "Browse to a file path.")
        elif dir_mode:
            self._tip(btn, "Browse to a directory path.")

        if file_mode:
            btn.clicked.connect(lambda: self._browse_file(line))
        elif dir_mode:
            btn.clicked.connect(lambda: self._browse_dir(line))

        row.addWidget(btn)
        return line, row_widget

    def _browse_file(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select File", str(self.repo_root), "CSV Files (*.csv);;All Files (*.*)")
        if path:
            target.setText(path)

    def _browse_dir(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Directory", str(self.repo_root))
        if path:
            target.setText(path)

    def goto_home(self) -> None:
        self.stack.setCurrentWidget(self.home_page)

    def goto_step(self, step: int) -> None:
        page = self.step_pages.get(step)
        if page is not None:
            self.stack.setCurrentWidget(page)

    def log(self, message: str) -> None:
        self.log_output.appendPlainText(message)
        QApplication.processEvents()

    def _required_project(self) -> radio_project:
        if self.project is None:
            raise RuntimeError("Project is not initialized. Run Step 01 project initialization first.")
        return self.project

    def _parse_literal(self, raw: str, field_name: str, expected_type: Optional[type] = None, allow_empty: bool = False) -> Any:
        text = raw.strip()
        if not text:
            if allow_empty:
                return None
            raise ValueError(f"{field_name} is required.")
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid literal for {field_name}: {exc}") from exc
        if expected_type is not None and not isinstance(value, expected_type):
            raise ValueError(f"{field_name} must be {expected_type.__name__}.")
        return value

    def _none_if_empty(self, text: str) -> Optional[str]:
        value = text.strip()
        return value if value else None

    def show_step_help(self, step: int) -> None:
        help_text = STEP_HELP.get(step, "No help available for this section yet.")
        title = STEP_TITLES.get(step, "Section Help")
        QMessageBox.information(self, f"Help: {title}", help_text)

    def _tip(self, widget: QWidget, text: str) -> None:
        widget.setToolTip(text)
        try:
            widget.setStatusTip(text)
        except AttributeError:
            pass

    def _full_logo_pixmap(self, width: int = 560) -> QPixmap:
        pix = QPixmap(str(self.logo_path))
        return pix.scaledToWidth(width, Qt.SmoothTransformation)

    def _icon_logo_pixmap(self, width: int = 220) -> QPixmap:
        pix = QPixmap(str(self.logo_path))
        if pix.isNull():
            return pix

        # Crop tightly to fish + signal region only (minimal font pickup).
        x = int(pix.width() * 0.22)
        y = int(pix.height() * 0.20)
        w = int(pix.width() * 0.44)
        h = int(pix.height() * 0.30)
        icon = pix.copy(x, y, w, h)
        return icon.scaledToWidth(width, Qt.SmoothTransformation)

    def _run_action(self, label: str, fn) -> None:
        self.log(f"\n=== {label} ===")
        try:
            fn()
            self.log(f"✓ {label} complete")
        except Exception as exc:  # noqa: BLE001
            self.log(f"✗ {label} failed: {exc}")
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "PyMAST GUI Error", f"{label} failed:\n{exc}")

    def initialize_project_from_form(self) -> None:
        def _impl() -> None:
            project_dir = self.project_dir_edit.text().strip()
            if not project_dir:
                raise ValueError("Project directory is required.")

            db_name = self.db_name_edit.text().strip()
            if not db_name:
                raise ValueError("Database name is required.")
            if db_name.lower().endswith(".h5"):
                db_name = db_name[:-3]

            tag_path = self.tag_csv_edit.text().strip() or os.path.join(project_dir, "tblMasterTag.csv")
            rec_path = self.receiver_csv_edit.text().strip() or os.path.join(project_dir, "tblMasterReceiver.csv")
            nodes_path = self.nodes_csv_edit.text().strip() or os.path.join(project_dir, "tblNodes.csv")

            for path, label in [(tag_path, "Tag CSV"), (rec_path, "Receiver CSV"), (nodes_path, "Nodes CSV")]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{label} not found: {path}")

            tag_data = pd.read_csv(tag_path)
            receiver_data = pd.read_csv(rec_path)
            nodes_data = pd.read_csv(nodes_path)

            self.project = radio_project(
                project_dir=project_dir,
                db_name=db_name,
                detection_count=int(self.det_count_spin.value()),
                duration=float(self.duration_spin.value()),
                tag_data=tag_data,
                receiver_data=receiver_data,
                nodes_data=nodes_data,
            )
            self.project.non_interactive = True

            db_path = os.path.join(project_dir, f"{db_name}.h5")
            self.import_db_dir.setText(db_path)
            self.cjs_output_ws.setText(os.path.join(project_dir, "Output"))
            self.log(f"Project DB: {db_path}")
            self.log(f"Loaded tags={len(tag_data)}, receivers={len(receiver_data)}, nodes={len(nodes_data)}")

        self._run_action("Initialize / Reload Project", _impl)

    def run_import(self) -> None:
        def _impl() -> None:
            proj = self._required_project()
            ant_map = self._parse_literal(self.import_ant_map.toPlainText(), "Antenna map", dict, allow_empty=True)
            if ant_map is None:
                ant_map = None

            rec_id = self.import_rec_id.text().strip()
            rec_type = self.import_rec_type.currentText().strip()
            file_dir = self.import_file_dir.text().strip()
            db_dir = self.import_db_dir.text().strip() or proj.db

            proj.telem_data_import(
                rec_id=rec_id,
                rec_type=rec_type,
                file_dir=file_dir,
                db_dir=db_dir,
                scan_time=float(self.import_scan_time.value()),
                channels=int(self.import_channels.value()),
                ant_to_rec_dict=ant_map,
                ka_format=bool(self.import_ka_format.isChecked()),
            )
            self.log(f"Imported receiver {rec_id} from {file_dir}")

        self._run_action("Run Import", _impl)

    def undo_import(self) -> None:
        self._run_action("Undo Import", lambda: self._required_project().undo_import(self.import_rec_id.text().strip()))

    def run_training(self) -> None:
        def _impl() -> None:
            proj = self._required_project()
            rec_id = self.train_rec_id.text().strip()

            if self.train_all_fish.isChecked():
                fishes = proj.get_fish(rec_id=rec_id)
            else:
                fishes = [x.strip() for x in self.train_fish_codes.text().split(",") if x.strip()]

            if not fishes:
                raise ValueError("No fish found to train.")

            self.log(f"Training {len(fishes)} fish at {rec_id}...")
            for fish in fishes:
                proj.train(fish, rec_id)
                QApplication.processEvents()

            if self.train_summary.isChecked():
                proj.training_summary(self.train_rec_type.text().strip(), site=[rec_id])

        self._run_action("Run Training", _impl)

    def undo_training(self) -> None:
        self._run_action("Undo Training", lambda: self._required_project().undo_training(self.train_rec_id.text().strip()))

    def run_classification(self) -> None:
        def _impl() -> None:
            proj = self._required_project()
            likelihood = []
            if self.like_hit.isChecked():
                likelihood.append("hit_ratio")
            if self.like_cons.isChecked():
                likelihood.append("cons_length")
            if self.like_noise.isChecked():
                likelihood.append("noise_ratio")
            if self.like_power.isChecked():
                likelihood.append("power")
            if self.like_lag.isChecked():
                likelihood.append("lag_diff")

            if not likelihood:
                raise ValueError("At least one likelihood predictor must be selected.")

            rec_list = None
            if self.class_rec_list.text().strip():
                rec_list = self._parse_literal(self.class_rec_list.text(), "rec_list", list)

            proj.reclassify(
                project=proj,
                rec_id=self.class_rec_id.text().strip(),
                threshold_ratio=float(self.class_threshold.value()),
                likelihood_model=likelihood,
                rec_type=self.class_rec_type.text().strip() or None,
                rec_list=rec_list,
            )

        self._run_action("Run Classification", _impl)

    def undo_classification(self) -> None:
        self._run_action(
            "Undo Classification",
            lambda: self._required_project().undo_classification(self.class_rec_id.text().strip()),
        )

    def run_bouts(self) -> None:
        def _impl() -> None:
            proj = self._required_project()
            if self.bout_all_receivers.isChecked():
                rec_ids = list(proj.receivers.index)
            else:
                rec_id = self.bout_rec_id.text().strip()
                if not rec_id:
                    raise ValueError("Receiver ID is required when not running all receivers.")
                rec_ids = [rec_id]

            for rec_id in rec_ids:
                self.log(f"Running bout detection for {rec_id}...")
                b = bout(
                    radio_project=proj,
                    rec_id=rec_id,
                    eps_multiplier=int(self.bout_eps.value()),
                    lag_window=int(self.bout_lag.value()),
                )
                b.presence()
                if self.bout_visualize.isChecked():
                    b.visualize_bout_lengths()
                QApplication.processEvents()

        self._run_action("Run Bout Detection", _impl)

    def undo_bouts(self) -> None:
        def _impl() -> None:
            proj = self._required_project()
            if self.bout_all_receivers.isChecked():
                proj.undo_bouts()
            else:
                proj.undo_bouts(self.bout_rec_id.text().strip())

        self._run_action("Undo Bouts", _impl)

    def _resolve_nodes_edges(self) -> Tuple[List[str], List[Tuple[str, str]]]:
        proj = self._required_project()

        if self.overlap_nodes.text().strip():
            nodes = self._parse_literal(self.overlap_nodes.text(), "nodes", list)
        else:
            nodes = list(proj.receivers.index)

        if self.overlap_use_all_pairs.isChecked():
            edges = [(i, j) for i in nodes for j in nodes if i != j]
        else:
            edges = self._parse_literal(self.overlap_edges.text(), "edges", list)

        return nodes, edges

    def run_overlap_unsupervised(self) -> None:
        def _impl() -> None:
            proj = self._required_project()
            nodes, edges = self._resolve_nodes_edges()
            overlap_obj = overlap_reduction(nodes=nodes, edges=edges, radio_project=proj)

            conf_raw = self.overlap_conf.text().strip()
            confidence = float(conf_raw) if conf_raw else None

            overlap_obj.unsupervised_removal(
                method=self.overlap_method.currentText(),
                p_value_threshold=float(self.overlap_p.value()),
                effect_size_threshold=float(self.overlap_effect.value()),
                power_threshold=float(self.overlap_power.value()),
                min_detections=int(self.overlap_min_det.value()),
                bout_expansion=int(self.overlap_expand.value()),
                confidence_threshold=confidence,
            )

        self._run_action("Run Unsupervised Overlap", _impl)

    def run_overlap_nested(self) -> None:
        def _impl() -> None:
            proj = self._required_project()
            nodes, edges = self._resolve_nodes_edges()
            overlap_obj = overlap_reduction(nodes=nodes, edges=edges, radio_project=proj)
            overlap_obj.nested_doll()

        self._run_action("Run Nested Doll", _impl)

    def undo_overlap(self) -> None:
        self._run_action("Undo Overlap", lambda: self._required_project().undo_overlap())

    def run_recaptures(self) -> None:
        def _impl() -> None:
            proj = self._required_project()
            proj.make_recaptures_table(
                export=bool(self.recap_export.isChecked()),
                pit_study=bool(self.recap_pit_study.isChecked()),
            )

        self._run_action("Create Recaptures Table", _impl)

    def undo_recaptures(self) -> None:
        self._run_action("Undo Recaptures", lambda: self._required_project().undo_recaptures())

    def run_tte(self) -> None:
        def _impl() -> None:
            proj = self._required_project()
            node_to_state = self._parse_literal(self.tte_node_map.toPlainText(), "receiver_to_state", dict)

            adjacency = None
            if self.tte_adjacency.text().strip():
                adjacency = self._parse_literal(self.tte_adjacency.text(), "adjacency_filter", list)

            unknown_state = None
            if self.tte_unknown_state.text().strip():
                unknown_state = int(self.tte_unknown_state.text().strip())

            tte = formatter.time_to_event(
                receiver_to_state=node_to_state,
                project=proj,
                initial_state_release=bool(self.tte_initial_release.isChecked()),
                last_presence_time0=bool(self.tte_last_presence.isChecked()),
                hit_ratio_filter=bool(self.tte_hit_ratio_filter.isChecked()),
                cap_loc=self._none_if_empty(self.tte_cap_loc.text()),
                rel_loc=self._none_if_empty(self.tte_rel_loc.text()),
                species=self._none_if_empty(self.tte_species.text()),
                rel_date=self._none_if_empty(self.tte_rel_date.text()),
                recap_date=self._none_if_empty(self.tte_recap_date.text()),
            )
            tte.data_prep(
                project=proj,
                unknown_state=unknown_state,
                bucket_length_min=int(self.tte_bucket_min.value()),
                adjacency_filter=adjacency,
            )
            tte.summary()
            self.tte_obj = tte

        self._run_action("Run TTE Data Prep", _impl)

    def run_cjs(self) -> None:
        def _impl() -> None:
            proj = self._required_project()
            receiver_to_recap = self._parse_literal(self.cjs_map.toPlainText(), "receiver_to_recap", dict)

            output_ws = self.cjs_output_ws.text().strip() or proj.output_dir
            os.makedirs(output_ws, exist_ok=True)
            model_name = self.cjs_model_name.text().strip()
            if not model_name:
                raise ValueError("model_name is required.")

            cjs = formatter.cjs_data_prep(
                receiver_to_recap=receiver_to_recap,
                project=proj,
                species=self._none_if_empty(self.cjs_species.text()),
                rel_loc=self._none_if_empty(self.cjs_rel_loc.text()),
                cap_loc=self._none_if_empty(self.cjs_cap_loc.text()),
                initial_recap_release=bool(self.cjs_initial_release.isChecked()),
            )
            cjs.input_file(model_name, output_ws)
            cjs.inp.to_csv(os.path.join(output_ws, f"{model_name}.csv"), index=False)
            self.cjs_obj = cjs
            self.log(f"CJS outputs written to {output_ws}")

        self._run_action("Run CJS Export", _impl)


def _find_repo_root() -> Path:
    module_dir = Path(__file__).resolve().parent
    return module_dir.parent


def main() -> None:
    app = QApplication(sys.argv)
    repo_root = _find_repo_root()

    window = WorkflowWindow(repo_root)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
