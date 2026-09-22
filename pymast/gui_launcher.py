"""PyMAST full GUI workflow runner (no script execution)."""

from __future__ import annotations

import ast
import datetime
import faulthandler
import io
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from pymast import formatter
from pymast.overlap_removal import bout, overlap_reduction
from pymast.radio_project import radio_project

_FAULT_LOG_HANDLE = None

matplotlib.use('Agg')

try:
    from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal
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
        QSplitter,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    try:
        from PyQt5.QtCore import QObject, QThread, Qt, QTimer, pyqtSignal as Signal
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
            QSplitter,
            QStackedWidget,
            QTableWidget,
            QTableWidgetItem,
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

TRAINING_PLOT_PARAMETERS = [
    "Hit Ratio Distribution",
    "Consecutive Hit Length",
    "Signal Power Distribution",
    "Noise Ratio Distribution",
    "Lag Differences",
]


class ActionCancelled(Exception):
    """Raised when a background GUI action is cancelled by the user."""


class AsyncActionWorker(QObject):
    finished = Signal()
    cancelled = Signal(str)
    failed = Signal(str, str)
    output = Signal(str)

    def __init__(self, fn, cancel_check=None):
        super().__init__()
        self._fn = fn
        self._cancel_check = cancel_check

    def _is_cancel_requested(self) -> bool:
        return bool(self._cancel_check and self._cancel_check())

    def run(self):
        capture = io.StringIO()
        try:
            if self._is_cancel_requested():
                raise ActionCancelled("Cancelled before execution started.")
            with redirect_stdout(capture), redirect_stderr(capture):
                self._fn()
            if self._is_cancel_requested():
                raise ActionCancelled("Cancelled.")
            output_text = capture.getvalue().strip()
            if output_text:
                self.output.emit(output_text)
            self.finished.emit()
        except ActionCancelled as exc:
            output_text = capture.getvalue().strip()
            if output_text:
                self.output.emit(output_text)
            self.cancelled.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            output_text = capture.getvalue().strip()
            if output_text:
                self.output.emit(output_text)
            self.failed.emit(str(exc), traceback.format_exc())

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
        self._active_thread: Optional[QThread] = None
        self._active_worker: Optional[AsyncActionWorker] = None
        self._cancel_requested = False
        self._active_action_label: Optional[str] = None
        self._session_state_path: Optional[Path] = None

        self.setWindowTitle("PyMAST Full GUI Workflow")
        self.resize(1300, 900)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(10)
        main_splitter.setOpaqueResize(True)
        root_layout.addWidget(main_splitter, stretch=1)

        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setChildrenCollapsible(False)
        left_splitter.setHandleWidth(10)
        left_splitter.setOpaqueResize(True)
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.setHandleWidth(10)
        right_splitter.setOpaqueResize(True)

        workflow_panel = QWidget()
        workflow_layout = QVBoxLayout(workflow_panel)
        workflow_layout.setContentsMargins(0, 0, 0, 0)
        workflow_layout.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        if self.logo_path.exists():
            self.header_logo_label = QLabel()
            self.header_logo_label.setPixmap(self._icon_logo_pixmap(width=44))
            header_row.addWidget(self.header_logo_label)

        self.header_step_title = QLabel("Home")
        self.header_step_title.setStyleSheet("font-size: 18px; font-weight: 700;")
        header_row.addWidget(self.header_step_title)

        self.header_help_btn = QPushButton("?")
        self.header_help_btn.setFixedWidth(28)
        self._tip(self.header_help_btn, "Open detailed help for the current section.")
        self.header_help_btn.clicked.connect(self._show_current_step_help)
        header_row.addWidget(self.header_help_btn)
        header_row.addStretch(1)

        self.cancel_action_btn = QPushButton("Cancel Running Action")
        self.cancel_action_btn.setEnabled(False)
        self.cancel_action_btn.clicked.connect(self.cancel_active_action)
        self._tip(self.cancel_action_btn, "Request cancellation for the currently running background action.")
        self.save_session_btn = QPushButton("Save GUI Session")
        self.save_session_btn.setEnabled(False)
        self.save_session_btn.clicked.connect(self.save_gui_session)
        self._tip(self.save_session_btn, "Save current GUI form values to a sidecar file next to the project HDF5 database.")
        self.load_session_btn = QPushButton("Load GUI Session")
        self.load_session_btn.setEnabled(False)
        self.load_session_btn.clicked.connect(self.load_gui_session)
        self._tip(self.load_session_btn, "Reload previously saved GUI form values for the current project database.")
        header_row.addWidget(self.save_session_btn)
        header_row.addWidget(self.load_session_btn)
        header_row.addWidget(self.cancel_action_btn)
        workflow_layout.addLayout(header_row)

        self.stack = QStackedWidget()
        workflow_layout.addWidget(self.stack, stretch=1)

        log_group = QGroupBox("Workflow Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(6, 18, 6, 6)
        log_layout.setSpacing(4)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Workflow output and errors will appear here.")
        log_layout.addWidget(self.log_output, stretch=1)

        self.plot_viewer_group = self._build_plot_viewer_group()
        self.data_viewer_group = self._build_data_viewer_group()

        left_splitter.addWidget(workflow_panel)
        left_splitter.addWidget(log_group)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 1)

        right_splitter.addWidget(self.plot_viewer_group)
        right_splitter.addWidget(self.data_viewer_group)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setCollapsible(0, True)  # plot panel is collapsible

        self._left_splitter = left_splitter
        self._right_splitter = right_splitter

        # Keep the two vertical splitters in sync so dividers snap to the same row.
        left_splitter.splitterMoved.connect(lambda pos, idx: self._right_splitter.setSizes(self._left_splitter.sizes()))
        right_splitter.splitterMoved.connect(lambda pos, idx: self._left_splitter.setSizes(self._right_splitter.sizes()))

        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setSizes([700, 1000])

        self.setCentralWidget(root)

        self.home_page = self._build_home_page()
        self.stack.addWidget(self.home_page)

        self.step_pages: Dict[int, QWidget] = {}
        for step in range(0, 9):
            page = self._build_step_page(step)
            self.step_pages[step] = page
            self.stack.addWidget(page)
        self._update_step_header()
        self._update_plot_controls_for_step(self._current_step_index())

    def _build_plot_viewer_group(self) -> QGroupBox:
        group = QGroupBox("Plot Viewer")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        self.plot_selector_label = QLabel("Parameter:")
        self.plot_selector_combo = QComboBox()
        self.plot_selector_combo.currentIndexChanged.connect(self._on_plot_selection_changed)
        self._tip(self.plot_selector_combo, "Select which training parameter to view")
        
        # Initialize dropdown with training/classification parameters.
        self.plot_parameter_names = list(TRAINING_PLOT_PARAMETERS)
        self.plot_selector_combo.blockSignals(True)
        self.plot_selector_combo.clear()
        for param_name in self.plot_parameter_names:
            self.plot_selector_combo.addItem(param_name)
        self.plot_selector_combo.blockSignals(False)
        
        controls_row.addWidget(self.plot_selector_label)
        controls_row.addWidget(self.plot_selector_combo)
        controls_row.addStretch(1)
        layout.addLayout(controls_row)

        self.plot_scroll = QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_container = QWidget()
        self.plot_container_layout = QVBoxLayout(self.plot_container)
        self.plot_container_layout.setContentsMargins(4, 4, 4, 4)
        self.plot_container_layout.setSpacing(8)

        self.plot_viewer_label = QLabel("Plot output preview area.\nGenerated figures can be shown here.")
        self.plot_viewer_label.setAlignment(Qt.AlignCenter)
        self.plot_viewer_label.setStyleSheet("color: #999; font-size: 11px;")
        self.plot_container_layout.addWidget(self.plot_viewer_label)
        self.plot_image_label = QLabel()
        self.plot_image_label.setAlignment(Qt.AlignCenter)
        self.plot_image_label.hide()
        self.plot_container_layout.addWidget(self.plot_image_label)
        self.plot_container_layout.addStretch(1)

        self.plot_scroll.setWidget(self.plot_container)
        layout.addWidget(self.plot_scroll, stretch=1)
        
        self._plot_figures: Dict[str, Any] = {}
        self._plot_pixmaps: Dict[str, QPixmap] = {}
        self._trained_table_cache: Dict[str, pd.DataFrame] = {}
        self._active_plot_context: Optional[Tuple[str, str]] = None
        
        return group

    def _build_data_viewer_group(self) -> QGroupBox:
        group = QGroupBox("Project Data Viewer")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        shortcut_row = QHBoxLayout()
        filter_row = QHBoxLayout()

        key_label = QLabel("HDF Key")
        self.viewer_key_combo = QComboBox()
        self.viewer_key_combo.setMinimumWidth(180)
        self.viewer_key_combo.currentTextChanged.connect(self._on_viewer_key_changed)
        self._tip(self.viewer_key_combo, "Select an HDF table/group to preview.")

        refresh_keys_btn = QPushButton("Refresh Keys")
        refresh_keys_btn.clicked.connect(self.refresh_data_viewer_keys)
        self._tip(refresh_keys_btn, "Reload available HDF keys from the current project database.")

        self.viewer_limit_spin = QSpinBox()
        self.viewer_limit_spin.setRange(1, 10000)
        self.viewer_limit_spin.setValue(250)
        self._tip(self.viewer_limit_spin, "Number of rows to preview per page.")

        self.viewer_offset_spin = QSpinBox()
        self.viewer_offset_spin.setRange(0, 100000000)
        self.viewer_offset_spin.setSingleStep(250)
        self._tip(self.viewer_offset_spin, "Row offset into the selected dataset for paging through data.")

        self.viewer_where_edit = QLineEdit()
        self.viewer_where_edit.setPlaceholderText("Optional where clause, e.g. rec_id == 'R01'")
        self._tip(self.viewer_where_edit, "Optional PyTables where clause for fast filtered preview when the table supports it.")

        refresh_view_btn = QPushButton("Load Preview")
        refresh_view_btn.clicked.connect(self.refresh_data_viewer)
        self._tip(refresh_view_btn, "Load a preview slice from the selected HDF key.")

        step_view_btn = QPushButton("Load Step Output")
        step_view_btn.clicked.connect(self.load_current_step_view)
        self._tip(step_view_btn, "Select the HDF table most relevant to the currently visible workflow step.")

        filter_rec_btn = QPushButton("Filter Current Receiver")
        filter_rec_btn.clicked.connect(self.filter_viewer_to_current_receiver)
        self._tip(filter_rec_btn, "Apply a quick where-clause filter using the receiver ID from the active workflow step when available.")

        clear_filter_btn = QPushButton("Clear Filters")
        clear_filter_btn.clicked.connect(self.clear_viewer_filter)
        self._tip(clear_filter_btn, "Clear all manual filters and where clause.")

        qc_summary_btn = QPushButton("Load Step QC")
        qc_summary_btn.clicked.connect(self.load_step_qc_summary)
        self._tip(qc_summary_btn, "Generate a scientist-oriented QC summary for the active workflow stage.")

        controls.addWidget(key_label)
        controls.addWidget(self.viewer_key_combo)
        controls.addWidget(refresh_keys_btn)
        controls.addWidget(QLabel("Rows"))
        controls.addWidget(self.viewer_limit_spin)
        controls.addWidget(QLabel("Offset"))
        controls.addWidget(self.viewer_offset_spin)
        controls.addWidget(self.viewer_where_edit, stretch=1)
        controls.addWidget(refresh_view_btn)

        # Filter row for dynamic filter dropdowns
        self.viewer_filter_dropdowns: Dict[str, QComboBox] = {}
        filter_row.addWidget(QLabel("Filters:"))
        filter_row.addStretch(1)

        shortcut_row.addWidget(step_view_btn)
        shortcut_row.addWidget(filter_rec_btn)
        shortcut_row.addWidget(clear_filter_btn)
        shortcut_row.addWidget(qc_summary_btn)
        shortcut_row.addStretch(1)

        self.viewer_status_label = QLabel("No project database loaded.")
        self.viewer_status_label.setStyleSheet("color: #666;")

        self.viewer_summary = QPlainTextEdit()
        self.viewer_summary.setReadOnly(True)
        self.viewer_summary.setPlaceholderText("Dataset summary will appear here.")
        self.viewer_summary.setMaximumHeight(100)

        self.viewer_qc_summary = QPlainTextEdit()
        self.viewer_qc_summary.setReadOnly(True)
        self.viewer_qc_summary.setPlaceholderText("Stage-specific QC summary will appear here.")
        self.viewer_qc_summary.setMaximumHeight(150)

        self.viewer_table = QTableWidget()
        self.viewer_table.setAlternatingRowColors(True)
        self.viewer_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.viewer_table.setSortingEnabled(False)
        self.viewer_table.verticalHeader().setVisible(False)

        layout.addLayout(controls)
        layout.addLayout(shortcut_row)
        layout.addLayout(filter_row)
        layout.addWidget(self.viewer_status_label)
        layout.addWidget(self.viewer_summary)
        layout.addWidget(self.viewer_qc_summary)
        layout.addWidget(self.viewer_table)

        return group

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
            btn = QPushButton(STEP_TITLES[step])
            btn.setMinimumHeight(72)
            btn.setMinimumWidth(80)
            btn.setStyleSheet("font-size: 10px; font-weight: 700;")
            btn.setToolTip(STEP_TITLES[step])
            btn.clicked.connect(lambda checked=False, s=step: self.goto_step(s))
            row = (i) // 3
            col = (i) % 3
            grid_layout.addWidget(btn, row, col)

        grid.setLayout(grid_layout)
        right_layout.addWidget(grid)

        info_text = QLabel("Click a workflow tile to configure and run that stage.")
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
        outer.setContentsMargins(0, 0, 0, 4)
        outer.setSpacing(4)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 0)

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

        # Absorb extra vertical space so form widgets don't stretch to fill the scroll area.
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(0)  # remove border so inner groupbox aligns with log panel
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

        # Project directory — browse triggers a database scan.
        self.project_dir_edit = QLineEdit()
        self._tip(self.project_dir_edit, "Root project folder. Should contain setup CSV files and will hold Data/, Output/, and the HDF5 database.")
        self.project_dir_edit.editingFinished.connect(self._scan_project_dir_for_databases)
        project_dir_browse_btn = QPushButton("Browse")
        self._tip(project_dir_browse_btn, "Browse to the project folder. Existing .h5 databases in that folder will be listed automatically.")
        project_dir_browse_btn.clicked.connect(self._browse_project_dir)
        project_dir_row = QWidget()
        _pdl = QHBoxLayout(project_dir_row)
        _pdl.setContentsMargins(0, 0, 0, 0)
        _pdl.addWidget(self.project_dir_edit)
        _pdl.addWidget(project_dir_browse_btn)

        # Database selector — populated from directory scan; editable for new names.
        self.db_name_combo = QComboBox()
        self.db_name_combo.setEditable(True)
        self.db_name_combo.lineEdit().setPlaceholderText("Select existing or enter new database name")
        self._tip(self.db_name_combo, "Pick an existing .h5 database found in the project folder, or type a new name (no .h5 extension needed).")
        self.db_status_label = QLabel("Browse to a project directory to scan for databases.")
        self.db_status_label.setStyleSheet("color: #666; font-style: italic; font-size: 11px;")
        self._tip(self.db_status_label, "Shows how many .h5 databases were found in the selected directory.")

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
        self.nodes_csv_edit.editingFinished.connect(self._auto_preview_receiver_network_from_setup)

        f_project.addRow("Project Directory", project_dir_row)
        f_project.addRow("Database", self.db_name_combo)
        f_project.addRow("", self.db_status_label)
        f_project.addRow("Detection Count", self.det_count_spin)
        f_project.addRow("Duration", self.duration_spin)
        f_project.addRow("Tag Metadata CSV", tag_row)
        f_project.addRow("Receiver Metadata CSV", rec_row)
        f_project.addRow("Nodes Metadata CSV", nodes_row)

        init_btn = QPushButton("Initialize / Reload Project")
        init_btn.clicked.connect(self.initialize_project_from_form)
        self._tip(init_btn, "Loads metadata CSVs and initializes/reloads the project database object. Creates a new database if the name doesn't exist yet.")
        f_project.addRow(init_btn)

        parent_layout.addWidget(g_project)

    def _build_step1(self, parent_layout: QVBoxLayout) -> None:
        g_import = QGroupBox("Data Import Parameters")
        f_import = QFormLayout(g_import)

        self.import_rec_id = QComboBox()
        self.import_rec_id.setMinimumWidth(180)
        self.import_rec_id.setEditable(False)
        self._tip(self.import_rec_id, "Receiver ID to import. Populated from the receiver metadata table after project initialization.")
        self.import_rec_type = QComboBox()
        self.import_rec_type.addItems([
            "srx1200", "srx800", "srx600", "orion", "ares", "vr2", "pit", "pit_multiple"
        ])
        self.import_rec_id.currentTextChanged.connect(self._sync_import_rec_type_from_receiver)
        self._refresh_import_receiver_ids()
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
 
        self.train_rec_id = QComboBox()
        self.train_rec_id.setMinimumWidth(180)
        self.train_rec_id.setEditable(False)
        self._tip(self.train_rec_id, "Receiver ID to train against. Populated from the receiver metadata table after project initialization.")
        self.train_rec_type = QComboBox()
        self.train_rec_type.setMinimumWidth(180)
        self.train_rec_type.setEditable(False)
        self._tip(self.train_rec_type, "Receiver type label used for training summary output.")
        self.train_rec_id.currentTextChanged.connect(self._sync_train_rec_type_from_receiver)
        self._refresh_train_receiver_ids()
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
            if target is self.nodes_csv_edit:
                self._auto_preview_receiver_network_from_setup()

    def _browse_dir(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Directory", str(self.repo_root))
        if path:
            target.setText(path)

    def _browse_project_dir(self) -> None:
        start = self.project_dir_edit.text().strip() or str(self.repo_root)
        path = QFileDialog.getExistingDirectory(self, "Select Project Directory", start)
        if path:
            self.project_dir_edit.setText(path)
            self._scan_project_dir_for_databases()

    def _scan_project_dir_for_databases(self) -> None:
        """Scan the project directory for .h5 files and populate the database combo."""
        project_dir = self.project_dir_edit.text().strip()
        if not project_dir:
            return
        p = Path(project_dir)
        if not p.is_dir():
            self.db_status_label.setText("Directory not found.")
            self.db_status_label.setStyleSheet("color: #cc0000; font-style: italic; font-size: 11px;")
            return

        h5_files = sorted(list(p.glob("*.h5")) + list(p.glob("*.hdf5")))
        current_text = self.db_name_combo.currentText().strip()
        self.db_name_combo.blockSignals(True)
        self.db_name_combo.clear()

        if h5_files:
            for f in h5_files:
                self.db_name_combo.addItem(f.stem)
            idx = self.db_name_combo.findText(current_text)
            self.db_name_combo.setCurrentIndex(idx if idx >= 0 else 0)
            n = len(h5_files)
            self.db_status_label.setText(f"{n} database{'s' if n != 1 else ''} found — select one or type a new name.")
            self.db_status_label.setStyleSheet("color: #336699; font-style: italic; font-size: 11px;")
        else:
            self.db_name_combo.lineEdit().setPlaceholderText("No databases found — enter a new name")
            self.db_status_label.setText("No existing databases found. Enter a new name above.")
            self.db_status_label.setStyleSheet("color: #666; font-style: italic; font-size: 11px;")

        self.db_name_combo.blockSignals(False)
        self._autofill_csv_from_dir(p)
        self._auto_preview_receiver_network_from_setup()

    def _autofill_csv_from_dir(self, p: Path) -> None:
        """Fill empty CSV path fields if standard files exist in the given directory."""
        for filename, widget in [
            ("tblMasterTag.csv", self.tag_csv_edit),
            ("tblMasterReceiver.csv", self.receiver_csv_edit),
            ("tblNodes.csv", self.nodes_csv_edit),
        ]:
            if not widget.text().strip():
                candidate = p / filename
                if candidate.exists():
                    widget.setText(str(candidate))

    def _browse_existing_hdf_for_setup(self) -> None:
        start_dir = self.project_dir_edit.text().strip() or str(self.repo_root)
        hdf_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Existing HDF5 Database",
            start_dir,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*.*)",
        )
        if not hdf_path:
            return

        selected = Path(hdf_path)
        self.project_dir_edit.setText(str(selected.parent))
        self._trained_table_cache.clear()
        self._plot_pixmaps.clear()
        self._plot_figures.clear()
        self._active_plot_context = None
        # Rescan so the combo is populated, then select the chosen file.
        self._scan_project_dir_for_databases()
        idx = self.db_name_combo.findText(selected.stem)
        if idx >= 0:
            self.db_name_combo.setCurrentIndex(idx)
        else:
            self.db_name_combo.setCurrentText(selected.stem)
        if hasattr(self, "import_db_dir"):
            self.import_db_dir.setText(str(selected))
        self.log(f"Selected existing database: {selected}")
        self._auto_preview_receiver_network_from_setup()

    def _find_column_name(self, frame: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        by_lower = {str(col).lower(): str(col) for col in frame.columns}
        for candidate in candidates:
            found = by_lower.get(candidate.lower())
            if found is not None:
                return found
        return None

    def _extract_node_edges_from_frame(self, frame: pd.DataFrame) -> List[Tuple[str, str]]:
        pairs = [
            ("parent", "child"),
            ("source", "target"),
            ("from", "to"),
            ("from_node", "to_node"),
            ("node_from", "node_to"),
            ("node1", "node2"),
            ("upstream", "downstream"),
        ]
        for left_name, right_name in pairs:
            left_col = self._find_column_name(frame, [left_name])
            right_col = self._find_column_name(frame, [right_name])
            if left_col is None or right_col is None:
                continue
            edges: List[Tuple[str, str]] = []
            for _, row in frame[[left_col, right_col]].dropna().iterrows():
                left_val = str(row[left_col]).strip()
                right_val = str(row[right_col]).strip()
                if left_val and right_val and left_val != right_val:
                    edges.append((left_val, right_val))
            if edges:
                return edges
        return []

    def _collect_setup_network_edges(
        self,
        nodes_df: pd.DataFrame,
        receivers_df: Optional[pd.DataFrame],
    ) -> List[Tuple[str, str]]:
        # 1) Prefer explicit edge columns directly on nodes metadata.
        edges = self._extract_node_edges_from_frame(nodes_df)
        if edges:
            return edges

        # 2) Try optional /project_setup/lines table from HDF if available.
        db_path = self._resolve_project_db_path()
        if db_path is not None and db_path.exists():
            try:
                with pd.HDFStore(str(db_path), mode="r") as store:
                    if "/project_setup/lines" in store.keys():
                        lines_df = store.select("/project_setup/lines")
                        edges = self._extract_node_edges_from_frame(lines_df)
                        if edges:
                            return edges
            except Exception:  # noqa: BLE001
                pass

        # 3) Try receiver-level relationships and map rec_id -> node.
        if receivers_df is not None and not receivers_df.empty:
            rec_id_col = self._find_column_name(receivers_df, ["rec_id"])
            rec_node_col = self._find_column_name(receivers_df, ["node"])
            if rec_id_col is not None and rec_node_col is not None:
                rec_map = (
                    receivers_df[[rec_id_col, rec_node_col]]
                    .dropna()
                    .assign(**{rec_id_col: lambda x: x[rec_id_col].astype(str), rec_node_col: lambda x: x[rec_node_col].astype(str)})
                )
                rec_to_node = dict(zip(rec_map[rec_id_col], rec_map[rec_node_col]))

                rec_edges = self._extract_node_edges_from_frame(receivers_df)
                node_edges: List[Tuple[str, str]] = []
                for left_rec, right_rec in rec_edges:
                    left_node = rec_to_node.get(left_rec)
                    right_node = rec_to_node.get(right_rec)
                    if left_node and right_node and left_node != right_node:
                        node_edges.append((left_node, right_node))
                if node_edges:
                    return node_edges

        return []

    def _auto_preview_receiver_network_from_setup(self) -> None:
        if self._current_step_index() != 0:
            return
        nodes_path = self.nodes_csv_edit.text().strip()
        if not nodes_path:
            return
        if not os.path.exists(nodes_path):
            self.log(f"Nodes CSV not found for network preview: {nodes_path}")
            return

        try:
            self.preview_receiver_network_from_setup()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Setup network preview skipped: {exc}")

    def preview_receiver_network_from_setup(self) -> None:
        nodes_df: Optional[pd.DataFrame] = None
        receivers_df: Optional[pd.DataFrame] = None

        if self.project is not None and isinstance(getattr(self.project, "nodes", None), pd.DataFrame):
            nodes_df = self.project.nodes.copy()
        else:
            nodes_path = self.nodes_csv_edit.text().strip()
            if not nodes_path:
                raise ValueError("Nodes CSV path is required to preview receiver network.")
            if not os.path.exists(nodes_path):
                raise FileNotFoundError(f"Nodes CSV not found: {nodes_path}")
            nodes_df = pd.read_csv(nodes_path)

        if self.project is not None and isinstance(getattr(self.project, "receivers", None), pd.DataFrame):
            receivers_df = self.project.receivers.reset_index()
        else:
            receivers_path = self.receiver_csv_edit.text().strip()
            if receivers_path and os.path.exists(receivers_path):
                receivers_df = pd.read_csv(receivers_path)

        node_col = self._find_column_name(nodes_df, ["node"])
        x_col = self._find_column_name(nodes_df, ["X", "x"])
        y_col = self._find_column_name(nodes_df, ["Y", "y"])
        if node_col is None or x_col is None or y_col is None:
            raise ValueError(
                "Nodes CSV must include columns: node, X, Y (case-insensitive for X/Y)."
            )

        working = nodes_df[[node_col, x_col, y_col]].copy()
        working.columns = ["node", "X", "Y"]
        working["node"] = working["node"].astype(str)
        working["X"] = pd.to_numeric(working["X"], errors="coerce")
        working["Y"] = pd.to_numeric(working["Y"], errors="coerce")
        working = working.dropna(subset=["X", "Y"])
        if working.empty:
            raise ValueError("Nodes CSV does not contain any valid numeric X/Y coordinates.")

        receiver_labels: Dict[str, List[str]] = {}
        if receivers_df is not None and not receivers_df.empty:
            rec_id_col = self._find_column_name(receivers_df, ["rec_id"])
            rec_node_col = self._find_column_name(receivers_df, ["node"])
            if rec_id_col is not None and rec_node_col is not None:
                rec_work = receivers_df[[rec_id_col, rec_node_col]].copy().dropna()
                rec_work.columns = ["rec_id", "node"]
                rec_work["rec_id"] = rec_work["rec_id"].astype(str)
                rec_work["node"] = rec_work["node"].astype(str)
                grouped = rec_work.groupby("node")["rec_id"].apply(list)
                receiver_labels = {str(node): values for node, values in grouped.items()}

        graph = nx.Graph()
        node_ids: List[str] = []
        for _, row in working.iterrows():
            node = str(row["node"])
            graph.add_node(node)
            node_ids.append(node)

        explicit_edges = self._collect_setup_network_edges(nodes_df, receivers_df)
        filtered_edges = [(a, b) for a, b in explicit_edges if a in graph.nodes and b in graph.nodes and a != b]
        graph.add_edges_from(filtered_edges)

        # Relationship view: circular topology layout for readable node connectivity.
        draw_pos = nx.circular_layout(graph) if graph.number_of_nodes() > 0 else {}

        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            nx.draw_networkx_edges(graph, draw_pos, ax=ax, width=1.5, alpha=0.8, edge_color="#4d4d4d")
            nx.draw_networkx_nodes(
                graph,
                draw_pos,
                ax=ax,
                node_color="#2a9d8f",
                edgecolors="black",
                linewidths=0.6,
                node_size=220,
            )

            for node, (x_val, y_val) in draw_pos.items():
                label = node
                receivers_at_node = receiver_labels.get(node)
                if receivers_at_node:
                    receiver_count = len(receivers_at_node)
                    if receiver_count <= 3:
                        label = f"{node}\n" + ", ".join(receivers_at_node)
                    else:
                        label = f"{node}\n({receiver_count} receivers)"
                ax.annotate(
                    label,
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )

            ax.set_title("Receiver Network Graph", fontsize=12, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.set_aspect("equal", adjustable="datalim")
            fig.tight_layout()

            self._display_plot(fig)
            if graph.number_of_edges() == 0:
                self.log(
                    "Rendered receiver network graph with nodes only (no explicit edges found in nodes/lines metadata)."
                )
            else:
                self.log(f"Rendered receiver network graph with {graph.number_of_nodes()} node(s) and {graph.number_of_edges()} edge(s).")
        finally:
            plt.close(fig)

    def goto_home(self) -> None:
        self.stack.setCurrentWidget(self.home_page)
        self._update_step_header()
        self._update_plot_controls_for_step(self._current_step_index())

    def goto_step(self, step: int) -> None:
        page = self.step_pages.get(step)
        if page is not None:
            self.stack.setCurrentWidget(page)
            self.refresh_data_viewer_keys()
            self._set_viewer_key_for_step(step)
            self._update_step_header()
            self._update_plot_controls_for_step(step)
            if step == 0:
                self._auto_preview_receiver_network_from_setup()

    def _update_plot_controls_for_step(self, step: int) -> None:
        enabled = step in {2, 3}

        self.plot_selector_combo.blockSignals(True)
        self.plot_selector_combo.clear()
        if enabled:
            self.plot_selector_combo.addItems(TRAINING_PLOT_PARAMETERS)
            self.plot_selector_combo.setEnabled(True)
            self.plot_selector_label.setText("Parameter:")
            self.plot_viewer_label.setText("Select a training/classification parameter to preview.")
            # Expand the plot panel when the step supports it.
            total = self._right_splitter.height()
            sizes = self._right_splitter.sizes()
            if sizes[0] == 0:
                half = total // 2
                self._right_splitter.setSizes([half, total - half])
        else:
            self.plot_selector_combo.addItem("No plots for this workflow step")
            self.plot_selector_combo.setEnabled(False)
            self.plot_selector_label.setText("Plot:")
            self.plot_viewer_label.setText(
                "This workflow step does not provide a parameter histogram view.\n"
                "Use Training or Classification to view these plots."
            )
            # Collapse the plot panel for steps that don't use it.
            total = self._right_splitter.height()
            self._right_splitter.setSizes([0, total])
        self.plot_selector_combo.blockSignals(False)

    def _show_current_step_help(self) -> None:
        if self.stack.currentWidget() is self.home_page:
            return
        self.show_step_help(self._current_step_index())

    def _update_step_header(self) -> None:
        if self.stack.currentWidget() is self.home_page:
            self.header_step_title.setText("Home")
            self.header_help_btn.setEnabled(False)
            return
        step = self._current_step_index()
        if step == 0:
            title = "Project Setup"
        else:
            title = f"Step {step:02d}: {STEP_TITLES[step]}"
        self.header_step_title.setText(title)
        self.header_help_btn.setEnabled(True)

    def log(self, message: str) -> None:
        self.log_output.appendPlainText(message)
        QApplication.processEvents()

    def _resolve_project_db_path(self) -> Optional[Path]:
        if self.project is not None and getattr(self.project, "db", None):
            return Path(self.project.db)

        raw_path = self.import_db_dir.text().strip() if hasattr(self, "import_db_dir") else ""
        if raw_path:
            return Path(raw_path)
        return None

    def _update_session_state_path(self) -> Optional[Path]:
        db_path = self._resolve_project_db_path()
        self._session_state_path = db_path.with_suffix(db_path.suffix + ".gui_session.json") if db_path else None
        enabled = self._session_state_path is not None
        self.save_session_btn.setEnabled(enabled)
        self.load_session_btn.setEnabled(enabled)
        return self._session_state_path

    def _current_step_index(self) -> int:
        if self.stack.currentWidget() is self.home_page:
            return 0
        for step, page in self.step_pages.items():
            if self.stack.currentWidget() is page:
                return step
        return 0

    def _suggested_viewer_key_for_step(self, step: int) -> Optional[str]:
        mapping = {
            0: "/project_setup/receivers",
            1: "/raw_data",
            2: "/trained",
            3: "/classified",
            4: "/presence",
            5: "/overlapping",
            6: "/recaptures",
            7: "/recaptures",
            8: "/recaptures",
        }
        return mapping.get(step)

    def _set_viewer_key_for_step(self, step: int) -> None:
        key = self._suggested_viewer_key_for_step(step)
        if not key:
            return
        idx = self.viewer_key_combo.findText(key)
        if idx >= 0:
            self.viewer_key_combo.setCurrentIndex(idx)

    def _current_receiver_context(self) -> Optional[str]:
        step = self._current_step_index()
        field_by_step = {
            1: "import_rec_id",
            2: "train_rec_id",
            3: "class_rec_id",
            4: "bout_rec_id",
        }
        field_name = field_by_step.get(step)
        if not field_name:
            return None
        widget = getattr(self, field_name, None)
        if isinstance(widget, QLineEdit):
            value = widget.text().strip()
            return value or None
        if isinstance(widget, QComboBox):
            value = widget.currentText().strip()
            return value or None
        return None

    def _project_receiver_ids(self) -> List[str]:
        proj = self.project
        if proj is None or not hasattr(proj, "receivers"):
            return []
        receivers = proj.receivers
        if isinstance(receivers, pd.DataFrame):
            ids = [str(idx).strip() for idx in receivers.index.tolist()]
            return [receiver_id for receiver_id in ids if receiver_id]
        return []

    def _sync_import_rec_type_from_receiver(self, rec_id: str) -> None:
        rec = rec_id.strip()
        if not rec or self.project is None or not hasattr(self.project, "receivers"):
            return
        receivers = self.project.receivers
        if not isinstance(receivers, pd.DataFrame):
            return
        if rec not in receivers.index or "rec_type" not in receivers.columns:
            return
        rec_type = str(receivers.loc[rec, "rec_type"]).strip().lower()
        if not rec_type:
            return
        idx = self.import_rec_type.findText(rec_type)
        if idx >= 0:
            self.import_rec_type.setCurrentIndex(idx)

    def _refresh_import_receiver_ids(self, preferred_rec_id: Optional[str] = None) -> None:
        receiver_ids = self._project_receiver_ids()
        current = preferred_rec_id or self.import_rec_id.currentText().strip()
        self.import_rec_id.clear()
        self.import_rec_id.addItems(receiver_ids)
        self.import_rec_id.setEnabled(bool(receiver_ids))
        if receiver_ids:
            idx = self.import_rec_id.findText(current)
            self.import_rec_id.setCurrentIndex(idx if idx >= 0 else 0)
            self._sync_import_rec_type_from_receiver(self.import_rec_id.currentText())
 
    def _sync_train_rec_type_from_receiver(self, rec_id: str) -> None:
        rec = rec_id.strip()
        if not rec or self.project is None or not hasattr(self.project, "receivers"):
            return
        receivers = self.project.receivers
        if not isinstance(receivers, pd.DataFrame):
            return
        if rec not in receivers.index or "rec_type" not in receivers.columns:
            return
        rec_type = str(receivers.loc[rec, "rec_type"]).strip().lower()
        if not rec_type:
            return
        self.train_rec_type.clear()
        self.train_rec_type.addItem(rec_type)
 
    def _refresh_train_receiver_ids(self, preferred_rec_id: Optional[str] = None) -> None:
        receiver_ids = self._project_receiver_ids()
        current = preferred_rec_id or self.train_rec_id.currentText().strip()
        self.train_rec_id.clear()
        self.train_rec_id.addItems(receiver_ids)
        self.train_rec_id.setEnabled(bool(receiver_ids))
        if receiver_ids:
            idx = self.train_rec_id.findText(current)
            self.train_rec_id.setCurrentIndex(idx if idx >= 0 else 0)
            self._sync_train_rec_type_from_receiver(self.train_rec_id.currentText())

    def _supported_state_widgets(self) -> Dict[str, QWidget]:
        supported = {}
        excluded = {
            "log_output",
            "viewer_table",
            "stack",
            "home_page",
        }
        for name, value in self.__dict__.items():
            if name.startswith("_") or name in excluded:
                continue
            if isinstance(value, (QLineEdit, QPlainTextEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox)):
                supported[name] = value
        return supported

    def _collect_gui_session_state(self) -> Dict[str, Any]:
        widget_state: Dict[str, Any] = {}
        for name, widget in self._supported_state_widgets().items():
            if isinstance(widget, QLineEdit):
                widget_state[name] = widget.text()
            elif isinstance(widget, QPlainTextEdit):
                widget_state[name] = widget.toPlainText()
            elif isinstance(widget, QCheckBox):
                widget_state[name] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                widget_state[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                widget_state[name] = widget.value()
            elif isinstance(widget, QComboBox):
                widget_state[name] = widget.currentText()

        current_step = 0
        for step, page in self.step_pages.items():
            if self.stack.currentWidget() is page:
                current_step = step
                break

        return {
            "current_step": current_step,
            "widgets": widget_state,
        }

    def _apply_gui_session_state(self, state: Dict[str, Any]) -> None:
        widgets = state.get("widgets", {})
        for name, value in widgets.items():
            widget = getattr(self, name, None)
            if widget is None:
                continue

            if isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QPlainTextEdit):
                widget.setPlainText(str(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QComboBox):
                idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                elif widget.isEditable():
                    # Editable combos (e.g. db_name_combo) may not yet contain
                    # this value as an item (list not scanned/populated), so
                    # fall back to setting the edit text directly.
                    widget.setCurrentText(str(value))

        step = state.get("current_step")
        if isinstance(step, int) and step in self.step_pages:
            self.goto_step(step)

    def save_gui_session(self) -> None:
        session_path = self._update_session_state_path()
        if session_path is None:
            raise RuntimeError("No project database is available for GUI session persistence yet.")

        session_path.write_text(json.dumps(self._collect_gui_session_state(), indent=2), encoding="utf-8")
        self.log(f"Saved GUI session: {session_path}")

    def load_gui_session(self) -> None:
        session_path = self._update_session_state_path()
        if session_path is None:
            raise RuntimeError("No project database is available for GUI session persistence yet.")
        if not session_path.exists():
            self.log(f"No saved GUI session found at: {session_path}")
            return
 
        self._refresh_import_receiver_ids()
        self._refresh_train_receiver_ids()
        state = json.loads(session_path.read_text(encoding="utf-8"))
        self._apply_gui_session_state(state)
        self._sync_import_rec_type_from_receiver(self.import_rec_id.currentText())
        self._sync_train_rec_type_from_receiver(self.train_rec_id.currentText())
        self.refresh_data_viewer_keys()
        self._set_viewer_key_for_step(self._current_step_index())
        self.log(f"Loaded GUI session: {session_path}")

    def _get_available_viewer_keys_for_step(self, step: int) -> Optional[List[str]]:
        """
        Get available HDF keys for the current step.
        
        Returns None to show all keys (on home page),
        or a list of specific keys for that step (workflow pages).
        """
        # On home page, show all keys
        if step < 0 or step == 0:
            return None
        
        # Map each step to its allowed viewer key(s)
        step_key_mapping = {
            1: ["/project_setup/receivers", "/raw_data"],
            2: ["/trained"],
            3: ["/classified"],
            4: ["/presence"],
            5: ["/overlapping"],
            6: ["/recaptures"],
            7: ["/recaptures"],
            8: ["/recaptures"],
        }
        return step_key_mapping.get(step)

    def refresh_data_viewer_keys(self) -> None:
        db_path = self._resolve_project_db_path()
        if db_path is None or not db_path.exists():
            self.viewer_key_combo.clear()
            self.viewer_status_label.setText("No project database loaded.")
            self.viewer_summary.setPlainText("")
            self.viewer_qc_summary.setPlainText("")
            self.viewer_table.clear()
            self.viewer_table.setRowCount(0)
            self.viewer_table.setColumnCount(0)
            return

        current_key = self.viewer_key_combo.currentText()
        with pd.HDFStore(str(db_path), mode="r") as store:
            all_keys = sorted(store.keys())

        # Filter keys based on current step
        step = self._current_step_index()
        allowed_keys = self._get_available_viewer_keys_for_step(step)
        
        if allowed_keys is not None:
            # On a workflow step: show only allowed keys
            keys = [k for k in all_keys if k in allowed_keys]
        else:
            # On home page: show all keys
            keys = all_keys

        self.viewer_key_combo.clear()
        self.viewer_key_combo.addItems(keys)
        if current_key:
            idx = self.viewer_key_combo.findText(current_key)
            if idx >= 0:
                self.viewer_key_combo.setCurrentIndex(idx)

        key_count = len(keys)
        if allowed_keys is not None:
            self.viewer_status_label.setText(f"Loaded {key_count} HDF key(s) for this step from {db_path.name}")
        else:
            self.viewer_status_label.setText(f"Loaded {key_count} HDF key(s) from {db_path.name}")
        self._set_viewer_key_for_step(self._current_step_index())

    def _summarize_loaded_preview(self, preview: pd.DataFrame, selected_key: str, total_rows: Optional[int]) -> None:
        lines = [f"Key: {selected_key}"]
        if total_rows is not None:
            lines.append(f"Total rows in table: {total_rows}")
        lines.append(f"Preview rows loaded: {len(preview)}")
        lines.append(f"Columns: {len(preview.columns)}")

        if "rec_id" in preview.columns:
            lines.append(f"Preview receivers: {preview['rec_id'].nunique()}")
        if "freq_code" in preview.columns:
            lines.append(f"Preview fish: {preview['freq_code'].nunique()}")
        if "time_stamp" in preview.columns and not preview.empty:
            timestamps = pd.to_datetime(preview["time_stamp"], errors="coerce")
            if timestamps.notna().any():
                lines.append(f"Preview time range: {timestamps.min()} -> {timestamps.max()}")

        self.viewer_summary.setPlainText("\n".join(lines))

    def _select_hdf_rows(
        self,
        store: pd.HDFStore,
        selected_key: str,
        where_clause: Optional[str] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> pd.DataFrame:
        try:
            frame = store.select(selected_key, where=where_clause, start=start, stop=stop)
        except (ValueError, KeyError) as exc:
            if where_clause is None:
                raise
            frame_full = store.select(selected_key)
            try:
                frame = frame_full.query(where_clause)
                if start is not None or stop is not None:
                    frame = frame.iloc[start:stop]
            except Exception as query_exc:  # noqa: BLE001
                raise ValueError(
                    f"Viewer filter could not be applied as HDF where-clause or pandas query: {query_exc}"
                ) from query_exc
            self.log(f"Viewer used in-memory filtering fallback for {selected_key}: {exc}")

        if isinstance(frame, pd.Series):
            frame = frame.to_frame()
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        return frame

    def _summary_where_clause_for_step(self, step: int) -> Optional[str]:
        rec_id = self._current_receiver_context()
        if rec_id and step in {1, 2, 3, 4, 6}:
            return f"rec_id == '{rec_id}'"
        return None

    def _build_stage_qc_summary(
        self,
        selected_key: str,
        frame: pd.DataFrame,
        total_rows: Optional[int],
        sampled: bool,
    ) -> str:
        lines = [f"Stage QC: {selected_key}"]
        if sampled:
            lines.append("Summary scope: sampled preview only (table too large for full scan)")
        elif total_rows is not None:
            lines.append("Summary scope: full selected dataset")

        if frame.empty:
            lines.append("No rows available for the selected stage/filter.")
            return "\n".join(lines)

        lines.append(f"Rows: {len(frame):,}")
        if 'freq_code' in frame.columns:
            lines.append(f"Fish: {frame['freq_code'].nunique():,}")
        if 'rec_id' in frame.columns:
            lines.append(f"Receivers: {frame['rec_id'].nunique():,}")

        if selected_key == "/raw_data":
            orphan_count = None
            if self.project is not None and hasattr(self.project, 'tags') and 'freq_code' in frame.columns:
                master_codes = set(self.project.tags.index.astype(str)) if self.project.tags.index.name == 'freq_code' else set(self.project.tags['freq_code'].astype(str))
                orphan_count = int((~frame['freq_code'].astype(str).isin(master_codes)).sum())
            if orphan_count is not None:
                lines.append(f"Orphan detections: {orphan_count:,}")

        elif selected_key == "/classified":
            if 'test' in frame.columns:
                kept = int((frame['test'] == 1).sum())
                lines.append(f"Classified true detections: {kept:,} ({100 * kept / len(frame):.1f}%)")
            if 'iter' in frame.columns:
                lines.append(f"Latest iteration in view: {frame['iter'].max()}")
            if 'posterior_T' in frame.columns:
                post = pd.to_numeric(frame['posterior_T'], errors='coerce')
                if post.notna().any():
                    lines.append(f"Mean posterior_T: {post.mean():.3f}")

        elif selected_key == "/presence":
            if 'bout_no' in frame.columns:
                bout_counts = frame.groupby(['freq_code', 'rec_id', 'bout_no']).size()
                lines.append(f"Bouts: {len(bout_counts):,}")
                if len(bout_counts) > 0:
                    lines.append(f"Mean detections per bout: {bout_counts.mean():.2f}")

        elif selected_key == "/overlapping":
            if 'overlapping' in frame.columns:
                overlap_n = int((frame['overlapping'] == 1).sum())
                lines.append(f"Marked overlapping: {overlap_n:,}")
            if 'ambiguous_overlap' in frame.columns:
                ambig_n = int((frame['ambiguous_overlap'] == 1).sum())
                lines.append(f"Marked ambiguous: {ambig_n:,}")

        elif selected_key == "/recaptures":
            if 'overlapping' in frame.columns:
                lines.append(f"Remaining overlapping rows: {int((frame['overlapping'] == 1).sum()):,}")
            if 'ambiguous_overlap' in frame.columns:
                lines.append(f"Remaining ambiguous rows: {int((frame['ambiguous_overlap'] == 1).sum()):,}")
            if 'bout_no' in frame.columns:
                lines.append(f"Distinct bouts in view: {frame['bout_no'].nunique():,}")

        if 'time_stamp' in frame.columns:
            timestamps = pd.to_datetime(frame['time_stamp'], errors='coerce')
            if timestamps.notna().any():
                lines.append(f"Time range: {timestamps.min()} -> {timestamps.max()}")

        return "\n".join(lines)

    def load_step_qc_summary(self) -> None:
        db_path = self._resolve_project_db_path()
        if db_path is None or not db_path.exists():
            raise RuntimeError("Project database is not available for QC summary.")

        step = self._current_step_index()
        self._set_viewer_key_for_step(step)
        selected_key = self.viewer_key_combo.currentText().strip()
        if not selected_key:
            raise ValueError("Select an HDF key before generating QC summary.")

        where_clause = self.viewer_where_edit.text().strip() or self._summary_where_clause_for_step(step)
        sampled = False
        with pd.HDFStore(str(db_path), mode='r') as store:
            storer = store.get_storer(selected_key)
            total_rows = getattr(storer, 'nrows', None)
            if where_clause:
                frame = self._select_hdf_rows(store, selected_key, where_clause=where_clause)
            elif total_rows is not None and total_rows > 50000:
                frame = self._select_hdf_rows(store, selected_key, start=0, stop=min(5000, total_rows))
                sampled = True
            else:
                frame = self._select_hdf_rows(store, selected_key)

        self.viewer_qc_summary.setPlainText(
            self._build_stage_qc_summary(selected_key, frame, total_rows, sampled)
        )
        if where_clause and not self.viewer_where_edit.text().strip():
            self.viewer_where_edit.setText(where_clause)
        self.log(f"Generated QC summary for {selected_key} ({len(frame)} row(s) inspected).")

    def _populate_data_viewer_table(self, frame: pd.DataFrame) -> None:
        display_frame = frame.copy()
        if not isinstance(display_frame.index, pd.RangeIndex):
            display_frame = display_frame.reset_index()

        display_frame = display_frame.fillna("")
        self.viewer_table.setRowCount(len(display_frame))
        self.viewer_table.setColumnCount(len(display_frame.columns))
        self.viewer_table.setHorizontalHeaderLabels([str(col) for col in display_frame.columns])

        for row_idx, (_, row) in enumerate(display_frame.iterrows()):
            for col_idx, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.viewer_table.setItem(row_idx, col_idx, item)

        self.viewer_table.resizeColumnsToContents()

    def refresh_data_viewer(self) -> None:
        db_path = self._resolve_project_db_path()
        if db_path is None or not db_path.exists():
            raise RuntimeError("Project database is not available for preview.")

        selected_key = self.viewer_key_combo.currentText().strip()
        if not selected_key:
            raise ValueError("Select an HDF key before loading preview data.")

        start = int(self.viewer_offset_spin.value())
        stop = start + int(self.viewer_limit_spin.value())
        where_clause = self.viewer_where_edit.text().strip() or None

        with pd.HDFStore(str(db_path), mode="r") as store:
            storer = store.get_storer(selected_key)
            total_rows = getattr(storer, "nrows", None)
            preview = self._select_hdf_rows(store, selected_key, where_clause=where_clause, start=start, stop=stop)

        self._populate_data_viewer_table(preview)
        self._summarize_loaded_preview(preview, selected_key, total_rows)
        self.viewer_status_label.setText(
            f"Previewing {selected_key}: rows {start} to {max(start, stop - 1)} | loaded {len(preview)} row(s), {len(preview.columns)} column(s)"
        )
        self.log(f"Loaded data preview for {selected_key} ({len(preview)} row(s)).")

    def load_current_step_view(self) -> None:
        self.refresh_data_viewer_keys()
        self._set_viewer_key_for_step(self._current_step_index())
        self._populate_viewer_filters()
        self.refresh_data_viewer()

    def filter_viewer_to_current_receiver(self) -> None:
        rec_id = self._current_receiver_context()
        if not rec_id:
            self.log("No receiver context is available for the current step.")
            return
        self.viewer_where_edit.setText(f"rec_id == '{rec_id}'")
        self.log(f"Applied viewer filter for receiver {rec_id}.")
        self.refresh_data_viewer()

    def clear_viewer_filter(self) -> None:
        self.viewer_where_edit.clear()
        # Clear all filter dropdowns
        for combo in self.viewer_filter_dropdowns.values():
            combo.setCurrentIndex(0)
        self.log("Cleared all viewer filters.")

    def _on_viewer_key_changed(self) -> None:
        """Populate filter dropdowns when the HDF key changes."""
        self._populate_viewer_filters()

    def _populate_viewer_filters(self) -> None:
        """Populate filter dropdowns with unique values from the selected HDF table."""
        db_path = self._resolve_project_db_path()
        if db_path is None or not db_path.exists():
            return
        
        selected_key = self.viewer_key_combo.currentText().strip()
        if not selected_key:
            # Clear all filter combos if no key is selected
            for combo in self.viewer_filter_dropdowns.values():
                combo.clear()
                combo.addItem("(All)")
            return

        try:
            with pd.HDFStore(str(db_path), mode="r") as store:
                # Get a small sample to identify available columns
                sample = self._select_hdf_rows(store, selected_key, start=0, stop=min(1000, 10000))
            
            # Define filter columns to look for
            filter_cols = ["rec_id", "rec_type", "freq_code", "tag_id", "fish_id"]
            available_cols = [col for col in filter_cols if col in sample.columns]
            
            # Clear existing filter dropdowns and create new ones as needed
            # Remove old dropdowns that are no longer relevant
            old_combos = list(self.viewer_filter_dropdowns.keys())
            for old_col in old_combos:
                if old_col not in available_cols:
                    combo = self.viewer_filter_dropdowns.pop(old_col)
                    # Find and remove from layout
                    for i in range(self.data_viewer_group.layout().count()):
                        widget = self.data_viewer_group.layout().itemAt(i).widget()
                        if widget is combo or (hasattr(widget, 'layout') and combo in widget.children()):
                            break
            
            # Populate filter dropdowns
            filter_row = None
            for i in range(self.data_viewer_group.layout().count()):
                layout_item = self.data_viewer_group.layout().itemAt(i)
                if isinstance(layout_item, QHBoxLayout.__class__):
                    # Check if this is the filter row by looking for "Filters:" label
                    if layout_item.count() > 0:
                        widget = layout_item.itemAt(0).widget()
                        if isinstance(widget, QLabel) and widget.text() == "Filters:":
                            filter_row = layout_item
                            break
            
            if filter_row is None:
                return

            # Populate dropdowns for available columns
            for col in available_cols:
                if col not in self.viewer_filter_dropdowns:
                    # Create new dropdown
                    combo = QComboBox()
                    combo.setMaximumWidth(120)
                    self.viewer_filter_dropdowns[col] = combo
                    
                    # Connect to filter update
                    combo.currentTextChanged.connect(self._apply_filter_from_dropdowns)
                    
                    # Add label + combo to filter row
                    filter_row.addWidget(QLabel(f"{col}:"))
                    filter_row.addWidget(combo)
                
                combo = self.viewer_filter_dropdowns[col]
                combo.blockSignals(True)
                combo.clear()
                combo.addItem("(All)")
                
                # Get unique values from the full table for this column
                try:
                    with pd.HDFStore(str(db_path), mode="r") as store:
                        full_data = self._select_hdf_rows(store, selected_key, start=0, stop=min(10000, 50000))
                    
                    unique_vals = sorted(full_data[col].dropna().unique())
                    for val in unique_vals:
                        combo.addItem(str(val))
                except Exception:
                    pass
                
                combo.blockSignals(False)

        except Exception as e:
            self.log(f"Could not populate filters: {e}")

    def _apply_filter_from_dropdowns(self) -> None:
        """Apply filters from the dropdown selections to the where clause."""
        where_clause = self._build_filter_where_clause()
        if where_clause:
            self.viewer_where_edit.setText(where_clause)
        else:
            self.viewer_where_edit.clear()

    def _build_filter_where_clause(self) -> str:
        """Build a where clause from selected filter dropdowns."""
        conditions = []
        for col, combo in self.viewer_filter_dropdowns.items():
            selected = combo.currentText().strip()
            if selected and selected != "(All)":
                # Check if the value is numeric
                try:
                    float(selected)
                    conditions.append(f"{col} == {selected}")
                except ValueError:
                    conditions.append(f"{col} == '{selected}'")
        
        return " & ".join(f"({c})" for c in conditions) if conditions else ""

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

    def _normalize_rec_type(self, rec_type: str) -> str:
        normalized = rec_type.strip().lower()
        aliases = {
            'vr2': 'vr2',
            'pit': 'pit',
            'pit_multiple': 'pit_multiple',
            'pit-multiple': 'pit_multiple',
            'pit multiple': 'pit_multiple',
        }
        return aliases.get(normalized, normalized)

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
 
    def _clear_plot_viewer(self) -> None:
        self.plot_image_label.clear()
        self.plot_image_label.hide()
        self.plot_viewer_label.show()
        self._plot_figures.clear()
        self._plot_pixmaps.clear()
        self._active_plot_context = None
 
    def _on_plot_selection_changed(self, index: int) -> None:
        if index < 0:
            return
        if not self.plot_selector_combo.isEnabled():
            return
        label = self.plot_selector_combo.currentText()
        if not label:
            return

        db_path = self._resolve_project_db_path()
        rec_type = self.train_rec_type.currentText().strip().lower()
        context_key = (str(db_path) if db_path else "", rec_type)
        if self._active_plot_context != context_key:
            self._plot_figures.clear()
            self._plot_pixmaps.clear()
            self._active_plot_context = context_key
        
        try:
            if label in self._plot_pixmaps:
                self._show_plot_pixmap(self._plot_pixmaps[label])
                return

            fig = self._generate_parameter_plot(label)
            if fig:
                self._plot_figures[label] = fig
                pixmap = self._render_figure_to_pixmap(fig)
                if pixmap is not None:
                    self._plot_pixmaps[label] = pixmap
                    self._show_plot_pixmap(pixmap)
                plt.close(fig)
        except Exception as e:
            self.log(f"Error displaying plot '{label}': {e}")
 
    def _generate_parameter_plot(self, param_name: str):
        """Generate a single parameter histogram on-demand from trained data."""
        try:
            db_path = self._resolve_project_db_path()
            if db_path is None or not db_path.exists():
                self.log("Select an existing project database (.h5) or initialize a project before plotting.")
                return None

            rec_type = self.train_rec_type.currentText().strip()

            trained_full = self._get_trained_data_for_db(db_path)
            if trained_full is None:
                return None

            trained_dat = trained_full.copy()

            if rec_type and "rec_type" in trained_dat.columns:
                trained_dat = trained_dat[trained_dat.rec_type.astype(str).str.lower() == rec_type.lower()]
            elif not rec_type and "rec_type" in trained_dat.columns and not trained_dat.empty:
                # Keep the dropdown in sync when plotting from a loaded DB without project initialization.
                fallback_rec_type = str(trained_dat["rec_type"].dropna().iloc[0]).strip()
                if fallback_rec_type:
                    if self.train_rec_type.findText(fallback_rec_type) < 0:
                        self.train_rec_type.addItem(fallback_rec_type)
                    self.train_rec_type.setCurrentText(fallback_rec_type)
            
            if trained_dat.empty:
                if rec_type:
                    self.log(f"No trained data available for receiver type '{rec_type}'.")
                else:
                    self.log("No trained data available in /trained.")
                return None
            
            trues = trained_dat[trained_dat['detection'] == 1]
            falses = trained_dat[trained_dat['detection'] == 0]
            
            # Define the 5 parameters
            parameters = {
                'Hit Ratio Distribution': {
                    'column': 'hit_ratio',
                    'bins': np.arange(0, 1.05, 0.05),
                    'xlabel': 'Hit Ratio'
                },
                'Consecutive Hit Length': {
                    'column': 'cons_length',
                    'bins': np.arange(0, 12, 1),
                    'xlabel': 'Consecutive Hit Length'
                },
                'Signal Power Distribution': {
                    'column': 'power',
                    'bins': np.arange(0, 110, 10),
                    'xlabel': 'Signal Power'
                },
                'Noise Ratio Distribution': {
                    'column': 'noise_ratio',
                    'bins': np.arange(0, 1.1, 0.1),
                    'xlabel': 'Noise Ratio'
                },
                'Lag Differences': {
                    'column': 'lag_diff',
                    'bins': np.arange(0, 150, 15),
                    'xlabel': 'Lag Differences'
                },
            }
            
            if param_name not in parameters:
                self.log(f"Unknown parameter: {param_name}")
                return None
            
            param = parameters[param_name]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
            ax_false, ax_true = axes
            
            if param['column'] in trues.columns and param['column'] in falses.columns:
                ax_false.hist(
                    falses[param['column']].values,
                    bins=param['bins'],
                    alpha=0.8,
                    color='red',
                    edgecolor='black',
                    linewidth=1,
                )
                ax_true.hist(
                    trues[param['column']].values,
                    bins=param['bins'],
                    alpha=0.8,
                    color='green',
                    edgecolor='black',
                    linewidth=1,
                )

                ax_false.set_title('False Positive', fontsize=11, fontweight='bold')
                ax_true.set_title('True Detection', fontsize=11, fontweight='bold')
                ax_false.set_xlabel(param['xlabel'], fontsize=10)
                ax_true.set_xlabel(param['xlabel'], fontsize=10)
                ax_false.set_ylabel('Count', fontsize=10)
                ax_false.grid(True, alpha=0.3)
                ax_true.grid(True, alpha=0.3)

                fig.suptitle(f"{param_name} ({rec_type})", fontsize=12, fontweight='bold')
                fig.tight_layout()
                
                return fig
            else:
                self.log(f"Column '{param['column']}' not found in trained data.")
                return None
        
        except Exception as e:
            self.log(f"Error generating plot for {param_name}: {e}")
            return None

    def _get_trained_data_for_db(self, db_path: Path) -> Optional[pd.DataFrame]:
        key = str(db_path)
        cached = self._trained_table_cache.get(key)
        if cached is not None:
            return cached

        try:
            trained = pd.read_hdf(str(db_path), key='trained')
        except (KeyError, FileNotFoundError, OSError, ValueError) as exc:
            self.log(f"Could not load trained data from {db_path}: {exc}")
            return None

        self._trained_table_cache[key] = trained
        return trained

    def _render_figure_to_pixmap(self, fig) -> Optional[QPixmap]:
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            pixmap = QPixmap()
            if not pixmap.loadFromData(buf.getvalue()):
                self.log("Failed to load image data from figure.")
                return None
            return pixmap
        finally:
            buf.close()

    def _show_plot_pixmap(self, pixmap: QPixmap) -> None:
        self.plot_viewer_label.hide()
        self.plot_image_label.setPixmap(pixmap.scaledToWidth(500, Qt.SmoothTransformation))
        self.plot_image_label.show()

    def _display_plot(self, fig) -> None:
        try:
            pixmap = self._render_figure_to_pixmap(fig)
            if pixmap is None:
                return
            self._show_plot_pixmap(pixmap)
        except Exception as e:
            self.log(f"Error rendering plot to display: {e}")
 
    def _add_figure_to_viewer(self, fig, label: str = None) -> None:
        """Cache a figure that was just generated (e.g., during training)."""
        if label is None:
            label = f"Plot {len(self._plot_figures) + 1}"
        # Only cache it; don't add to dropdown (dropdown already has all 5 parameters)
        self._plot_figures[label] = fig
        plt.close(fig)
 
    def _capture_and_display_plots(self, fn, *args, **kwargs) -> None:
        self._clear_plot_viewer()
        plt.ioff()
        try:
            # Just run the training function (which generates reporting plots)
            result = fn(*args, **kwargs)
            # GUI plots will be generated on-demand when user selects them from dropdown
        finally:
            plt.close('all')


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

    def _set_busy(self, busy: bool) -> None:
        self.stack.setEnabled(not busy)
        self.cancel_action_btn.setEnabled(busy)

    def _is_cancel_requested(self) -> bool:
        return self._cancel_requested

    def _check_cancel_requested(self) -> None:
        if self._cancel_requested:
            raise ActionCancelled("Cancelled by user request.")

    def cancel_active_action(self) -> None:
        if self._active_thread is None or not self._active_thread.isRunning():
            self.log("No active background action to cancel.")
            return

        self._cancel_requested = True
        self._active_thread.requestInterruption()
        label = self._active_action_label or "Current action"
        self.log(f"Cancellation requested for: {label}")
        self.log("The action will stop at the next safe cancellation checkpoint.")

    def _run_action_async(self, label: str, fn, success_message: Optional[str] = None) -> None:
        if self._active_thread is not None and self._active_thread.isRunning():
            QMessageBox.warning(self, "PyMAST GUI Busy", "Another operation is currently running. Please wait.")
            return

        self.log(f"\n=== {label} ===")
        self.log("Running in background thread...")
        self._cancel_requested = False
        self._active_action_label = label
        self._set_busy(True)

        thread = QThread(self)
        worker = AsyncActionWorker(fn, cancel_check=self._is_cancel_requested)
        worker.moveToThread(thread)

        def _cleanup() -> None:
            thread.quit()
            thread.wait()
            worker.deleteLater()
            thread.deleteLater()
            self._active_worker = None
            self._active_thread = None
            self._active_action_label = None
            self._cancel_requested = False
            self._set_busy(False)

        def _on_success() -> None:
            if success_message:
                self.log(success_message)
            self.log(f"✓ {label} complete")
            _cleanup()

        def _on_error(err: str, tb: str) -> None:
            self.log(f"✗ {label} failed: {err}")
            self.log(tb)
            QMessageBox.critical(self, "PyMAST GUI Error", f"{label} failed:\n{err}")
            _cleanup()

        def _on_cancelled(msg: str) -> None:
            self.log(f"⊘ {label} cancelled: {msg}")
            _cleanup()

        thread.started.connect(worker.run)
        worker.output.connect(self.log)
        worker.finished.connect(_on_success)
        worker.failed.connect(_on_error)
        worker.cancelled.connect(_on_cancelled)

        self._active_thread = thread
        self._active_worker = worker
        thread.start()

    def initialize_project_from_form(self) -> None:
        def _impl() -> None:
            project_dir = self.project_dir_edit.text().strip()
            if not project_dir:
                raise ValueError("Project directory is required.")

            db_name = self.db_name_combo.currentText().strip()
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
            self._refresh_import_receiver_ids()
            self._refresh_train_receiver_ids()
            self._update_session_state_path()
            self.refresh_data_viewer_keys()
            self.log(f"Project DB: {db_path}")
            self.log(f"Loaded tags={len(tag_data)}, receivers={len(receiver_data)}, nodes={len(nodes_data)}")
            self._auto_preview_receiver_network_from_setup()

            if self._session_state_path is not None and self._session_state_path.exists():
                self.load_gui_session()

        self._run_action("Initialize / Reload Project", _impl)

    def run_import(self) -> None:
        proj = self._required_project()
        ant_map = self._parse_literal(self.import_ant_map.toPlainText(), "Antenna map", dict, allow_empty=True)
        if ant_map is None:
            ant_map = None

        rec_id = self.import_rec_id.currentText().strip()
        if not rec_id:
            raise ValueError("Receiver ID is required. Initialize the project to load receiver IDs.")
        rec_type = self._normalize_rec_type(self.import_rec_type.currentText())
        file_dir = self.import_file_dir.text().strip()
        db_dir = self.import_db_dir.text().strip() or proj.db
        scan_time = float(self.import_scan_time.value())
        channels = int(self.import_channels.value())
        ka_format = bool(self.import_ka_format.isChecked())

        def _impl() -> None:
            proj.telem_data_import(
                rec_id=rec_id,
                rec_type=rec_type,
                file_dir=file_dir,
                db_dir=db_dir,
                scan_time=scan_time,
                channels=channels,
                ant_to_rec_dict=ant_map,
                ka_format=ka_format,
            )

        self._run_action_async("Run Import", _impl, success_message=f"Imported receiver {rec_id} from {file_dir}")

    def undo_import(self) -> None:
        rec_id = self.import_rec_id.currentText().strip()
        if not rec_id:
            raise ValueError("Receiver ID is required. Initialize the project to load receiver IDs.")
        self._run_action("Undo Import", lambda: self._required_project().undo_import(rec_id))

    def run_training(self) -> None:
        def _impl() -> None:
            proj = self._required_project()
            rec_id = self.train_rec_id.currentText().strip()
            if not rec_id:
                raise ValueError("Receiver ID is required. Initialize the project to load receiver IDs.")
 
            if self.train_all_fish.isChecked():
                fishes = proj.get_fish(rec_id=rec_id)
                if isinstance(fishes, (list, tuple)):
                    fishes = list(fishes)
                else:
                    if isinstance(fishes, np.ndarray):
                        fishes = fishes.tolist()
                    else:
                        fishes = list(fishes) if fishes is not None else []
            else:
                fishes = [x.strip() for x in self.train_fish_codes.text().split(",") if x.strip()]
 
            if len(fishes) == 0:
                raise ValueError("No fish found to train.")
 
            self.log(f"Training {len(fishes)} fish at {rec_id}...")
            for fish in fishes:
                self._check_cancel_requested()
                proj.train(fish, rec_id)
                QApplication.processEvents()
 
            if self.train_summary.isChecked():
                self._capture_and_display_plots(
                    proj.training_summary,
                    self.train_rec_type.currentText().strip(),
                    site=[rec_id]
                )
 
        self._run_action("Run Training", _impl)

    def undo_training(self) -> None:
        rec_id = self.train_rec_id.currentText().strip()
        if not rec_id:
            raise ValueError("Receiver ID is required. Initialize the project to load receiver IDs.")
        self._run_action("Undo Training", lambda: self._required_project().undo_training(rec_id))

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
        proj = self._required_project()
        if self.bout_all_receivers.isChecked():
            rec_ids = list(proj.receivers.index)
        else:
            rec_id = self.bout_rec_id.text().strip()
            if not rec_id:
                raise ValueError("Receiver ID is required when not running all receivers.")
            rec_ids = [rec_id]

        eps_multiplier = int(self.bout_eps.value())
        lag_window = int(self.bout_lag.value())
        visualize = bool(self.bout_visualize.isChecked())

        def _impl() -> None:
            for rec_id in rec_ids:
                self._check_cancel_requested()
                b = bout(
                    radio_project=proj,
                    rec_id=rec_id,
                    eps_multiplier=eps_multiplier,
                    lag_window=lag_window,
                )
                b.presence()
                if visualize:
                    b.visualize_bout_lengths()

        self._run_action_async("Run Bout Detection", _impl, success_message=f"Processed bout detection for {len(rec_ids)} receiver(s).")

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
        proj = self._required_project()
        nodes, edges = self._resolve_nodes_edges()
        method = self.overlap_method.currentText()
        p_value_threshold = float(self.overlap_p.value())
        effect_size_threshold = float(self.overlap_effect.value())
        power_threshold = float(self.overlap_power.value())
        min_detections = int(self.overlap_min_det.value())
        bout_expansion = int(self.overlap_expand.value())

        conf_raw = self.overlap_conf.text().strip()
        confidence = float(conf_raw) if conf_raw else None

        def _impl() -> None:
            overlap_obj = overlap_reduction(nodes=nodes, edges=edges, radio_project=proj)
            overlap_obj.unsupervised_removal(
                method=method,
                p_value_threshold=p_value_threshold,
                effect_size_threshold=effect_size_threshold,
                power_threshold=power_threshold,
                min_detections=min_detections,
                bout_expansion=bout_expansion,
                confidence_threshold=confidence,
            )

        self._run_action_async("Run Unsupervised Overlap", _impl, success_message=f"Overlap removal completed for {len(nodes)} node(s).")

    def run_overlap_nested(self) -> None:
        proj = self._required_project()
        nodes, edges = self._resolve_nodes_edges()

        def _impl() -> None:
            overlap_obj = overlap_reduction(nodes=nodes, edges=edges, radio_project=proj)
            overlap_obj.nested_doll()

        self._run_action_async("Run Nested Doll", _impl, success_message=f"Nested doll overlap completed for {len(edges)} edge(s).")

    def undo_overlap(self) -> None:
        self._run_action("Undo Overlap", lambda: self._required_project().undo_overlap())

    def run_recaptures(self) -> None:
        proj = self._required_project()
        export = bool(self.recap_export.isChecked())
        pit_study = bool(self.recap_pit_study.isChecked())

        def _impl() -> None:
            proj.make_recaptures_table(export=export, pit_study=pit_study)

        self._run_action_async("Create Recaptures Table", _impl, success_message="Recaptures table generation completed.")

    def undo_recaptures(self) -> None:
        self._run_action("Undo Recaptures", lambda: self._required_project().undo_recaptures())

    def run_tte(self) -> None:
        proj = self._required_project()
        node_to_state = self._parse_literal(self.tte_node_map.toPlainText(), "receiver_to_state", dict)

        adjacency = None
        if self.tte_adjacency.text().strip():
            adjacency = self._parse_literal(self.tte_adjacency.text(), "adjacency_filter", list)

        unknown_state = None
        if self.tte_unknown_state.text().strip():
            unknown_state = int(self.tte_unknown_state.text().strip())

        initial_state_release = bool(self.tte_initial_release.isChecked())
        last_presence_time0 = bool(self.tte_last_presence.isChecked())
        hit_ratio_filter = bool(self.tte_hit_ratio_filter.isChecked())
        cap_loc = self._none_if_empty(self.tte_cap_loc.text())
        rel_loc = self._none_if_empty(self.tte_rel_loc.text())
        species = self._none_if_empty(self.tte_species.text())
        rel_date = self._none_if_empty(self.tte_rel_date.text())
        recap_date = self._none_if_empty(self.tte_recap_date.text())
        bucket_length_min = int(self.tte_bucket_min.value())

        def _impl() -> None:
            tte = formatter.time_to_event(
                receiver_to_state=node_to_state,
                project=proj,
                initial_state_release=initial_state_release,
                last_presence_time0=last_presence_time0,
                hit_ratio_filter=hit_ratio_filter,
                cap_loc=cap_loc,
                rel_loc=rel_loc,
                species=species,
                rel_date=rel_date,
                recap_date=recap_date,
            )
            tte.data_prep(
                project=proj,
                unknown_state=unknown_state,
                bucket_length_min=bucket_length_min,
                adjacency_filter=adjacency,
            )
            tte.summary()
            self.tte_obj = tte

        self._run_action_async("Run TTE Data Prep", _impl, success_message="TTE data preparation completed.")

    def run_cjs(self) -> None:
        proj = self._required_project()
        receiver_to_recap = self._parse_literal(self.cjs_map.toPlainText(), "receiver_to_recap", dict)

        output_ws = self.cjs_output_ws.text().strip() or proj.output_dir
        os.makedirs(output_ws, exist_ok=True)
        model_name = self.cjs_model_name.text().strip()
        if not model_name:
            raise ValueError("model_name is required.")

        species = self._none_if_empty(self.cjs_species.text())
        rel_loc = self._none_if_empty(self.cjs_rel_loc.text())
        cap_loc = self._none_if_empty(self.cjs_cap_loc.text())
        initial_recap_release = bool(self.cjs_initial_release.isChecked())
        csv_path = os.path.join(output_ws, f"{model_name}.csv")
        inp_path = os.path.join(output_ws, f"{model_name}.inp")

        def _impl() -> None:
            cjs = formatter.cjs_data_prep(
                receiver_to_recap=receiver_to_recap,
                project=proj,
                species=species,
                rel_loc=rel_loc,
                cap_loc=cap_loc,
                initial_recap_release=initial_recap_release,
            )
            cjs.input_file(model_name, output_ws)

            if hasattr(cjs, 'cross'):
                cjs.cross.to_csv(csv_path)
            else:
                raise RuntimeError("CJS export failed: cross-tab output is unavailable.")

            with open(inp_path, 'w', encoding='utf-8') as f:
                f.write(str(cjs.inp))

            self.cjs_obj = cjs

        success_msg = f"CJS outputs written to {output_ws}\n  CSV: {csv_path}\n  INP: {inp_path}"
        self._run_action_async("Run CJS Export", _impl, success_message=success_msg)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        # Defer so Qt finishes the initial layout before we force equal halves.
        QTimer.singleShot(0, self._equalize_splitters)

    def _equalize_splitters(self) -> None:
        for sp in (self._left_splitter, self._right_splitter):
            total = sp.height()
            half = total // 2
            sp.setSizes([half, half])
        # Start with the plot panel collapsed.
        total = self._right_splitter.height()
        self._right_splitter.setSizes([0, total])

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            if self._resolve_project_db_path() is not None:
                self.save_gui_session()
        except Exception as exc:  # noqa: BLE001
            self.log(f"GUI session save skipped on close: {exc}")
        super().closeEvent(event)


def _find_repo_root() -> Path:
    module_dir = Path(__file__).resolve().parent
    return module_dir.parent


def _install_unhandled_exception_logging(repo_root: Path) -> None:
    crash_log = repo_root / "logs" / "gui_crash.log"
    prior_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_traceback):
        stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        formatted = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        try:
            crash_log.parent.mkdir(parents=True, exist_ok=True)
            with crash_log.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[{stamp}] Unhandled GUI exception\n")
                handle.write(formatted)
                handle.write("\n")
        except Exception:  # noqa: BLE001
            pass

        print(formatted, file=sys.stderr)
        prior_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = _hook


def _install_fault_handler_logging(repo_root: Path) -> None:
    global _FAULT_LOG_HANDLE
    fault_log = repo_root / "logs" / "gui_fault.log"
    fault_log.parent.mkdir(parents=True, exist_ok=True)
    _FAULT_LOG_HANDLE = fault_log.open("a", encoding="utf-8")
    stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    _FAULT_LOG_HANDLE.write(f"\n[{stamp}] GUI process started\n")
    _FAULT_LOG_HANDLE.flush()
    faulthandler.enable(_FAULT_LOG_HANDLE, all_threads=True)


def main() -> None:
    repo_root = _find_repo_root()
    _install_fault_handler_logging(repo_root)
    _install_unhandled_exception_logging(repo_root)
    app = QApplication(sys.argv)

    window = WorkflowWindow(repo_root)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
