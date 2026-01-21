"""
Integration test for initial_state_release=True workflows.
Requires a local project directory supplied via environment variable.
"""

from pathlib import Path
import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pymast.radio_project import radio_project
from pymast import formatter

PROJECT_ENV = "PYMAST_TEST_PROJECT_DIR"
PROJECT_DIR = os.environ.get(PROJECT_ENV)

if not PROJECT_DIR:
    pytest.skip(f"Set {PROJECT_ENV} to run initial_state_release tests.", allow_module_level=True)

PROJECT_PATH = Path(PROJECT_DIR)
if not PROJECT_PATH.exists():
    pytest.skip(f"{PROJECT_ENV} points to missing path: {PROJECT_PATH}", allow_module_level=True)


def _require_file(path):
    if not path.exists():
        pytest.skip(f"Missing required project file: {path}", allow_module_level=True)


def test_initial_state_release_flow():
    _require_file(PROJECT_PATH / "tblMasterTag.csv")
    _require_file(PROJECT_PATH / "tblMasterReceiver.csv")
    _require_file(PROJECT_PATH / "tblNodes.csv")

    tag_data = pd.read_csv(PROJECT_PATH / "tblMasterTag.csv")
    receiver_data = pd.read_csv(PROJECT_PATH / "tblMasterReceiver.csv")
    nodes_data = pd.read_csv(PROJECT_PATH / "tblNodes.csv")

    project = radio_project(
        str(PROJECT_PATH),
        os.environ.get("PYMAST_TEST_DB_NAME", "pymast_test"),
        5,
        1,
        tag_data,
        receiver_data,
        nodes_data,
    )

    states = {
        "R1696": 1,
        "R1699-1": 2,
        "R1699-2": 3,
        "R1698": 4,
        "R1699-3": 5,
        "R1695": 5,
        "R0004": 6,
        "R0005": 6,
        "R0001": 7,
        "R0002": 7,
        "R0003": 8,
    }

    tte = formatter.time_to_event(
        states,
        project,
        initial_state_release=True,
        last_presence_time0=False,
        cap_loc=None,
        rel_loc=None,
        species=None,
        rel_date=None,
        recap_date=None,
    )

    tte.data_prep(project)
    summary = tte.summary()

    assert not tte.master_state_table.empty
    assert "unique_fish_count" in summary
