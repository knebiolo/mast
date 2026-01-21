"""
Tests for PIT parsers that require local data files.
"""

from pathlib import Path
import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pymast import parsers

PIT_ENV = "PYMAST_TEST_PIT_FILE"
PIT_FILE = os.environ.get(PIT_ENV)

if not PIT_FILE:
    pytest.skip(f"Set {PIT_ENV} to run PIT parser tests.", allow_module_level=True)

PIT_PATH = Path(PIT_FILE)
if not PIT_PATH.exists():
    pytest.skip(f"{PIT_ENV} points to missing file: {PIT_PATH}", allow_module_level=True)

REC_ID = os.environ.get("PYMAST_TEST_PIT_REC_ID", "R0001")

try:
    SKIPROWS = int(os.environ.get("PYMAST_TEST_PIT_SKIPROWS", "6"))
except ValueError as exc:
    raise ValueError("PYMAST_TEST_PIT_SKIPROWS must be an integer") from exc


def test_pit_parser(tmp_path):
    test_db = tmp_path / "pit_parser_test.h5"

    parsers.PIT(
        file_name=str(PIT_PATH),
        db_dir=str(test_db),
        rec_id=REC_ID,
        study_tags=[],
        skiprows=SKIPROWS,
        scan_time=1,
        channels=1,
        rec_type="PIT",
    )

    with pd.HDFStore(test_db, "r") as store:
        assert "/raw_data" in store.keys()
        data = store["raw_data"]
    assert not data.empty
    assert (data["rec_id"] == REC_ID).any()
