"""
Tests for PIT_Multiple parser with Biomark multi-antenna format.
Requires a local CSV file supplied via environment variable.
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

CSV_ENV = "PYMAST_TEST_PIT_CSV_FILE"
CSV_FILE = os.environ.get(CSV_ENV)

if not CSV_FILE:
    pytest.skip(f"Set {CSV_ENV} to run PIT_Multiple parser tests.", allow_module_level=True)

CSV_PATH = Path(CSV_FILE)
if not CSV_PATH.exists():
    pytest.skip(f"{CSV_ENV} points to missing file: {CSV_PATH}", allow_module_level=True)


def test_pit_multiple_parser(tmp_path):
    test_db = tmp_path / "pit_multiple_test.h5"
    antennae_to_rec_id = {
        1: "R0001",
        2: "R0002",
        3: "R0003",
        4: "R0004",
        5: "R0005",
    }

    parsers.PIT_Multiple(
        file_name=str(CSV_PATH),
        db_dir=str(test_db),
        ant_to_rec_dict=antennae_to_rec_id,
        study_tags=[],
        skiprows=0,
        scan_time=1,
        channels=1,
        rec_type="PIT_Multiple",
    )

    with pd.HDFStore(test_db, "r") as store:
        assert "/raw_data" in store.keys()
        data = store["raw_data"]
    assert not data.empty
    assert (data["rec_type"] == "PIT_Multiple").any()
