#!/usr/bin/env python3
"""
Simple test for the unified PIT parser CSV functionality.
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
    pytest.skip(f"Set {CSV_ENV} to run PIT CSV parser tests.", allow_module_level=True)

CSV_PATH = Path(CSV_FILE)
if not CSV_PATH.exists():
    pytest.skip(f"{CSV_ENV} points to missing file: {CSV_PATH}", allow_module_level=True)


def test_csv_pit_parser(tmp_path):
    test_db = tmp_path / "test_csv_pit.h5"
    ant_mapping = {1: "R0001", 2: "R0002", 3: "R0003", 4: "R0004", 5: "R0005"}

    parsers.PIT(
        file_name=str(CSV_PATH),
        db_dir=str(test_db),
        rec_id=None,  # Not used in multi-antenna mode
        study_tags=None,
        skiprows=0,  # Auto-detected for CSV
        rec_type="PIT_CSV_Test",
        ant_to_rec_dict=ant_mapping,
    )

    with pd.HDFStore(test_db, "r") as store:
        assert "/raw_data" in store.keys()
        data = store["raw_data"]
    assert not data.empty
    assert "rec_id" in data.columns
