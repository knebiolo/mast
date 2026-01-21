#!/usr/bin/env python3
"""
Unified PIT parser tests for single and multi-antenna modes.
Requires local test files supplied via environment variables.
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

SINGLE_ENV = "PYMAST_TEST_PIT_FILE"
CSV_ENV = "PYMAST_TEST_PIT_CSV_FILE"

SINGLE_FILE = os.environ.get(SINGLE_ENV)
CSV_FILE = os.environ.get(CSV_ENV)


def _require_file(env_var, file_path):
    if not file_path:
        pytest.skip(f"Set {env_var} to run this test.", allow_module_level=False)
    path = Path(file_path)
    if not path.exists():
        pytest.skip(f"{env_var} points to missing file: {path}", allow_module_level=False)
    return path


def test_unified_pit_single():
    temp_dir = Path(".pytest_cache")
    temp_dir.mkdir(exist_ok=True)
    pit_path = _require_file(SINGLE_ENV, SINGLE_FILE)
    test_db = temp_dir / "pit_single_test.h5"

    parsers.PIT(
        file_name=str(pit_path),
        db_dir=str(test_db),
        rec_id="R0001",
        study_tags=None,
        skiprows=6,
        rec_type="PIT",
    )

    with pd.HDFStore(test_db, "r") as store:
        assert "/raw_data" in store.keys()
        data = store["raw_data"]
    assert not data.empty


def test_unified_pit_multi():
    temp_dir = Path(".pytest_cache")
    temp_dir.mkdir(exist_ok=True)
    csv_path = _require_file(CSV_ENV, CSV_FILE)
    test_db = temp_dir / "pit_multi_test.h5"
    ant_mapping = {1: "R0001", 2: "R0002", 3: "R0003", 4: "R0004", 5: "R0005"}

    parsers.PIT(
        file_name=str(csv_path),
        db_dir=str(test_db),
        rec_id=None,
        study_tags=None,
        skiprows=0,
        rec_type="PIT_MultiAntenna",
    )

    parsers.PIT_Multiple(
        file_name=str(csv_path),
        db_dir=str(test_db),
        ant_to_rec_dict=ant_mapping,
        rec_type="PIT_Multiple_Legacy",
    )

    with pd.HDFStore(test_db, "r") as store:
        assert "/raw_data" in store.keys()
        data = store["raw_data"]
    assert not data.empty
    assert (data["rec_type"].isin(["PIT_MultiAntenna", "PIT_Multiple_Legacy"])).any()
