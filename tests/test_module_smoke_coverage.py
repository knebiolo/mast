"""Additional smoke tests to ensure every core module is exercised at least at import level."""

from importlib import import_module

import pandas as pd
import pytest

from pymast.logger import setup_logging
from pymast.validation import ValidationError, validate_tag_data


CORE_MODULES = [
    "pymast.fish_history",
    "pymast.formatter",
    "pymast.logger",
    "pymast.naive_bayes",
    "pymast.overlap_removal",
    "pymast.parsers",
    "pymast.predictors",
    "pymast.radio_project",
    "pymast.validation",
]


@pytest.mark.smoke
@pytest.mark.unit
def test_core_modules_importable_explicitly():
    """Import each core module directly so coverage includes every package module."""
    for module_name in CORE_MODULES:
        module = import_module(module_name)
        assert module is not None, f"Failed to import {module_name}"


@pytest.mark.unit
def test_logger_setup_writes_file(tmp_path):
    """Verify logger setup creates both console and file handlers."""
    log_file = tmp_path / "pymast.log"
    logger = setup_logging(log_file=str(log_file))
    logger.info("logger smoke test")

    assert log_file.exists()
    assert len(logger.handlers) >= 2


@pytest.mark.unit
def test_validate_tag_data_happy_path():
    """Validation accepts a minimal valid tag table."""
    tag_data = pd.DataFrame(
        {
            "freq_code": ["166.380 7"],
            "pulse_rate": [5.0],
            "tag_type": ["study"],
            "rel_date": [pd.Timestamp("2024-05-15 00:00:00")],
            "cap_loc": ["Tailrace"],
            "rel_loc": ["Tailrace"],
        }
    )

    assert validate_tag_data(tag_data) is True


@pytest.mark.unit
def test_validate_tag_data_invalid_type_raises():
    """Validation fails loudly when tag_type values are invalid."""
    tag_data = pd.DataFrame(
        {
            "freq_code": ["166.380 7"],
            "pulse_rate": [5.0],
            "tag_type": ["invalid_type"],
            "rel_date": [pd.Timestamp("2024-05-15 00:00:00")],
            "cap_loc": ["Tailrace"],
            "rel_loc": ["Tailrace"],
        }
    )

    with pytest.raises(ValidationError, match="Invalid tag_type values"):
        validate_tag_data(tag_data)


@pytest.mark.unit
def test_fish_history_symbol_available():
    """Fish history module exports the expected fish_history class."""
    module = import_module("pymast.fish_history")
    assert hasattr(module, "fish_history")
    assert module.fish_history is not None
