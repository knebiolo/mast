# Infrastructure Summary

## Overview

PyMAST is a Python package for radio telemetry data processing and movement analysis. The repository is structured around a core `pymast/` package with supporting documentation, scripts, examples, and tests.

## Repository Layout

- `pymast/` - Core package modules (parsers, naive Bayes classifier, overlap removal, formatter, etc.)
- `tests/` - Pytest test suite
- `docs/` - Documentation (tutorials and API reference)
- `scripts/` - End-to-end and helper scripts
- `examples/` - Example notebooks and sample usage
- `data/` - Sample data files and inputs
- `.ai_journal/` - Assistant journal and long-term notes (not for version control)

## Data Flow (High Level)

1. Raw receiver files are imported via parsers into HDF5.
2. Naive Bayes classification filters false positives.
3. Bouts and overlap removal refine presence windows.
4. Recaptures tables and formatters export to CJS/LRDR/TTE for R/MARK.

## Tooling and Configuration

- Build metadata: `pyproject.toml`
- Tests: `pytest.ini` with markers for unit/integration/slow tests
- Optional dev dependencies listed under `project.optional-dependencies.dev`

## Notes

- Large intermediate datasets are stored in HDF5 files created per project.
- Output artifacts (figures, recaptures, exports) are generated in project output folders created by `radio_project`.
