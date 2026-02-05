# Changelog

All notable changes to PyMAST (Movement Analysis Software for Telemetry) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v1.1.0
- Interactive UI
- R package wrapper

---
## [1.0.5] - 2026-02-05

### Fixed
- **Critical Import Fix**: Resolved `InvalidVersion` error when dask package has malformed version string
  - Moved all `dask` module imports from module-level to function-level (lazy loading)
  - Prevents import failures when `dask.__version__ = 'unknown'` or when dask/dask-ml have version conflicts
  - Affected modules: `pymast.overlap_removal`, `pymast.radio_project`
  - Package now imports successfully even with broken dask installations
  - Dask is only imported when `make_recaptures_table()` is called

### Technical Details
- Root cause: Some Anaconda environments have dask packages with `__version__ = 'unknown'`
- When `dask-ml` was present (from v1.0.3 or earlier), it would fail parsing dask version during module import
- Solution: Deferred all dask imports until runtime when specific functions need them
- Backward compatible: All existing code continues to work unchanged

---
## [1.0.5] - 2026-02-05

### Fixed
- **Critical Import Fix**: Resolved `InvalidVersion` error when dask package has malformed version string
  - Moved all `dask` module imports from module-level to function-level (lazy loading)
  - Prevents import failures when `dask.__version__ = 'unknown'` or when dask/dask-ml have version conflicts
  - Affected modules: `pymast.overlap_removal`, `pymast.radio_project`
  - Package now imports successfully even with broken dask installations
  - Dask is only imported when `make_recaptures_table()` is called

### Technical Details
- Root cause: Some Anaconda environments have dask packages with `__version__ = 'unknown'`
- When `dask-ml` was present (from v1.0.3 or earlier), it would fail parsing dask version during module import
- Solution: Deferred all dask imports until runtime when specific functions need them
- Backward compatible: All existing code continues to work unchanged

---

## [1.0.4] - 2026-02-04

### Fixed
- Removed hard runtime import failures on systems without `dask-ml`.
- Removed `dask-ml` from package dependency manifests (`pyproject.toml`, `requirements.txt`, `environment.yml`).
- Updated installation smoke tests to validate `scikit-learn` availability instead of `dask-ml`.
- Updated docs to reflect that `dask-ml` is no longer required for installation.

---

## [1.0.3] - 2026-02-03

### Fixed
- **Critical Installation Fix**: Ensured `dask-ml` dependency is always installed when using `pip install pymast`
  - Raised minimum Python version from 3.8 to 3.9 (required for modern dask-ml compatibility)
  - Updated `dask-ml` version constraint to `>=2022.1.22` for better stability
  - Removed silent fallback that masked missing dask-ml installation
  - Added clear error messages when dask-ml is missing, guiding users to correct Python version
  - Updated Python classifiers in `pyproject.toml` to reflect 3.9-3.12 support

### Added
- **Installation smoke test**: `tests/test_installation_smoke.py` verifies all critical dependencies install correctly
- **Improved documentation**: Added Spyder/Anaconda installation guidance to README
  - Recommend `python -m pip install pymast` to ensure correct environment
  - Added Python version requirement (3.9+) prominently in README

### Changed
- Updated badge in README from Python 3.8+ to Python 3.9+
- Synchronized `requirements.txt` with `pyproject.toml` dependencies

---

## [1.0.2] - 2026-02-02

### Added
- **Import Statistics**: `telem_data_import()` now displays comprehensive statistics after import:
  - Total detection count
  - Mean detections per file
  - Unique tag count
  - Tag table validation: checks for duplicate freq_codes in the master tags table (configuration error)
  - Orphan tags check: identifies tags detected but not defined in the tags table
  - Time coverage: start/end dates, duration, and detection rate per hour
- Statistics are formatted for readability and printed to console (visible without logging configuration)
- Both print() and logger output for maximum visibility
- Graceful error handling if statistics cannot be calculated

---

## [1.0.1] - 2026-01-28

### Fixed
- PIT parser: allow fixed-width Biomark exports to map receivers via `Reader ID` when `ant_to_rec_dict` is provided.
- PIT parser: clearer error message when no antenna/reader column is found in fixed-width files.

---

## [1.0.0] - 2025-11-24

**First Official Public Release** - Implementing peer-reviewed algorithms from Nebiolo & Castro-Santos (2024)

### Added

#### Documentation & Usability
- **Comprehensive Documentation**: All modules now have Google/NumPy style docstrings
  - `parsers.py`: Complete documentation for all 8 parser functions
  - `predictors.py`: Documentation for all predictor functions
  - `naive_bayes.py`: Full module and function documentation
  - `radio_project.py`: Enhanced module overview with HDF5 structure
  - `overlap_removal.py`: Detailed bout detection and overlap resolution docs
  - `fish_history.py`: Enhanced class documentation
  - All modules accessible via Python `help()` system
- **User Guides**:
  - `GETTING_STARTED.md`: Complete beginner's guide with 5-step workflow
  - `GITHUB_DESKTOP_GUIDE.md`: Step-by-step merge instructions
  - `PUBLICATIONS.md`: Track academic papers using PyMAST
  - `RELEASE_CHECKLIST.md`: Standardized release process
  - `RELEASE_NOTES_v1.0.0.md`: Comprehensive release documentation
- **Testing Framework**:
  - Automated test suite with pytest
  - `autotest.py`: One-click testing for non-developers
  - CI/CD pipeline with GitHub Actions
  - Test coverage reporting

#### Features
- **Visualization Enhancements**:
  - Bout length visualization with 4-panel analysis
  - Comprehensive 8-panel overlap analysis (network, temporal, power, etc.)
  - Posterior ratio analysis for overlap quality assessment
- **Bout Spatial Filter**: Automatically identifies temporally overlapping bouts across receivers
- **Enhanced HDF5 Storage**: Conditionally includes power and posterior columns in overlapping table
- **Logging Support**: Comprehensive logging throughout codebase
- **Input Validation**: Enhanced error handling and data validation

### Changed
- **README.md**: 
  - Streamlined with clear navigation to documentation
  - Added peer-reviewed publication citation prominently
  - Updated badges and installation instructions
- **Adjacency Filter**: Fixed bugs and added summary statistics output
- **Overlap Resolution**: Moved bout spatial filter to correct module (overlap_removal.py)
- **Python Version**: Updated minimum requirement from 3.5 to 3.8
- **Dependencies**: Modernized to current stable versions (numpy≥1.20, pandas≥1.3, etc.)
- **Package Name**: Standardized as 'pymast' across all platforms
- **Database Backend**: Migrated from SQLite to HDF5 for performance

### Fixed
- **Adjacency Filter Bugs**:
  - Fixed assignment operator (== vs =) causing silent failure
  - Fixed tuple vs list bug in carryforward logic
  - Reduced excessive print statement spam
  - Added summary statistics (transitions removed, percentage filtered)
- **Documentation Syntax**: Removed leftover comment lines causing compile errors
- **Bout Visualization**: Fixed calculation of bout summaries from presence_df
- **Import Path Resolution**: Fixed module import issues in example scripts
- **Dependency Conflicts**: Resolved scikit-learn version mismatches

### Removed
- Temporary development documentation files (REFACTOR_STATUS.md, OVERLAP_REMOVAL_ANALYSIS.md, etc.)
- Backup file (overlap_removal_backup.py)
- Old docstring remnants causing syntax errors
- **Private PyPI Index**: Removed hard-coded private repository references
- **Hard-coded Paths**: Removed development-specific file paths from code
- **SQLite Support**: Fully replaced by HDF5 (see Migration Guide in RELEASE_NOTES)
- **Python 2.x Support**: Dropped legacy Python versions
- **Profanity**: Cleaned up debug statements and code comments

### Breaking Changes
- **Database Format**: SQLite → HDF5 (requires data migration)
- **Function Signatures**: Several functions now use `radio_project` object instead of database connection
- **Python Version**: Minimum version now 3.8 (was 3.5)
- **Import Paths**: Some module reorganization (see Migration Guide)

---

## [0.0.6] - 2024-XX-XX (Internal Release)

### Added
- First official external release
- Comprehensive documentation and README
- Proper Python package structure
- .gitignore for Python projects
- CHANGELOG.md for version tracking
- CONTRIBUTING.md for external contributors
- Proper requirements.txt without private dependencies
- Input validation and error handling improvements
- Logging support throughout the codebase

### Changed
- Updated from v0.0.6 to v1.0.0 for external release
- Renamed from ABTAS to MAST throughout codebase
- Updated Python requirement from 3.5+ to 3.8+
- Modernized dependency versions (numpy, pandas, etc.)
- Removed hard-coded file paths from example scripts
- Improved code documentation with docstrings
- Cleaned up debug statements and profanity in code
- Fixed sklearn → scikit-learn in dependencies
- Updated setup.py with modern packaging standards

### Removed
- Private PyPI index reference in requirements.txt
- Hard-coded development paths from scripts
- Outdated references to SQLite (now using HDF5)
- References to Bitbucket (now on GitHub)

### Fixed
- Package installation issues
- Import path resolution in example scripts
- Dependency version conflicts
- Documentation inconsistencies

## [0.0.6] - Previous Internal Release

### Note
Previous versions (0.0.1 - 0.0.6) were internal releases at Kleinschmidt Associates.
Version history prior to 1.0.0 is available in Git commit history.
