# Changelog

All notable changes to PyMAST (Movement Analysis Software for Telemetry) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- **Visualization Enhancements**:
  - Bout length visualization with 4-panel analysis
  - Comprehensive 8-panel overlap analysis (network, temporal, power, etc.)
  - Posterior ratio analysis for overlap quality assessment
- **Bout Spatial Filter**: Automatically identifies temporally overlapping bouts across receivers
- **Enhanced HDF5 Storage**: Conditionally includes power and posterior columns in overlapping table

### Changed
- **README.md**: Streamlined with clear navigation to documentation
- **Adjacency Filter**: Fixed bugs and added summary statistics output
- **Overlap Resolution**: Moved bout spatial filter to correct module (overlap_removal.py)

### Fixed
- **Adjacency Filter Bugs**:
  - Fixed assignment operator (== vs =) causing silent failure
  - Fixed tuple vs list bug in carryforward logic
  - Reduced excessive print statement spam
  - Added summary statistics (transitions removed, percentage filtered)
- **Documentation Syntax**: Removed leftover comment lines causing compile errors
- **Bout Visualization**: Fixed calculation of bout summaries from presence_df

### Removed
- Temporary development documentation files (REFACTOR_STATUS.md, OVERLAP_REMOVAL_ANALYSIS.md, etc.)
- Backup file (overlap_removal_backup.py)
- Old docstring remnants causing syntax errors

## [1.0.0] - 2025-10-06

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
