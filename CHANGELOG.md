# Changelog

All notable changes to MAST (Movement Analysis Software for Telemetry) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
