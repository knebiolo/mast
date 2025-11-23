# 🧪 PyMAST Testing Quick Reference

## 🚀 Quick Commands

### Run Tests Locally

```bash
# Quick smoke tests (< 5 sec)
python run_tests.py --quick

# Unit tests with coverage (< 30 sec)
python run_tests.py --unit --coverage

# Full test suite with HTML report
python run_tests.py --all --html

# Watch mode (rerun on file changes)
python run_tests.py --watch --unit
```

### Using pytest Directly

```bash
# All tests
pytest

# Fast unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Specific module
pytest -m classifier

# With coverage
pytest --cov=pymast --cov-report=html

# Verbose output
pytest -v

# Show print statements
pytest -s

# Stop on first failure
pytest -x
```

## 📊 Test Categories

| Marker | Description | Speed | Database Required |
|--------|-------------|-------|-------------------|
| `smoke` | Critical path checks | < 0.1s | No |
| `unit` | Isolated function tests | < 1s | No |
| `integration` | Component interaction | 1-10s | Yes |
| `slow` | Long-running tests | > 10s | Maybe |
| `classifier` | Naive Bayes tests | < 1s | No |
| `overlap` | Overlap removal tests | 1-5s | Yes |
| `bout` | Bout detection tests | < 1s | No |
| `parser` | Data parser tests | < 1s | No |

## 🎯 Common Workflows

### Before Commit
```bash
pytest -m smoke  # Quick sanity check
```

### During Development
```bash
python run_tests.py --watch --unit  # Auto-rerun on changes
```

### Before Pull Request
```bash
python run_tests.py --all --coverage --html
# Check htmlcov/index.html for coverage gaps
```

### Debug Failed Test
```bash
# Run single test with debugging
pytest tests/test_basic.py::TestPredictors::test_noise_ratio -vv -s

# Drop into debugger on failure
pytest --pdb
```

## 📁 File Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_basic.py            # Core functionality
├── test_overlap_*.py        # Overlap tests
├── run_unit_tests.py        # Legacy runner
└── data/                    # Test data
```

## 🔧 Setup

### Install Test Dependencies
```bash
pip install pytest pytest-cov pytest-benchmark pytest-watch
```

### Run First Test
```bash
pytest -v
```

## 📈 Coverage Goals

- **Predictors**: > 90%
- **Naive Bayes**: > 90%
- **Overlap Removal**: > 85%
- **Parsers**: > 80%
- **Formatter**: > 80%

## 🤖 Automated Testing

### GitHub Actions
- Runs automatically on push/PR
- Tests Python 3.8-3.12 on Windows/Mac/Linux
- Daily scheduled runs at 2 AM UTC
- Coverage reports uploaded to Codecov

### Local Pre-Commit Hook
```bash
# Create .git/hooks/pre-commit
#!/bin/bash
pytest -m smoke -q || exit 1
```

## 🐛 Troubleshooting

### Tests Not Found
```bash
# Check test discovery
pytest --collect-only
```

### Import Errors
```bash
# Install in editable mode
pip install -e .
```

### Slow Tests
```bash
# Show slowest tests
pytest --durations=20
```

### Coverage Not Working
```bash
# Reinstall pytest-cov
pip install --upgrade pytest-cov
```

## 📚 Resources

- [Full Testing Guide](TESTING.md)
- [pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Guide](https://coverage.readthedocs.io/)

---

**Quick Start:** `python run_tests.py --quick`
