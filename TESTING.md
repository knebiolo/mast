# PyMAST Testing Guide

## 🧪 Overview

PyMAST uses **pytest** for automated testing with comprehensive coverage reporting. Tests are organized by functionality and speed for efficient development workflows.

## 📋 Quick Start

### Install Test Dependencies

```bash
pip install pytest pytest-cov pytest-benchmark
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Fast unit tests only
pytest -m unit

# Integration tests (requires test database)
pytest -m integration

# Specific module tests
pytest -m classifier
pytest -m overlap
pytest -m bout

# Smoke tests (quick sanity checks)
pytest -m smoke
```

### Run with Coverage Report

```bash
# Terminal output with missing lines
pytest --cov=pymast --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=pymast --cov-report=html
# Open htmlcov/index.html in browser
```

## 📁 Test Organization

```
tests/
├── test_basic.py                      # Core functionality tests
├── test_overlap_unit.py               # Overlap removal unit tests
├── test_overlap_small.py              # Small dataset integration
├── test_overlap_hdf5_integration.py   # HDF5 database integration
├── run_unit_tests.py                  # Legacy test runner
└── conftest.py                        # Pytest fixtures (create this)
```

## 🏗️ Test Categories

### Unit Tests (`-m unit`)
- **Fast** (< 1 second each)
- No database required
- Test individual functions in isolation
- Mock external dependencies

**Examples:**
- `test_predictors.py` - Predictor calculations
- `test_naive_bayes.py` - Classifier math
- `test_parsers.py` - File parsing logic

### Integration Tests (`-m integration`)
- **Moderate speed** (1-10 seconds)
- Require test database
- Test component interactions
- Use sample datasets

**Examples:**
- `test_overlap_hdf5_integration.py` - Database operations
- `test_end_to_end.py` - Full workflow

### Smoke Tests (`-m smoke`)
- **Very fast** (< 0.1 seconds)
- Critical path verification
- Run before commits
- Basic sanity checks

## 📝 Writing Tests

### Test File Template

```python
"""
Test module for [component name]
"""

import pytest
import numpy as np
import pandas as pd
from pymast import [module]


@pytest.mark.unit
class TestComponentName:
    """Test [component] functionality"""
    
    def test_basic_functionality(self):
        """Test basic case"""
        result = [module].function(input_data)
        assert result == expected_value
        
    def test_edge_cases(self):
        """Test edge cases"""
        # Empty input
        with pytest.raises(ValueError):
            [module].function([])
            
    def test_invalid_input(self):
        """Test invalid input handling"""
        with pytest.raises(TypeError):
            [module].function(None)


@pytest.mark.integration
class TestComponentIntegration:
    """Integration tests requiring database"""
    
    @pytest.fixture
    def sample_db(self, tmp_path):
        """Create temporary test database"""
        db_path = tmp_path / "test.h5"
        # Setup database
        yield db_path
        # Teardown handled by tmp_path
        
    def test_database_operations(self, sample_db):
        """Test database read/write"""
        # Test with temporary database
        pass
```

### Using Fixtures

Create `tests/conftest.py`:

```python
"""
Shared pytest fixtures for PyMAST tests
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_tags():
    """Sample tag data for testing"""
    return pd.DataFrame({
        'freq_code': ['164.123 45', '164.456 78'],
        'pulse_rate': [3.0, 5.0],
        'tag_type': ['study', 'study'],
        'rel_date': pd.to_datetime(['2024-01-01', '2024-01-02'])
    })


@pytest.fixture
def sample_receivers():
    """Sample receiver data for testing"""
    return pd.DataFrame({
        'rec_id': ['R01', 'R02'],
        'rec_type': ['srx800', 'srx800'],
        'node': ['N01', 'N02']
    })


@pytest.fixture
def temp_project(tmp_path):
    """Create temporary project directory"""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    return project_dir
```

## 🎯 Test Markers

Mark tests with decorators:

```python
@pytest.mark.unit           # Fast unit test
@pytest.mark.integration    # Integration test
@pytest.mark.slow           # Slow test (> 10 seconds)
@pytest.mark.classifier     # Naive Bayes tests
@pytest.mark.overlap        # Overlap removal tests
@pytest.mark.bout           # Bout detection tests
@pytest.mark.skip(reason="...")  # Skip test
@pytest.mark.xfail          # Expected to fail
```

## 📊 Coverage Goals

| Component | Target Coverage |
|-----------|----------------|
| `predictors.py` | > 90% |
| `naive_bayes.py` | > 90% |
| `overlap_removal.py` | > 85% |
| `parsers.py` | > 80% |
| `formatter.py` | > 80% |
| `radio_project.py` | > 75% |

## 🚀 Continuous Testing Workflow

### Pre-Commit Checks

```bash
# Run smoke tests (< 5 seconds)
pytest -m smoke

# Run unit tests (< 30 seconds)
pytest -m unit
```

### Before Pull Request

```bash
# Run all tests with coverage
pytest --cov=pymast --cov-report=html

# Check coverage report
# Ensure new code has > 80% coverage
```

### Full Test Suite

```bash
# Run everything including slow tests
pytest -v --durations=20

# Generate comprehensive coverage report
pytest --cov=pymast --cov-report=html --cov-report=term
```

## 🐛 Debugging Tests

### Run Single Test

```bash
# Specific test function
pytest tests/test_basic.py::TestPredictors::test_noise_ratio

# Specific test class
pytest tests/test_basic.py::TestPredictors

# Specific file
pytest tests/test_basic.py
```

### Verbose Output

```bash
# Show print statements
pytest -s

# Very verbose
pytest -vv

# Show local variables on failure
pytest -l
```

### Debug Mode

```python
# In test file, add breakpoint
def test_something():
    x = calculate_value()
    breakpoint()  # Drops into debugger
    assert x == expected
```

Then run:
```bash
pytest --pdb  # Drop into debugger on failure
```

## 📈 Performance Testing

### Benchmark Tests

```python
import pytest

@pytest.mark.benchmark
def test_performance(benchmark):
    """Benchmark function performance"""
    result = benchmark(function_to_test, arg1, arg2)
    assert result == expected
```

### Identify Slow Tests

```bash
# Show 20 slowest tests
pytest --durations=20
```

## 🔄 Test Data Management

### Sample Data Structure

```
tests/
├── data/
│   ├── sample_receiver_data.csv
│   ├── sample_tags.csv
│   ├── sample_receivers.csv
│   └── expected_outputs/
│       ├── classified.csv
│       └── bouts.csv
└── fixtures/
    └── sample_database.h5
```

### Using Test Data

```python
import pytest
from pathlib import Path

@pytest.fixture
def test_data_dir():
    """Path to test data directory"""
    return Path(__file__).parent / "data"

def test_parser(test_data_dir):
    """Test parser with sample data"""
    input_file = test_data_dir / "sample_receiver_data.csv"
    result = parse_file(input_file)
    assert len(result) > 0
```

## 🤖 Automated Testing Agent

### GitHub Actions Workflow

Create `.github/workflows/tests.yml`:

```yaml
name: PyMAST Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest --cov=pymast --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Local Pre-Commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Run smoke tests before allowing commit

echo "Running pre-commit tests..."
pytest -m smoke -q

if [ $? -ne 0 ]; then
    echo "❌ Smoke tests failed. Commit aborted."
    exit 1
fi

echo "✅ Tests passed. Proceeding with commit."
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

## 📚 Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov plugin](https://pytest-cov.readthedocs.io/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

## 🎓 Testing Best Practices

1. **Write tests first** (TDD) or alongside code
2. **Keep tests fast** - Mock slow operations
3. **Test edge cases** - Empty inputs, None, invalid types
4. **Use descriptive names** - `test_noise_ratio_with_empty_array`
5. **One assertion per test** (when possible)
6. **Use fixtures** - Don't repeat setup code
7. **Test behavior, not implementation**
8. **Maintain > 80% coverage** for new code

---

**Happy Testing! 🧪✅**
