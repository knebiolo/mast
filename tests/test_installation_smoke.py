"""
Smoke test to verify package installation and critical dependencies.

This test ensures:
1. The pymast package can be imported
2. Critical dependencies are installed
3. Core functionality is accessible
"""

import pytest
import sys


def test_pymast_imports():
    """Verify pymast package can be imported."""
    import pymast
    assert hasattr(pymast, '__version__')
    assert pymast.__version__ is not None


def test_scikit_learn_dependency():
    """Verify scikit-learn clustering primitives are available."""
    try:
        import sklearn.cluster
        assert hasattr(sklearn.cluster, 'KMeans')
    except ImportError as e:
        pytest.fail(
            f"scikit-learn is not installed but is required. "
            f"Ensure Python >= 3.9 and install with: pip install pymast\n"
            f"Error: {e}"
        )


def test_python_version():
    """Verify Python version meets minimum requirements."""
    assert sys.version_info >= (3, 9), (
        f"Python 3.9+ is required (found {sys.version_info.major}.{sys.version_info.minor}). "
        "Core dependencies require Python >= 3.9."
    )


def test_core_modules_importable():
    """Verify core modules can be imported."""
    import pymast
    
    # Verify key classes/functions are accessible
    assert hasattr(pymast, 'radio_project')
    assert hasattr(pymast, 'bout')
    assert hasattr(pymast, 'overlap_reduction')


def test_critical_dependencies():
    """Verify all critical dependencies are importable."""
    critical_deps = [
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'sklearn',
        'h5py',
        'dask',
        'distributed',
        'numba',
        'tables',
        'intervaltree',
    ]
    
    missing = []
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    assert not missing, f"Missing critical dependencies: {', '.join(missing)}"


if __name__ == '__main__':
    # Allow running directly for quick verification
    pytest.main([__file__, '-v'])
