# Installation Guide

## Quick Install

### For Users

Install from GitHub (once v1.0 is released):

```bash
pip install git+https://github.com/knebiolo/mast.git
```

Or for development:

```bash
git clone https://github.com/knebiolo/mast.git
cd mast
pip install -e .
```

## Detailed Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- 8+ GB RAM recommended for large datasets
- 10+ GB disk space for project data

### Step 1: Install Python

If you don't have Python installed:

**Windows:**
1. Download from [python.org](https://www.python.org/downloads/)
2. Run installer, **check "Add Python to PATH"**
3. Verify: Open PowerShell and type `python --version`

**macOS:**
```bash
brew install python@3.10
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install python3.10 python3-pip
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv mast_env

# Activate it
# Windows:
mast_env\Scripts\activate

# Mac/Linux:
source mast_env/bin/activate
```

### Step 3: Install MAST

#### Option A: From GitHub (Recommended)

```bash
pip install git+https://github.com/knebiolo/mast.git
```

#### Option B: From Source (For Development)

```bash
# Clone the repository
git clone https://github.com/knebiolo/mast.git
cd mast

# Install in editable mode
pip install -e .
```

#### Option C: Using Conda

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate pymast
```

### Step 4: Verify Installation

```python
python -c "import pymast; print(pymast.__version__)"
```

Should print version number without errors.

## Dependencies

MAST requires these packages (installed automatically):

**Core:**
- numpy (>=1.20)
- pandas (>=1.3)
- h5py (>=3.0)
- scipy (>=1.7)

**Scientific Computing:**
- statsmodels (>=0.13)
- scikit-learn (>=1.0)
- numba (>=0.54)

**Visualization:**
- matplotlib (>=3.4)
- networkx (>=2.6)

**Parallel Processing:**
- dask (>=2021.10)
- dask-ml (>=2021.10)

**Data Storage:**
- tables (>=3.7) - PyTables for HDF5
- intervaltree (>=3.1)

## Optional Dependencies

For Jupyter notebook usage:
```bash
pip install jupyter ipykernel
```

For development:
```bash
pip install pytest pytest-cov black flake8 mypy
```

## Platform-Specific Notes

### Windows

- Use PowerShell or Command Prompt, not Git Bash
- Paths use backslashes: `C:\path\to\project`
- May need Visual C++ Build Tools for some dependencies

### macOS

- May need Xcode Command Line Tools: `xcode-select --install`
- Use forward slashes in paths: `/Users/yourname/project`

### Linux

- May need python3-dev: `sudo apt-get install python3-dev`
- Some HPC systems require module loading: `module load python/3.10`

## Troubleshooting

### ImportError: No module named 'pymast'

**Solution:** Make sure you're in the activated virtual environment and MAST is installed:
```bash
pip list | grep pymast
```

### HDF5 Library Errors

**Solution:** Reinstall h5py and tables:
```bash
pip uninstall h5py tables
pip install h5py tables --no-cache-dir
```

### Numba Compilation Warnings

**Solution:** Ignore or disable numba warnings:
```python
import warnings
warnings.filterwarnings('ignore', category=numba.NumbaPendingDeprecationWarning)
```

### Memory Errors with Large Datasets

**Solution:** 
1. Increase available RAM
2. Use dask for chunked processing
3. Process receivers individually
4. Clear intermediate results frequently

### Permission Errors on Windows

**Solution:** Run as administrator or install to user directory:
```bash
pip install --user git+https://github.com/knebiolo/mast.git
```

## Updating MAST

### From GitHub:
```bash
pip install --upgrade git+https://github.com/knebiolo/mast.git
```

### From Source:
```bash
cd mast
git pull
pip install -e . --upgrade
```

## Uninstalling

```bash
pip uninstall pymast
```

## Alternative: Docker (Future)

For reproducible environments:
```bash
docker pull knebiolo/mast:latest
docker run -it -v /path/to/data:/data knebiolo/mast:latest
```

*(Docker image coming soon)*

## Getting Help

If installation fails:
1. Check Python version: `python --version` (must be >=3.9)
2. Check pip version: `pip --version` (should be recent)
3. Try installing in a fresh virtual environment
4. Check GitHub issues for similar problems
5. Open a new issue with error details

## Next Steps

After installation:
1. Read the [Tutorial](docs/TUTORIAL.md)
2. Try the [Quick Start Example](examples/quick_start_example.py)
3. Review the [API Reference](docs/API_REFERENCE.md)
