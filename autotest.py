"""
AUTO-TEST: Automatic Testing for PyMAST.

Usage:
    python autotest.py
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TEMP_ROOT = PROJECT_ROOT / ".temp"
PYTEST_BASETEMP = TEMP_ROOT / "pytest"


def build_temp_env() -> dict:
    """Pin temp variables to project-local .temp."""
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    PYTEST_BASETEMP.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TEMP"] = str(TEMP_ROOT)
    env["TMP"] = str(TEMP_ROOT)
    env["TMPDIR"] = str(TEMP_ROOT)
    return env


def print_header(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def run_tests() -> int:
    print_header("AUTO-TEST: Running PyMAST Tests")

    try:
        import pytest  # noqa: F401
    except ImportError:
        print("pytest not found. Install it first:")
        print("   python -m pip install pytest pytest-cov")
        return 1

    print("Checking PyMAST installation...")
    try:
        import pymast  # noqa: F401
    except ImportError:
        print("PyMAST is not importable. Install it first:")
        print("   python -m pip install -e .")
        print("   # or: python -m pip install pymast")
        return 1
    else:
        print("PyMAST is installed\n")

    print_header("Running Tests")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--basetemp",
            str(PYTEST_BASETEMP),
        ],
        capture_output=False,
        check=False,
        env=build_temp_env(),
    )

    print_header("Results")

    if result.returncode == 0:
        print("SUCCESS! All tests passed!")
    else:
        print("Some tests failed.")

    print("\n" + "=" * 60)
    print("\nDone!\n")

    return result.returncode


if __name__ == "__main__":
    try:
        exit_code = run_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTests stopped by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nError: {exc}")
        sys.exit(1)
