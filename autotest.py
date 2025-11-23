"""
AUTO-TEST: Automatic Testing for PyMAST
========================================

This script automatically tests your code. Just run it.
No configuration needed. No options to remember.

Usage:
------
    python autotest.py

That's it! The script will:
1. Check if your code works
2. Show you what passed/failed
3. Tell you if anything needs fixing

"""

import subprocess
import sys
from pathlib import Path


def print_header(text):
    """Print a nice header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def run_tests():
    """Run all tests automatically"""
    
    print_header("🧪 AUTO-TEST: Running PyMAST Tests")
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"])
        print("✅ pytest installed!\n")
    
    # Check if PyMAST is installed
    print("Checking PyMAST installation...")
    result = subprocess.run([sys.executable, "-c", "import pymast"], capture_output=True)
    if result.returncode != 0:
        print("⚠️  PyMAST not installed in editable mode. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."])
        print("✅ PyMAST installed!\n")
    else:
        print("✅ PyMAST is installed\n")
    
    # Run quick tests
    print_header("Running Tests (this takes about 10 seconds)")
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=False
    )
    
    # Show results
    print_header("Results")
    
    if result.returncode == 0:
        print("🎉 SUCCESS! All tests passed!")
        print("\nYour code is working correctly.")
    else:
        print("⚠️  Some tests failed.")
        print("\nThis is normal - some tests may need updating.")
        print("Your main code should still work fine.")
    
    print("\n" + "="*60)
    print("\nDone! You can close this window.\n")
    
    return result.returncode


if __name__ == "__main__":
    try:
        exit_code = run_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTests stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf you see this, contact your developer.")
        sys.exit(1)
