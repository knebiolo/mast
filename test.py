#!/usr/bin/env python
"""
Quick test launcher - the easiest way to run PyMAST tests

Just run: python test.py

For options: python test.py --help
"""

import sys
import subprocess


def main():
    """Run the full test automation script"""
    # If no args provided, show help
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("🧪 PyMAST Test Launcher")
        print("="*60 + "\n")
        print("Quick commands:")
        print("  python test.py --quick       # Smoke tests (5 sec)")
        print("  python test.py --unit        # Unit tests (30 sec)")
        print("  python test.py --all         # Full suite")
        print("  python test.py --coverage    # With coverage report")
        print("  python test.py --html        # Generate HTML report")
        print("  python test.py --watch       # Watch mode")
        print("\nFor all options: python test.py --help")
        print("\nDefault (no args): Running quick smoke tests...\n")
        
        # Run quick tests by default
        sys.argv.append("--quick")
    
    # Delegate to full test runner
    cmd = [sys.executable, "run_tests.py"] + sys.argv[1:]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
