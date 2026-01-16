"""
Automated Test Runner for PyMAST

This script provides a comprehensive test automation interface for local development.
It runs different test suites based on the development stage and provides detailed reports.

Usage:
    python run_tests.py [OPTIONS]

Options:
    --quick         Run smoke tests only (< 5 seconds)
    --unit          Run unit tests (< 30 seconds)
    --integration   Run integration tests (requires database)
    --all           Run all tests including slow tests
    --coverage      Generate coverage report
    --html          Generate HTML coverage report
    --benchmark     Run performance benchmarks
    --watch         Watch mode - rerun on file changes
    --verbose       Verbose output
    --parallel      Run tests in parallel (faster)

Examples:
    python run_tests.py --quick
    python run_tests.py --unit --coverage
    python run_tests.py --all --html
    python run_tests.py --watch --unit
"""

import sys
import subprocess
import argparse
from pathlib import Path
import time
import webbrowser


def run_command(cmd, verbose=False):
    """
    Run shell command and return result
    
    Args:
        cmd (list): Command and arguments
        verbose (bool): Print command before running
        
    Returns:
        int: Return code
    """
    if verbose:
        print(f"\n🚀 Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, check=False)
    return result.returncode


def smoke_tests(verbose=False):
    """Run quick smoke tests"""
    print("\n" + "="*60)
    print("💨 Running Smoke Tests (< 5 seconds)")
    print("="*60 + "\n")
    
    cmd = ["pytest", "-m", "smoke", "-v", "--tb=short"]
    if not verbose:
        cmd.append("-q")
    
    return run_command(cmd, verbose)


def unit_tests(coverage=False, verbose=False):
    """Run unit tests"""
    print("\n" + "="*60)
    print("🧪 Running Unit Tests (< 30 seconds)")
    print("="*60 + "\n")
    
    cmd = ["pytest", "-m", "unit", "-v"]
    
    if coverage:
        cmd.extend(["--cov=pymast", "--cov-report=term-missing"])
    
    if not verbose:
        cmd.append("-q")
    
    return run_command(cmd, verbose)


def integration_tests(verbose=False):
    """Run integration tests"""
    print("\n" + "="*60)
    print("🔗 Running Integration Tests (< 2 minutes)")
    print("="*60 + "\n")
    
    cmd = ["pytest", "-m", "integration", "-v"]
    
    if not verbose:
        cmd.append("-q")
    
    return run_command(cmd, verbose)


def all_tests(coverage=False, html=False, parallel=False, verbose=False):
    """Run all tests"""
    print("\n" + "="*60)
    print("🎯 Running Full Test Suite")
    print("="*60 + "\n")
    
    cmd = ["pytest", "-v"]
    
    if coverage:
        cmd.extend(["--cov=pymast", "--cov-report=term-missing"])
        if html:
            cmd.append("--cov-report=html")
    
    if parallel:
        cmd.extend(["-n", "auto"])  # Requires pytest-xdist
    
    cmd.append("--durations=10")
    
    if not verbose:
        cmd.remove("-v")
        cmd.append("-q")
    
    return run_command(cmd, verbose)


def benchmark_tests(verbose=False):
    """Run performance benchmarks"""
    print("\n" + "="*60)
    print("⚡ Running Performance Benchmarks")
    print("="*60 + "\n")
    
    cmd = ["pytest", "-m", "benchmark", "-v", "--benchmark-only"]
    
    if not verbose:
        cmd.append("-q")
    
    return run_command(cmd, verbose)


def watch_mode(test_type="unit", verbose=False):
    """
    Watch mode - rerun tests on file changes
    
    Requires: pip install pytest-watch
    """
    print("\n" + "="*60)
    print(f"👀 Watch Mode - Running {test_type} tests on changes")
    print("Press Ctrl+C to exit")
    print("="*60 + "\n")
    
    cmd = ["ptw", "--", "-m", test_type]
    
    if verbose:
        cmd.append("-v")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n✅ Watch mode stopped")


def open_coverage_report():
    """Open HTML coverage report in browser"""
    html_path = Path("htmlcov/index.html")
    
    if html_path.exists():
        print(f"\n📊 Opening coverage report: {html_path.absolute()}")
        webbrowser.open(f"file://{html_path.absolute()}")
    else:
        print("\n❌ Coverage report not found. Run with --html first.")


def print_summary(results):
    """Print test summary"""
    print("\n" + "="*60)
    print("📋 Test Summary")
    print("="*60 + "\n")
    
    for test_name, return_code in results.items():
        status = "✅ PASSED" if return_code == 0 else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print("\n")
    
    if all(rc == 0 for rc in results.values()):
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. See details above.")
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PyMAST Automated Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --quick           # Fast smoke tests
  python run_tests.py --unit --coverage # Unit tests with coverage
  python run_tests.py --all --html      # Full suite with HTML report
  python run_tests.py --watch --unit    # Watch mode for development
        """
    )
    
    # Test selection
    parser.add_argument("--quick", action="store_true",
                       help="Run smoke tests only (< 5 sec)")
    parser.add_argument("--unit", action="store_true",
                       help="Run unit tests (< 30 sec)")
    parser.add_argument("--integration", action="store_true",
                       help="Run integration tests (requires DB)")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests including slow tests")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmarks")
    
    # Options
    parser.add_argument("--coverage", action="store_true",
                       help="Generate coverage report")
    parser.add_argument("--html", action="store_true",
                       help="Generate HTML coverage report")
    parser.add_argument("--parallel", action="store_true",
                       help="Run tests in parallel (faster)")
    parser.add_argument("--watch", action="store_true",
                       help="Watch mode - rerun on changes")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--open-coverage", action="store_true",
                       help="Open HTML coverage report in browser")
    
    args = parser.parse_args()
    
    # Handle special modes
    if args.open_coverage:
        open_coverage_report()
        return 0
    
    if args.watch:
        test_type = "unit"
        if args.integration:
            test_type = "integration"
        elif args.quick:
            test_type = "smoke"
        watch_mode(test_type, args.verbose)
        return 0
    
    # Default to quick tests if nothing specified
    if not any([args.quick, args.unit, args.integration, args.all, args.benchmark]):
        args.quick = True
    
    # Run selected tests
    results = {}
    
    if args.quick:
        results["Smoke Tests"] = smoke_tests(args.verbose)
    
    if args.unit:
        results["Unit Tests"] = unit_tests(args.coverage, args.verbose)
    
    if args.integration:
        results["Integration Tests"] = integration_tests(args.verbose)
    
    if args.all:
        results["Full Test Suite"] = all_tests(
            args.coverage, args.html, args.parallel, args.verbose
        )
    
    if args.benchmark:
        results["Benchmarks"] = benchmark_tests(args.verbose)
    
    # Open coverage report if requested
    if args.html and args.coverage:
        time.sleep(1)  # Give time for report generation
        open_coverage_report()
    
    # Print summary and return
    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
