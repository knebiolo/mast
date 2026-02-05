"""
Automated Test Runner for PyMAST.

Usage:
  python run_tests.py [OPTIONS]
"""

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TEMP_ROOT = PROJECT_ROOT / ".temp"
PYTEST_BASETEMP = TEMP_ROOT / "pytest"


def build_temp_env() -> dict:
    """Pin temp variables to the project-local .temp directory."""
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    PYTEST_BASETEMP.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TEMP"] = str(TEMP_ROOT)
    env["TMP"] = str(TEMP_ROOT)
    env["TMPDIR"] = str(TEMP_ROOT)
    return env


def ensure_pytest_basetemp(cmd: list[str]) -> list[str]:
    """Ensure pytest writes temp files under .temp/pytest."""
    if "--basetemp" in cmd:
        return cmd

    is_pytest = (
        (len(cmd) > 0 and Path(cmd[0]).name.lower().startswith("pytest"))
        or (len(cmd) > 2 and cmd[1] == "-m" and cmd[2] == "pytest")
    )
    if is_pytest:
        return cmd + ["--basetemp", str(PYTEST_BASETEMP)]
    return cmd


def run_command(cmd: list[str], verbose: bool = False) -> int:
    """Run shell command and return returncode."""
    cmd = ensure_pytest_basetemp(cmd)
    env = build_temp_env()

    if verbose:
        print(f"\nRunning: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False, env=env)
    return result.returncode


def smoke_tests(verbose: bool = False) -> int:
    print("\n" + "=" * 60)
    print("Running Smoke Tests (< 5 seconds)")
    print("=" * 60 + "\n")

    cmd = ["pytest", "-m", "smoke", "-v", "--tb=short"]
    if not verbose:
        cmd.append("-q")

    return run_command(cmd, verbose)


def unit_tests(coverage: bool = False, verbose: bool = False) -> int:
    print("\n" + "=" * 60)
    print("Running Unit Tests (< 30 seconds)")
    print("=" * 60 + "\n")

    cmd = ["pytest", "-m", "unit", "-v"]
    if coverage:
        cmd.extend(["--cov=pymast", "--cov-report=term-missing"])
    if not verbose:
        cmd.append("-q")

    return run_command(cmd, verbose)


def integration_tests(verbose: bool = False) -> int:
    print("\n" + "=" * 60)
    print("Running Integration Tests (< 2 minutes)")
    print("=" * 60 + "\n")

    cmd = ["pytest", "-m", "integration", "-v"]
    if not verbose:
        cmd.append("-q")

    return run_command(cmd, verbose)


def all_tests(coverage: bool = False, html: bool = False, parallel: bool = False, verbose: bool = False) -> int:
    print("\n" + "=" * 60)
    print("Running Full Test Suite")
    print("=" * 60 + "\n")

    cmd = ["pytest", "-v"]
    if coverage:
        cmd.extend(["--cov=pymast", "--cov-report=term-missing"])
        if html:
            cmd.append("--cov-report=html")
    if parallel:
        cmd.extend(["-n", "auto"])

    cmd.append("--durations=10")

    if not verbose:
        cmd.remove("-v")
        cmd.append("-q")

    return run_command(cmd, verbose)


def benchmark_tests(verbose: bool = False) -> int:
    print("\n" + "=" * 60)
    print("Running Performance Benchmarks")
    print("=" * 60 + "\n")

    cmd = ["pytest", "-m", "benchmark", "-v", "--benchmark-only"]
    if not verbose:
        cmd.append("-q")

    return run_command(cmd, verbose)


def watch_mode(test_type: str = "unit", verbose: bool = False) -> None:
    """Watch mode - rerun tests on file changes."""
    print("\n" + "=" * 60)
    print(f"Watch Mode - Running {test_type} tests on changes")
    print("Press Ctrl+C to exit")
    print("=" * 60 + "\n")

    cmd = ["ptw", "--", "-m", test_type, "--basetemp", str(PYTEST_BASETEMP)]
    if verbose:
        cmd.append("-v")

    try:
        subprocess.run(cmd, env=build_temp_env(), check=False)
    except KeyboardInterrupt:
        print("\n\nWatch mode stopped")


def open_coverage_report() -> None:
    """Open HTML coverage report in browser."""
    html_path = Path("htmlcov/index.html")
    if html_path.exists():
        print(f"\nOpening coverage report: {html_path.absolute()}")
        webbrowser.open(f"file://{html_path.absolute()}")
    else:
        print("\nCoverage report not found. Run with --html first.")


def print_summary(results: dict[str, int]) -> int:
    """Print summary and return non-zero if any test failed."""
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60 + "\n")

    for test_name, return_code in results.items():
        status = "PASSED" if return_code == 0 else "FAILED"
        print(f"{test_name:.<40} {status}")

    print("\n")

    if all(rc == 0 for rc in results.values()):
        print("All tests passed")
        return 0

    print("Some tests failed. See details above.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PyMAST Automated Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --quick
  python run_tests.py --unit --coverage
  python run_tests.py --all --html
  python run_tests.py --watch --unit
        """,
    )

    parser.add_argument("--quick", action="store_true", help="Run smoke tests only (< 5 sec)")
    parser.add_argument("--unit", action="store_true", help="Run unit tests (< 30 sec)")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--all", action="store_true", help="Run all tests including slow tests")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")

    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--watch", action="store_true", help="Watch mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--open-coverage", action="store_true", help="Open HTML coverage report")

    args = parser.parse_args()

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

    if not any([args.quick, args.unit, args.integration, args.all, args.benchmark]):
        args.quick = True

    results: dict[str, int] = {}

    if args.quick:
        results["Smoke Tests"] = smoke_tests(args.verbose)
    if args.unit:
        results["Unit Tests"] = unit_tests(args.coverage, args.verbose)
    if args.integration:
        results["Integration Tests"] = integration_tests(args.verbose)
    if args.all:
        results["Full Test Suite"] = all_tests(args.coverage, args.html, args.parallel, args.verbose)
    if args.benchmark:
        results["Benchmarks"] = benchmark_tests(args.verbose)

    if args.html and args.coverage:
        time.sleep(1)
        open_coverage_report()

    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
