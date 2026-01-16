import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
proj_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root))


def main() -> int:
    try:
        import pytest
    except ImportError as exc:
        raise SystemExit("pytest is required: python -m pip install pytest") from exc

    tests_path = Path(__file__).resolve().parent / "test_overlap_unit.py"
    return pytest.main([str(tests_path)])


if __name__ == "__main__":
    raise SystemExit(main())
