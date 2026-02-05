import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))

# Force test temporary files into repo-local .temp to avoid system temp ACL issues.
TEST_TEMP_ROOT = PROJECT_ROOT / ".temp" / "runtime"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(TEST_TEMP_ROOT)
os.environ["TMP"] = str(TEST_TEMP_ROOT)
os.environ["TMPDIR"] = str(TEST_TEMP_ROOT)
tempfile.tempdir = str(TEST_TEMP_ROOT)
