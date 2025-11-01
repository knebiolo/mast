import runpy
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
proj_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root))

tests_path = Path(__file__).resolve().parent / 'test_overlap_unit.py'
src = tests_path.read_text()
ns = {}
exec(compile(src, str(tests_path), 'exec'), ns)

fns = [ns[name] for name in ['test_posterior_wins','test_ambiguous_keep_both','test_missing_posterior_raises','test_power_method']]
failed = []
for fn in fns:
    try:
        fn()
        print(f'PASS: {fn.__name__}')
    except AssertionError as e:
        import traceback
        print(f'FAIL: {fn.__name__} - {e}')
        traceback.print_exc()
        failed.append((fn.__name__, str(e)))
    except Exception as e:
        import traceback
        print(f'ERROR: {fn.__name__} - {e}')
        traceback.print_exc()
        failed.append((fn.__name__, str(e)))

if not failed:
    print('\nAll tests passed')
else:
    print('\nFailures:')
    for name, msg in failed:
        print(name, msg)
