import pytest
import sys

node = 'tests/test_basic.py::test_basic_project_init'
res = pytest.main(['-q', node])
print('PYTEST_EXIT', res)
sys.exit(res)
