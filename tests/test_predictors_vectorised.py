"""The vectorised detection history must reproduce the original implementation exactly."""
import importlib.util
import pathlib

import numpy as np
import pytest

from pymast import predictors


def _original():
    path = pathlib.Path(__file__).parent / "data" / "predictors_original.py"
    spec = importlib.util.spec_from_file_location("predictors_original", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
@pytest.mark.parametrize("pulse_rate,channels,scan_time", [(2.0, 1, 1.0), (3.0, 1, 3.5), (8.0, 3, 3.5), (12.0, 1, 1.0), (1.0, 3, 1.0)])
def test_detection_history_matches_original(pulse_rate, channels, scan_time):
    rng = np.random.default_rng(7)
    gaps = rng.choice([pulse_rate, pulse_rate, pulse_rate, 2 * pulse_rate, 1, 40, 300], size=400)
    epoch = (1_500_000_000 + np.cumsum(gaps)).astype(np.int64)
    n = len(epoch)
    ch, st = np.full(n, channels), np.full(n, scan_time)
    old = _original().detection_history(epoch, pulse_rate, 5, ch, st)
    new = predictors.detection_history(epoch, pulse_rate, 5, ch, st)
    for a, b in zip(old, new):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.unit
def test_detection_history_short_series():
    epoch = np.array([100, 105], dtype=np.int64)
    ch, st = np.ones(2, dtype=int), np.ones(2)
    old = _original().detection_history(epoch, 5.0, 5, ch, st)
    new = predictors.detection_history(epoch, 5.0, 5, ch, st)
    for a, b in zip(old, new):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.unit
def test_max_contiguous_runs_matches_string_version():
    rng = np.random.default_rng(1)
    hist = rng.integers(0, 2, size=(500, 11))
    hist[7] = 0
    expected = np.array([predictors.max_contiguous_sequence(r) for r in hist])
    np.testing.assert_array_equal(predictors.max_contiguous_runs(hist), expected)
