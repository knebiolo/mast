"""LikelihoodTable must reproduce calculate_likelihood, and the dtype fix must be switchable."""
import numpy as np
import pytest

from pymast import naive_bayes as nb


def _data(dtype):
    rng = np.random.default_rng(3)
    train = rng.integers(0, 12, size=5000).astype(dtype)
    labels = rng.integers(0, 2, size=5000)
    obs = rng.integers(-2, 15, size=800).astype(dtype)      # includes values never seen in training
    return train, labels, obs


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("assumption", [True, False])
def test_matches_calculate_likelihood(dtype, assumption):
    train, labels, obs = _data(dtype)
    old = nb.calculate_likelihood(train, labels, assumption, obs)
    new = nb.LikelihoodTable(train, labels).lookup(assumption, obs)
    np.testing.assert_allclose(new, old, rtol=1e-12, atol=0)


@pytest.mark.unit
def test_float64_observations_against_float32_training():
    """1/11 is stored as float32 in the training table but computed as float64 for the fish: the old lookup misses it."""
    train = np.array([1 / 11, 2 / 11, 1.0, 1 / 11], dtype=np.float32)
    labels = np.array([1, 0, 1, 1])
    obs = np.array([1 / 11, 2 / 11, 1.0], dtype=np.float64)
    old = nb.calculate_likelihood(train, labels, True, obs)
    legacy = nb.LikelihoodTable(train, labels, match_training_dtype=False).lookup(True, obs)
    fixed = nb.LikelihoodTable(train, labels).lookup(True, obs)
    np.testing.assert_allclose(legacy, old)                  # the switch reproduces the old behaviour exactly
    default = 1 / (3 + 3)                                    # laplace / (n_true + laplace * n_unique)
    assert legacy[0] == pytest.approx(default) and legacy[1] == pytest.approx(default)   # both missed the table
    assert fixed[0] == pytest.approx((2 + 1) / 6)            # found: two true rows with this value
    assert fixed[1] == pytest.approx(1 / 6)
    assert fixed[2] == legacy[2] == pytest.approx((1 + 1) / 6)   # 1.0 is exact in both types


@pytest.mark.unit
def test_empty_training_set_raises_with_a_message():
    with pytest.raises(ValueError, match="empty training set"):
        nb.LikelihoodTable(np.array([]), np.array([]))
