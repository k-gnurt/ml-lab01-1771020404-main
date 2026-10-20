import math
import numpy as np
import pytest

from nb_manual import compute_class_priors, gaussian_log_likelihood


def test_compute_class_priors():
    y = np.array([0, 0, 0, 1])
    priors = compute_class_priors(y)
    assert set(priors.keys()) == {0, 1}
    assert priors[0] == pytest.approx(0.75)
    assert priors[1] == pytest.approx(0.25)
    assert sum(priors.values()) == pytest.approx(1.0)


def test_gaussian_log_likelihood_at_mean():
    got = gaussian_log_likelihood(x=10.0, mean=10.0, var=4.0)
    expected = -0.5 * math.log(2 * math.pi * 4.0)
    assert got == pytest.approx(expected)


def test_gaussian_log_likelihood_general():
    got = gaussian_log_likelihood(x=12.0, mean=10.0, var=4.0)
    expected = -0.5 * math.log(2 * math.pi * 4.0) - ((12.0 - 10.0) ** 2) / (2 * 4.0)
    assert got == pytest.approx(expected)


def test_gaussian_log_likelihood_rejects_nonpositive_variance():
    with pytest.raises(ValueError):
        gaussian_log_likelihood(x=1.0, mean=0.0, var=0.0)
