"""Regression tests for expected improvement in minimization EGO."""

import numpy as np
import pytest
from scipy.stats import norm

from romtools.workflows.inverse.ego_optimization_methods import _expected_improvement


class FixedPredictionGP:
    def __init__(self, means, stds):
        self.means = np.asarray(means, dtype=float)
        self.stds = np.asarray(stds, dtype=float)

    def compute_qoi_and_var(self, run_directory, parameter_sample):
        return self.means.copy(), self.stds.copy()


@pytest.mark.parametrize(
    "mean, std, best, epsilon",
    [
        (0.0, 1.0, 1.0, 0.0),
        (2.0, 1.0, 1.0, 0.0),
        (1.0, 2.0, 1.0, 0.0),
        (0.8, 0.5, 1.0, 0.2),
    ],
)
def test_expected_improvement_minimization_formula(mean, std, best, epsilon):
    improvement = best + epsilon - mean
    z = improvement / std
    expected = improvement * norm.cdf(z) + std * norm.pdf(z)
    actual = _expected_improvement(
        FixedPredictionGP([mean], [std]), best, np.array([0.25]), epsilon
    )
    np.testing.assert_allclose(actual, [expected], rtol=1e-12, atol=1e-14)


def test_expected_improvement_zero_and_small_uncertainty():
    means = np.array([0.5, 1.0, 1.5, 0.7, 1.3])
    stds = np.array([0.0, 0.0, 0.0, 1e-15, 0.5])
    actual = _expected_improvement(
        FixedPredictionGP(means, stds), 1.0, np.array([0.25])
    )
    improvement = 1.0 - means
    expected_last = improvement[-1] * norm.cdf(improvement[-1] / stds[-1])
    expected_last += stds[-1] * norm.pdf(improvement[-1] / stds[-1])
    np.testing.assert_allclose(
        actual, [0.5, 0.0, 0.0, 0.3, expected_last],
        rtol=1e-12, atol=1e-14,
    )
    assert np.all(np.isfinite(actual))
    assert np.all(actual >= 0.0)
