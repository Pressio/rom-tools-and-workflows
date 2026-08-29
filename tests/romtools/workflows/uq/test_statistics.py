import numpy as np
import pytest

from romtools.workflows.uq.statistics import (
    compute_monte_carlo_statistics,
    compute_multifidelity_statistics,
)


@pytest.mark.mpi_skip
def test_monte_carlo_statistics_match_numpy():
    qois = np.array([[1.0, 4.0], [2.0, 2.0], [6.0, 0.0]])
    statistics = compute_monte_carlo_statistics(qois)

    expected_variance = np.var(qois, axis=0, ddof=1)
    np.testing.assert_allclose(statistics.mean, np.mean(qois, axis=0))
    np.testing.assert_allclose(statistics.variance, expected_variance)
    np.testing.assert_allclose(statistics.standard_deviation, np.sqrt(expected_variance))
    np.testing.assert_allclose(
        statistics.standard_error, np.sqrt(expected_variance / qois.shape[0])
    )


@pytest.mark.mpi_skip
def test_perfect_low_fidelity_reduces_to_full_low_mean():
    paired = np.array([[0.0], [1.0], [2.0]])
    low = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    statistics = compute_multifidelity_statistics(paired, low)

    np.testing.assert_allclose(statistics.control_variate_coefficients, 1.0)
    np.testing.assert_allclose(statistics.mean, np.mean(low, axis=0))
    np.testing.assert_allclose(statistics.variance, np.var(paired, ddof=1) / 5.0)


@pytest.mark.mpi_skip
def test_constant_low_fidelity_has_zero_coefficient():
    high = np.array([[0.0], [1.0], [2.0]])
    low = np.ones((6, 1))
    statistics = compute_multifidelity_statistics(high, low)

    np.testing.assert_allclose(statistics.control_variate_coefficients, 0.0)
    np.testing.assert_allclose(statistics.paired_correlations, 0.0)
    np.testing.assert_allclose(statistics.mean, np.mean(high, axis=0))


@pytest.mark.mpi_skip
def test_statistics_reject_nonfinite_qois():
    with pytest.raises(ValueError, match="only finite values"):
        compute_monte_carlo_statistics(np.array([1.0, np.inf]))
