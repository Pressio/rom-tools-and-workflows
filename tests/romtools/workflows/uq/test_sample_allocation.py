import numpy as np
import pytest

from romtools.workflows.uq.sample_allocation import allocate_samples_from_pilot


def _brute_force_expected_allocation(high, low, budget, cost_ratio):
    variance = np.var(high, ddof=1)
    rho_squared = np.corrcoef(high, low)[0, 1] ** 2
    pilot_count = high.size
    candidates = []
    for number_high in range(pilot_count, int(np.floor(budget)) + 1):
        for number_low in range(number_high, int(np.floor(budget / cost_ratio)) + 1):
            cost = number_high + cost_ratio * number_low
            if cost <= budget + 1e-12:
                estimator_variance = variance * (
                    (1.0 - rho_squared) / number_high + rho_squared / number_low
                )
                candidates.append(
                    (estimator_variance, cost, -number_high, number_high, number_low)
                )
    return min(candidates) if candidates else None


@pytest.mark.mpi_skip
def test_integer_allocation_matches_brute_force_search():
    low = np.linspace(-1.0, 1.0, 6)
    high = low + np.array([0.0, 0.03, -0.02, 0.01, -0.03, 0.02])
    budget = 25.0
    cost_ratio = 0.1

    allocation = allocate_samples_from_pilot(high, low, budget, cost_ratio)
    expected = _brute_force_expected_allocation(high, low, budget, cost_ratio)

    assert allocation.use_multifidelity_estimator
    assert allocation.number_high_fidelity_samples == expected[3]
    assert allocation.number_low_fidelity_samples == expected[4]
    assert allocation.high_fidelity_equivalent_cost <= budget


@pytest.mark.mpi_skip
def test_uncorrelated_pilot_falls_back_to_high_fidelity_mc():
    high = np.array([-1.0, 1.0, -1.0, 1.0])
    low = np.array([-1.0, -1.0, 1.0, 1.0])
    allocation = allocate_samples_from_pilot(
        high, low, high_fidelity_equivalent_budget=12.0,
        low_to_high_fidelity_cost_ratio=0.1,
    )

    assert not allocation.use_multifidelity_estimator
    assert allocation.number_high_fidelity_samples == 11
    assert allocation.number_low_fidelity_samples == 4
    np.testing.assert_allclose(allocation.control_variate_coefficients, 0.0)


@pytest.mark.mpi_skip
def test_vector_allocation_uses_requested_component():
    low = np.linspace(0.0, 1.0, 5)
    high = np.column_stack([
        np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
        low + np.array([0.0, 0.01, -0.01, 0.0, 0.01]),
    ])
    low_qois = np.column_stack([low, low])

    first = allocate_samples_from_pilot(high, low_qois, 20.0, 0.1, 0)
    second = allocate_samples_from_pilot(high, low_qois, 20.0, 0.1, 1)

    assert (
        first.number_high_fidelity_samples,
        first.number_low_fidelity_samples,
    ) != (
        second.number_high_fidelity_samples,
        second.number_low_fidelity_samples,
    )


@pytest.mark.mpi_skip
def test_pilot_must_fit_in_budget():
    with pytest.raises(ValueError, match="pilot exceeds"):
        allocate_samples_from_pilot(
            np.arange(4.0), np.arange(4.0),
            high_fidelity_equivalent_budget=4.0,
            low_to_high_fidelity_cost_ratio=0.1,
        )
