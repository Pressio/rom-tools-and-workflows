import os

import numpy as np
import pytest

from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows.uq import run_monte_carlo, run_multifidelity_monte_carlo


class DeterministicParameterSpace(ParameterSpace):
    def get_names(self):
        return ["x"]

    def get_dimensionality(self):
        return 1

    def generate_samples(self, number_of_samples, seed=None):
        return np.random.default_rng(seed).uniform(-1.0, 1.0, (number_of_samples, 1))


class AnalyticQoiModel:
    def __init__(self, linear=1.0, quadratic=0.0, constant=0.0):
        self.linear = linear
        self.quadratic = quadratic
        self.constant = constant

    def populate_run_directory(self, run_directory, parameter_sample):
        pass

    def run_model(self, run_directory, parameter_sample):
        return 0

    def compute_qoi(self, run_directory, parameter_sample):
        x = float(parameter_sample["x"])
        return np.array([
            self.constant + self.linear * x + self.quadratic * x * x,
            x + 1.0,
        ])


class CountingQoiModel(AnalyticQoiModel):
    def run_model(self, run_directory, parameter_sample):
        with open(os.path.join(run_directory, "runs.txt"), "a", encoding="utf-8") as file:
            file.write("run\n")
        return 0


class FailingQoiModel(AnalyticQoiModel):
    def compute_qoi(self, run_directory, parameter_sample):
        raise RuntimeError("cannot compute QoI")


@pytest.mark.mpi_skip
def test_monte_carlo_workflow_and_persistence(tmp_path):
    result = run_monte_carlo(
        AnalyticQoiModel(),
        DeterministicParameterSpace(),
        str(tmp_path),
        number_of_samples=8,
        random_seed=7,
    )

    expected_qois = np.column_stack([
        result.parameter_samples[:, 0], result.parameter_samples[:, 0] + 1.0
    ])
    np.testing.assert_allclose(result.qoi_values, expected_qois)
    np.testing.assert_allclose(result.mean, np.mean(expected_qois, axis=0))
    np.testing.assert_allclose(result.variance, np.var(expected_qois, axis=0, ddof=1))
    assert os.path.isfile(tmp_path / "uq_stats.npz")


@pytest.mark.mpi_skip
def test_monte_carlo_parallel_order_and_restart(tmp_path):
    first = run_monte_carlo(
        CountingQoiModel(),
        DeterministicParameterSpace(),
        str(tmp_path),
        number_of_samples=6,
        random_seed=5,
        evaluation_concurrency=2,
    )
    restarted = run_monte_carlo(
        CountingQoiModel(),
        DeterministicParameterSpace(),
        str(tmp_path),
        number_of_samples=6,
        random_seed=5,
        evaluation_concurrency=2,
    )

    np.testing.assert_allclose(restarted.qoi_values, first.qoi_values)
    for sample_index in range(6):
        with open(tmp_path / f"run_{sample_index}" / "runs.txt", encoding="utf-8") as file:
            assert file.readlines() == ["run\n"]


@pytest.mark.mpi_skip
def test_qoi_failure_does_not_write_passed_marker(tmp_path):
    with pytest.raises(RuntimeError, match="cannot compute QoI"):
        run_monte_carlo(
            FailingQoiModel(),
            DeterministicParameterSpace(),
            str(tmp_path),
            number_of_samples=2,
        )

    assert not os.path.exists(tmp_path / "run_0" / "passed.txt")


@pytest.mark.mpi_skip
def test_fixed_multifidelity_workflow(tmp_path):
    result = run_multifidelity_monte_carlo(
        AnalyticQoiModel(linear=1.0),
        AnalyticQoiModel(linear=1.0),
        DeterministicParameterSpace(),
        str(tmp_path),
        number_of_high_fidelity_samples=4,
        number_of_low_fidelity_samples=10,
        random_seed=4,
        low_to_high_fidelity_cost_ratio=0.1,
    )

    np.testing.assert_allclose(
        result.mean, np.mean(result.low_fidelity_qoi_values, axis=0)
    )
    np.testing.assert_allclose(result.control_variate_coefficients, 1.0)
    assert result.used_multifidelity_estimator
    assert result.high_fidelity_sample_count == 4
    assert result.low_fidelity_sample_count == 10


@pytest.mark.mpi_skip
def test_pilot_allocation_respects_budget(tmp_path):
    result = run_multifidelity_monte_carlo(
        AnalyticQoiModel(linear=1.0, quadratic=0.02),
        AnalyticQoiModel(linear=1.0),
        DeterministicParameterSpace(),
        str(tmp_path),
        pilot_sample_count=4,
        high_fidelity_equivalent_budget=20.0,
        low_to_high_fidelity_cost_ratio=0.1,
        allocation_qoi_index=0,
        random_seed=9,
    )

    assert result.used_multifidelity_estimator
    assert result.pilot_sample_count == 4
    assert result.high_fidelity_equivalent_cost <= 20.0
    assert result.low_fidelity_sample_count >= result.high_fidelity_sample_count
    stats = np.load(tmp_path / "uq_stats.npz")
    assert "pilot_correlations" in stats
    assert stats["requested_high_fidelity_equivalent_budget"][0] == 20.0


@pytest.mark.mpi_skip
def test_pilot_allocation_can_fall_back_to_mc(tmp_path):
    class UncorrelatedHighModel(AnalyticQoiModel):
        def compute_qoi(self, run_directory, parameter_sample):
            x = float(parameter_sample["x"])
            return np.array([x * x, x * x])

    result = run_multifidelity_monte_carlo(
        UncorrelatedHighModel(),
        AnalyticQoiModel(linear=1.0),
        DeterministicParameterSpace(),
        str(tmp_path),
        pilot_sample_count=4,
        high_fidelity_equivalent_budget=60.0,
        low_to_high_fidelity_cost_ratio=10.0,
        random_seed=2,
    )

    assert not result.used_multifidelity_estimator
    np.testing.assert_allclose(result.control_variate_coefficients, 0.0)


@pytest.mark.mpi_skip
def test_allocation_modes_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_multifidelity_monte_carlo(
            AnalyticQoiModel(),
            AnalyticQoiModel(),
            DeterministicParameterSpace(),
            str(tmp_path),
            number_of_high_fidelity_samples=2,
            number_of_low_fidelity_samples=4,
            pilot_sample_count=2,
            high_fidelity_equivalent_budget=10.0,
        )
