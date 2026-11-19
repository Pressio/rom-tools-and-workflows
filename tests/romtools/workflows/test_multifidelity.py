"""Integration coverage for fixed-allocation multifidelity sampling and CV."""
import numpy as np
import pytest

from romtools.rom.multifidelity import (
    AutoregressiveGPBuilder, AutoregressiveGPConfig, GPLevelConfig,
)
from romtools.rom.qoi_surrogates import GaussianProcessKernel
from romtools.workflows.multifidelity import (
    FidelityLevel, MultiFidelityKFoldValidation, run_adaptive_multifidelity,
)
from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.sampling_methods import MonteCarloSampler


class SyntheticQoiModel:
    def __init__(self, factor=1.0):
        self.factor = factor

    def populate_run_directory(self, run_directory, parameter_sample):
        pass

    def run_model(self, run_directory, parameter_sample):
        return 0

    def compute_qoi(self, run_directory, parameter_sample):
        x = parameter_sample["x"]
        return np.array([self.factor * np.sin(2.0 * x) + 0.1 * x])


def make_options(tmp_path, paired=True, budget=None, max_iterations=2):
    return dict(
        fidelities=[
            FidelityLevel(SyntheticQoiModel(1.0), name="low", cost=1.0),
            FidelityLevel(SyntheticQoiModel(1.2), name="high", cost=10.0),
        ],
        surrogate_builder=AutoregressiveGPBuilder(
            AutoregressiveGPConfig(
                levels=[
                    GPLevelConfig(GaussianProcessKernel(length_scale=0.3),
                                  noise_variance=1e-8),
                    GPLevelConfig(GaussianProcessKernel(length_scale=0.3),
                                  rho=1.0, noise_variance=1e-8),
                ],
                tune_hyperparameters=False,
            ),
            parameter_names=["x"],
        ),
        parameter_space=UniformParameterSpace(
            ["x"], [0.0], [1.0], MonteCarloSampler,
        ),
        absolute_work_dir=str(tmp_path),
        tolerance=0.0,
        initial_samples=[6, 3],
        batch_sizes=[2, 1],
        adaptive_sampling_strategy="random",
        pair_fidelity_samples=paired,
        validation_strategy=MultiFidelityKFoldValidation(num_folds=3),
        max_iterations=max_iterations,
        max_total_cost=budget,
        random_seed=31,
    )


def test_random_multifidelity_workflow_pairs_and_counts(tmp_path):
    result = run_adaptive_multifidelity(**make_options(tmp_path))
    assert result.termination_reason == "max_iterations"
    np.testing.assert_array_equal(result.sample_counts, [[6, 3], [8, 4]])
    assert result.total_cost == 48.0
    assert result.validation_errors.shape == (2,)
    assert np.all(np.isfinite(result.validation_errors))
    low, high = result.training_data
    assert low.qois.shape == (8, 1)
    assert high.qois.shape == (4, 1)
    assert set(high.parameters[:, 0]).issubset(set(low.parameters[:, 0]))
    assert all((tmp_path / "fidelity_1" / f"run_{i}" / "passed.txt").is_file()
               for i in range(4))
    assert (tmp_path / "multifidelity_stats.npz").is_file()


def test_random_multifidelity_without_pairing(tmp_path):
    result = run_adaptive_multifidelity(**make_options(tmp_path, paired=False))
    low, high = result.training_data
    assert not set(low.parameters[:, 0]).intersection(set(high.parameters[:, 0]))
    np.testing.assert_array_equal(result.sample_counts[-1], [8, 4])


def test_budget_stops_before_unaffordable_round(tmp_path):
    result = run_adaptive_multifidelity(
        **make_options(tmp_path, budget=36.0, max_iterations=4))
    assert result.termination_reason == "max_total_cost"
    np.testing.assert_array_equal(result.sample_counts, [[6, 3]])
    assert result.total_cost == 36.0
    assert not (tmp_path / "iteration_1").exists()


def test_initial_budget_is_validated_before_execution(tmp_path):
    with pytest.raises(ValueError, match="initial sample allocation"):
        run_adaptive_multifidelity(**make_options(tmp_path, budget=35.0))
    assert not tmp_path.exists()


def test_paired_counts_must_be_nonincreasing(tmp_path):
    options = make_options(tmp_path)
    options["batch_sizes"] = [1, 2]
    with pytest.raises(ValueError, match="nonincreasing"):
        run_adaptive_multifidelity(**options)


def test_three_level_random_batch_pairing_and_cost(tmp_path):
    options = make_options(tmp_path, max_iterations=1)
    options["fidelities"] = [
        FidelityLevel(SyntheticQoiModel(1), "low", 1),
        FidelityLevel(SyntheticQoiModel(1.1), "middle", 3),
        FidelityLevel(SyntheticQoiModel(1.2), "high", 10),
    ]
    options["surrogate_builder"] = AutoregressiveGPBuilder(
        AutoregressiveGPConfig(
            levels=[GPLevelConfig(GaussianProcessKernel(length_scale=0.3),
                                  noise_variance=1e-8)
                    for _ in range(3)],
            tune_hyperparameters=False,
        ),
        parameter_names=["x"],
    )
    options["initial_samples"] = [8, 5, 3]
    options["batch_sizes"] = [4, 2, 1]
    result = run_adaptive_multifidelity(**options)
    np.testing.assert_array_equal(result.sample_counts, [[8, 5, 3]])
    assert result.total_cost == 53
    for lower, higher in zip(result.training_data[:-1], result.training_data[1:]):
        assert set(higher.parameters[:, 0]).issubset(set(lower.parameters[:, 0]))
