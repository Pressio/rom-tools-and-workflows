import numpy as np
import pytest

import romtools.workflows
from romtools.workflows.inverse import vi_adaptive_sampling
from romtools.workflows.parameter_spaces import GaussianParameterSpace, MonteCarloSampler


class CountingLinearQoiModel:
    def __init__(self):
        self.run_model_calls = 0

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        return

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        self.run_model_calls += 1
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        return np.array([float(parameter_sample["theta"])])


def _parameter_space():
    return GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )


def _newton_config():
    return romtools.workflows.VINewtonOptimizerConfig(
        gradient_norm_tolerance=0.0,
        max_iterations=1,
        max_mean_update_std=0.5,
        newton_metric="natural",
        newton_regularization=1.0e-2,
        newton_hessian_type="full",
    )


def test_delete_one_jackknife_covariance_matches_classical_formula():
    steps = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    covariance = vi_adaptive_sampling._jackknife_covariance(steps)
    expected = np.array([[4.0 / 3.0, 8.0 / 3.0], [8.0 / 3.0, 16.0 / 3.0]])
    np.testing.assert_allclose(covariance, expected)


def test_adaptive_sample_config_validation():
    with pytest.raises(ValueError, match="growth_factor"):
        romtools.workflows.VIAdaptiveSampleConfig(growth_factor=1.0)
    with pytest.raises(ValueError, match="max_sample_size"):
        romtools.workflows.VIAdaptiveSampleConfig(max_sample_size=1)


@pytest.mark.mpi_skip
def test_vi_adaptive_sampling_appends_samples_until_maximum(tmp_path):
    model = CountingLinearQoiModel()
    romtools.workflows.run_vi(
        model=model,
        prior_parameter_space=_parameter_space(),
        observations=np.array([1.0]),
        observations_covariance=np.array([[0.2]]),
        absolute_work_dir=str(tmp_path),
        sample_size=8,
        optimizer_method="newton",
        optimizer_config=_newton_config(),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1.0,
            max_step_size=1.0,
            step_size_growth_factor=1.0,
            relaxation_parameter=1.0e12,
        ),
        adaptive_sample_config=romtools.workflows.VIAdaptiveSampleConfig(
            max_sample_size=16,
            growth_factor=2.0,
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
            step_norm_floor=0.0,
        ),
        bounded_parameter_handling="clip",
        score_function_entropy_strategy="joint",
        create_run_directories=False,
        random_seed=3,
        evaluation_concurrency=1,
    )
    assert model.run_model_calls == 16


@pytest.mark.mpi_skip
def test_vi_adaptive_sampling_does_not_enrich_when_variance_is_acceptable(tmp_path):
    model = CountingLinearQoiModel()
    romtools.workflows.run_vi(
        model=model,
        prior_parameter_space=_parameter_space(),
        observations=np.array([1.0]),
        observations_covariance=np.array([[0.2]]),
        absolute_work_dir=str(tmp_path),
        sample_size=8,
        optimizer_method="newton",
        optimizer_config=_newton_config(),
        adaptive_sample_config=romtools.workflows.VIAdaptiveSampleConfig(
            max_sample_size=16,
            growth_factor=2.0,
            relative_tolerance=1.0e12,
            absolute_tolerance=0.0,
        ),
        bounded_parameter_handling="clip",
        score_function_entropy_strategy="joint",
        create_run_directories=False,
        random_seed=3,
        evaluation_concurrency=1,
    )
    assert model.run_model_calls == 8


def test_adaptive_sampling_rejects_sample_reuse_combination():
    with pytest.raises(NotImplementedError, match="sample_reuse_config"):
        romtools.workflows.run_vi(
            model=CountingLinearQoiModel(),
            prior_parameter_space=_parameter_space(),
            observations=np.array([1.0]),
            observations_covariance=np.array([[0.2]]),
            sample_size=8,
            optimizer_method="newton",
            optimizer_config=_newton_config(),
            adaptive_sample_config=romtools.workflows.VIAdaptiveSampleConfig(),
            sample_reuse_config=romtools.workflows.VISampleReuseConfig(),
            create_run_directories=False,
        )
