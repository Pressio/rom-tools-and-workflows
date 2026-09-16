import numpy as np
import pytest

import romtools.workflows
from romtools.workflows.parameter_spaces import GaussianParameterSpace, MonteCarloSampler


class LinearQoiModel:
    def __init__(self, slope: float):
        self._slope = float(slope)

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        return

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        theta = float(parameter_sample["theta"])
        return np.array([self._slope * theta])


class LinearQoiRomBuilderWithTrainingData:
    def __init__(self, slope: float):
        self._model = LinearQoiModel(slope=slope)

    def build_from_training_dirs(
        self,
        offline_data_dir,
        training_data_dirs,
        training_parameters,
        training_qois,
    ):
        _ = offline_data_dir
        _ = training_data_dirs
        _ = training_parameters
        _ = training_qois
        return self._model


@pytest.mark.mpi_skip
@pytest.mark.parametrize("newton_metric", ["standard", "natural"])
def test_mf_vi_newton_respects_log_std_learning_rate_factor(tmp_path, newton_metric):
    initial_std = 0.7
    parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([initial_std]),
        sampler=MonteCarloSampler,
    )

    def run_case(factor: float, work_dir):
        _, stds, _, _ = romtools.workflows.run_mf_vi(
            model=LinearQoiModel(slope=1.5),
            rom_model_builder=LinearQoiRomBuilderWithTrainingData(slope=1.5),
            prior_parameter_space=parameter_space,
            initial_variational_parameter_space=parameter_space,
            observations=np.array([0.4]),
            observations_covariance=np.array([[0.2 ** 2]]),
            parameter_mins=np.array([-3.0]),
            parameter_maxes=np.array([3.0]),
            absolute_work_dir=str(work_dir),
            fom_sample_size=8,
            rom_extra_sample_size=0,
            rom_tolerance=0.0,
            optimizer_method="newton",
            optimizer_config=romtools.workflows.VINewtonOptimizerConfig(
                gradient_norm_tolerance=0.0,
                max_iterations=2,
                max_log_std_update=10.0,
                newton_metric=newton_metric,
            ),
            line_search_method="legacy",
            line_search_config=romtools.workflows.VILegacyLineSearchConfig(
                initial_step_size=1e-4,
                max_step_size=1e-4,
                step_size_growth_factor=1.0,
                relaxation_parameter=1e12,
                log_std_learning_rate_factor=factor,
            ),
            bounded_parameter_handling="clip",
            random_seed=41,
            fom_evaluation_concurrency=1,
            rom_evaluation_concurrency=1,
        )
        return stds

    full_rate_std = run_case(1.0, tmp_path / "factor_1")
    quarter_rate_std = run_case(0.25, tmp_path / "factor_quarter")

    full_rate_log_std_update = np.log(full_rate_std / initial_std)
    quarter_rate_log_std_update = np.log(quarter_rate_std / initial_std)

    assert abs(full_rate_log_std_update[0]) > 1e-10
    np.testing.assert_allclose(
        quarter_rate_log_std_update,
        0.25 * full_rate_log_std_update,
        rtol=1e-8,
        atol=1e-12,
    )
