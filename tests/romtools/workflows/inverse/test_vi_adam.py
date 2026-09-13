import numpy as np
import pytest

import romtools.workflows
from romtools.workflows.inverse import mf_vi_drivers, vi_drivers
from romtools.workflows.inverse.vi_optimization_methods import (
    AdamSolver,
    VIAdamOptimizerConfig,
    _resolve_optimizer_config,
)
from romtools.workflows.parameter_spaces import (
    GaussianParameterSpace,
    MultivariateGaussianParameterSpace,
    MonteCarloSampler,
)


class LinearQoiModel:
    def __init__(self, slope=1.0):
        self._slope = float(slope)

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        return

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        return np.array([self._slope * float(parameter_sample["theta"])])


class LinearQoiRomBuilderWithTrainingData:
    def __init__(self, slope=1.0):
        self._model = LinearQoiModel(slope=slope)

    def build_from_training_dirs(self, offline_data_dir, training_data_dirs,
                                 training_parameters, training_qois):
        _ = (offline_data_dir, training_data_dirs, training_parameters, training_qois)
        return self._model


class TwoParameterLinearQoiModel:
    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        return

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        return np.array([
            float(parameter_sample['theta0']) + 0.5 * float(parameter_sample['theta1'])
        ])


def _parameter_space():
    return GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )


def test_resolve_adam_optimizer_defaults_to_natural_gradient():
    method, config = _resolve_optimizer_config("adam", None)

    assert method == "adam"
    assert isinstance(config, VIAdamOptimizerConfig)
    assert config.gradient_method == "natural"
    assert config.learning_rate is None
    assert np.isclose(config.learning_rate_scale, 0.1)


def test_adam_solver_uses_abris_dimension_scaled_default_learning_rate():
    solver = AdamSolver()
    gradient = np.array([2.0, -3.0, 4.0, -5.0])

    step = solver.step(gradient)

    # Four optimizer parameters correspond to two physical parameters
    # (mean and log-standard-deviation blocks), so alpha = 0.1 / 2.
    np.testing.assert_allclose(step, 0.05 * np.sign(gradient), rtol=1e-7, atol=1e-9)


def test_adam_solver_accumulates_bias_corrected_moments():
    solver = AdamSolver(learning_rate=0.02, beta1=0.8, beta2=0.9, epsilon=1e-12)
    first_gradient = np.array([1.0, -2.0])
    second_gradient = np.array([3.0, 4.0])

    solver.step(first_gradient)
    second_step = solver.step(second_gradient)

    first_moment = 0.8 * (0.2 * first_gradient) + 0.2 * second_gradient
    second_moment = 0.9 * (0.1 * first_gradient ** 2) + 0.1 * second_gradient ** 2
    first_moment_hat = first_moment / (1.0 - 0.8 ** 2)
    second_moment_hat = second_moment / (1.0 - 0.9 ** 2)
    expected = 0.02 * first_moment_hat / np.sqrt(second_moment_hat)

    np.testing.assert_allclose(second_step, expected)


def test_adam_solver_applies_abris_fisher_damping_before_moments():
    solver = AdamSolver(learning_rate=0.01)
    gradient = np.array([4.0, 6.0])
    fisher_diagonal = np.array([2.0, 3.0])

    solver.step(gradient, fisher_diagonal=fisher_diagonal)

    expected_preconditioned = gradient / (fisher_diagonal + 1e-2)
    np.testing.assert_allclose(
        solver.first_moment,
        (1.0 - solver.beta1) * expected_preconditioned,
    )


def test_adam_solver_clips_preconditioned_gradient_norm():
    solver = AdamSolver(learning_rate=0.01, gradient_clip_norm=5.0)
    gradient = np.array([30.0, 40.0])

    solver.step(gradient)

    prepared_gradient = solver.first_moment / (1.0 - solver.beta1)
    assert np.isclose(np.linalg.norm(prepared_gradient), 5.0)


def test_adam_solver_restart_round_trip_preserves_state_and_schedule():
    solver = AdamSolver(learning_rate=0.02, fisher_damping_decay_start=2)
    solver.step(np.array([1.0, -2.0]), fisher_diagonal=np.array([2.0, 2.0]))
    solver.step(np.array([3.0, 4.0]), fisher_diagonal=np.array([2.0, 2.0]))
    restart_data = solver.restart_state_dict()

    restored = AdamSolver(learning_rate=0.02, fisher_damping_decay_start=2)
    restored.load_restart_state_dict(restart_data)

    assert restored.iteration == solver.iteration
    np.testing.assert_allclose(restored.first_moment, solver.first_moment)
    np.testing.assert_allclose(restored.second_moment, solver.second_moment)
    assert np.isclose(restored._current_fisher_damping(), solver._current_fisher_damping())


@pytest.mark.mpi_skip
def test_run_vi_supports_adam_natural_gradient(tmp_path):
    means, stds, parameter_samples, qois = vi_drivers.run_vi(
        model=LinearQoiModel(),
        prior_parameter_space=_parameter_space(),
        observations=np.array([1.0]),
        observations_covariance=np.array([[0.25]]),
        absolute_work_dir=str(tmp_path / "vi_adam"),
        sample_size=8,
        optimizer_method="adam",
        optimizer_config=VIAdamOptimizerConfig(
            learning_rate=0.03,
            gradient_norm_tolerance=0.0,
            max_iterations=2,
        ),
        bounded_parameter_handling="clip",
        random_seed=4,
        evaluation_concurrency=1,
    )

    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(stds))
    assert parameter_samples.shape[1] == 1
    assert qois.shape[0] == 1


@pytest.mark.mpi_skip
def test_run_mf_vi_supports_adam_natural_gradient(tmp_path):
    means, stds, parameter_samples, qois = mf_vi_drivers.run_mf_vi(
        model=LinearQoiModel(),
        rom_model_builder=LinearQoiRomBuilderWithTrainingData(),
        prior_parameter_space=_parameter_space(),
        observations=np.array([1.0]),
        observations_covariance=np.array([[0.25]]),
        absolute_work_dir=str(tmp_path / "mf_vi_adam"),
        fom_sample_size=4,
        rom_extra_sample_size=4,
        rom_tolerance=1.0,
        optimizer_method="adam",
        optimizer_config=VIAdamOptimizerConfig(
            learning_rate=0.03,
            gradient_norm_tolerance=0.0,
            max_iterations=2,
        ),
        bounded_parameter_handling="clip",
        random_seed=4,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
    )

    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(stds))
    assert parameter_samples.shape[1] == 1
    assert qois.shape[0] == 1


def test_adam_rejects_non_adam_config():
    with pytest.raises(TypeError, match="VIAdamOptimizerConfig"):
        romtools.workflows.run_vi(
            model=LinearQoiModel(),
            prior_parameter_space=_parameter_space(),
            observations=np.array([0.0]),
            observations_covariance=np.eye(1),
            optimizer_method="adam",
            optimizer_config=object(),
        )


@pytest.mark.mpi_skip
def test_run_vi_adam_restart_matches_uninterrupted_run(tmp_path):
    common = dict(
        model=LinearQoiModel(),
        prior_parameter_space=_parameter_space(),
        observations=np.array([0.4]),
        observations_covariance=np.array([[0.2]]),
        sample_size=8,
        optimizer_method="adam",
        bounded_parameter_handling="clip",
        random_seed=11,
        evaluation_concurrency=1,
    )
    final_config = VIAdamOptimizerConfig(
        learning_rate=0.02,
        gradient_norm_tolerance=0.0,
        max_iterations=4,
    )
    uninterrupted = vi_drivers.run_vi(
        absolute_work_dir=str(tmp_path / "uninterrupted"),
        optimizer_config=final_config,
        **common,
    )
    vi_drivers.run_vi(
        absolute_work_dir=str(tmp_path / "split"),
        optimizer_config=VIAdamOptimizerConfig(
            learning_rate=0.02,
            gradient_norm_tolerance=0.0,
            max_iterations=2,
        ),
        **common,
    )
    restarted = vi_drivers.run_vi(
        absolute_work_dir=str(tmp_path / "split"),
        restart_file=str(tmp_path / "split" / "iteration_1" / "restart.npz"),
        optimizer_config=final_config,
        **common,
    )

    for uninterrupted_value, restarted_value in zip(uninterrupted, restarted):
        np.testing.assert_allclose(uninterrupted_value, restarted_value)


@pytest.mark.mpi_skip
def test_run_mf_vi_adam_restart_matches_uninterrupted_run(tmp_path):
    common = dict(
        model=LinearQoiModel(),
        rom_model_builder=LinearQoiRomBuilderWithTrainingData(),
        prior_parameter_space=_parameter_space(),
        observations=np.array([0.4]),
        observations_covariance=np.array([[0.2]]),
        fom_sample_size=4,
        rom_extra_sample_size=4,
        rom_tolerance=1.0,
        optimizer_method="adam",
        bounded_parameter_handling="clip",
        random_seed=11,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
    )
    final_config = VIAdamOptimizerConfig(
        learning_rate=0.02,
        gradient_norm_tolerance=0.0,
        max_iterations=4,
    )
    uninterrupted = mf_vi_drivers.run_mf_vi(
        absolute_work_dir=str(tmp_path / "mf_uninterrupted"),
        optimizer_config=final_config,
        **common,
    )
    mf_vi_drivers.run_mf_vi(
        absolute_work_dir=str(tmp_path / "mf_split"),
        optimizer_config=VIAdamOptimizerConfig(
            learning_rate=0.02,
            gradient_norm_tolerance=0.0,
            max_iterations=2,
        ),
        **common,
    )
    restarted = mf_vi_drivers.run_mf_vi(
        absolute_work_dir=str(tmp_path / "mf_split"),
        restart_file=str(tmp_path / "mf_split" / "iteration_1" / "restart.npz"),
        optimizer_config=final_config,
        **common,
    )

    for uninterrupted_value, restarted_value in zip(uninterrupted, restarted):
        np.testing.assert_allclose(uninterrupted_value, restarted_value)


@pytest.mark.mpi_skip
def test_adam_natural_gradient_rejects_correlated_multivariate_vi(tmp_path):
    parameter_space = MultivariateGaussianParameterSpace(
        parameter_names=["theta0", "theta1"],
        means=np.array([0.0, 0.0]),
        covariance=np.array([[1.0, 0.4], [0.4, 1.0]]),
        sampler=MonteCarloSampler,
    )
    with pytest.raises(NotImplementedError, match="correlated multivariate"):
        vi_drivers.run_vi(
            model=TwoParameterLinearQoiModel(),
            prior_parameter_space=parameter_space,
            observations=np.array([0.0]),
            observations_covariance=np.eye(1),
            absolute_work_dir=str(tmp_path / "multivariate"),
            sample_size=4,
            optimizer_method="adam",
            optimizer_config=VIAdamOptimizerConfig(),
            bounded_parameter_handling="clip",
            evaluation_concurrency=1,
        )


@pytest.mark.mpi_skip
def test_adam_rejects_explicit_line_search_config(tmp_path):
    with pytest.raises(ValueError, match="line_search_config is not supported"):
        vi_drivers.run_vi(
            model=LinearQoiModel(),
            prior_parameter_space=_parameter_space(),
            observations=np.array([0.0]),
            observations_covariance=np.eye(1),
            absolute_work_dir=str(tmp_path / "line_search"),
            optimizer_method="adam",
            optimizer_config=VIAdamOptimizerConfig(max_iterations=1),
            line_search_config=romtools.workflows.VIStochasticNonmonotoneLineSearchConfig(),
        )


@pytest.mark.mpi_skip
def test_adam_diagnostics_report_learning_rate_not_dummy_step_size(tmp_path, capsys):
    vi_drivers.run_vi(
        model=LinearQoiModel(),
        prior_parameter_space=_parameter_space(),
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
        absolute_work_dir=str(tmp_path / "diagnostics"),
        sample_size=4,
        optimizer_method="adam",
        optimizer_config=VIAdamOptimizerConfig(max_iterations=1),
        bounded_parameter_handling="clip",
        evaluation_concurrency=1,
    )
    output = capsys.readouterr().out
    assert "Adam learning rate:" in output
    assert "Adam input gradient norm:" in output
    assert "Step size:" not in output
