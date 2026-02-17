import numpy as np
import pytest

import romtools.workflows


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


class SingleParameterSpace(romtools.workflows.ParameterSpace):
    def __init__(self, lower: float, upper: float):
        self._lower = float(lower)
        self._upper = float(upper)

    def get_names(self):
        return ["theta"]

    def get_dimensionality(self) -> int:
        return 1

    def generate_samples(self, number_of_samples: int, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(self._lower, self._upper, size=(number_of_samples, 1))


class TwoParameterLinearQoiModel:
    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        return

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        theta0 = float(parameter_sample["theta0"])
        theta1 = float(parameter_sample["theta1"])
        return np.array([theta0 + 0.2 * theta1, -0.3 * theta0 + 0.5 * theta1])


class TwoParameterSpace(romtools.workflows.ParameterSpace):
    def __init__(self, lower: float, upper: float):
        self._lower = float(lower)
        self._upper = float(upper)

    def get_names(self):
        return ["theta0", "theta1"]

    def get_dimensionality(self) -> int:
        return 2

    def generate_samples(self, number_of_samples: int, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(self._lower, self._upper, size=(number_of_samples, 2))


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "optimizer_method",
    ["natural_gradient", "newton_natural", "newton_whitened_natural"],
)
def test_run_vi_linear_problem(tmp_path, optimizer_method):
    model = LinearQoiModel(slope=2.0)
    parameter_space = SingleParameterSpace(lower=-2.0, upper=2.0)

    observed_theta = 0.6
    observations = np.array([2.0 * observed_theta])
    observations_covariance = np.array([[0.1**2]])
    initial_objective = 0.5 * (observations[0] - 2.0 * (-1.0))**2 / observations_covariance[0, 0]

    means, stds, _, losses = romtools.workflows.run_vi(
        model=model,
        parameter_space=parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=np.array([-2.0]),
        parameter_maxes=np.array([2.0]),
        absolute_vi_directory=str(tmp_path),
        sample_size=48,
        initial_means=np.array([-1.0]),
        initial_stds=np.array([1.0]),
        min_variational_std=1e-4,
        max_variational_std=2.0,
        optimizer_method=optimizer_method,
        initial_step_size=5e-2,
        step_size_growth_factor=1.05,
        step_size_decay_factor=2.0,
        max_step_size_decrease_trys=10,
        armijo_sufficient_decrease=1e-4,
        stochastic_acceptance_factor=1.0,
        gradient_norm_tolerance=5e-3,
        delta_params_tolerance=1e-6,
        max_line_search_iterations=8,
        max_iterations=30,
        random_seed=3,
        evaluation_concurrency=1,
    )

    assert means.shape == (1,)
    assert stds.shape == (1,)
    assert losses.shape == (48,)
    assert np.isfinite(np.mean(losses))
    assert np.mean(losses) < 0.2 * initial_objective
    assert abs(means[0] - observed_theta) < 0.6
    assert 1e-4 <= stds[0] <= 2.0
    restart_file = tmp_path / "iteration_0" / "restart.npz"
    assert restart_file.exists()
    with np.load(restart_file) as restart:
        assert "expected_loss" in restart
        assert "entropy" in restart
        assert "objective" in restart
        assert np.isclose(
            float(restart["objective"]),
            float(restart["expected_loss"] - restart["entropy"]),
        )


@pytest.mark.mpi_skip
def test_run_vi_multivariate_newton_supported(tmp_path):
    model = TwoParameterLinearQoiModel()
    parameter_space = TwoParameterSpace(lower=-1.0, upper=1.0)
    observations = np.array([0.2, -0.1])
    observations_covariance = np.diag([0.05**2, 0.08**2])

    means, stds, parameter_samples, qois = romtools.workflows.run_vi(
        model=model,
        parameter_space=parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        absolute_vi_directory=str(tmp_path),
        sample_size=12,
        optimizer_method="newton",
        optimizer_config=romtools.workflows.VINewtonOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=2,
        ),
        line_search_method="stochastic_nonmonotone",
        line_search_config=romtools.workflows.VIStochasticNonmonotoneLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
            line_search_armijo_coefficient=0.0,
            line_search_uncertainty_sigma=0.0,
        ),
        variational_distribution="multivariate",
        bounded_parameter_handling="clip",
        random_seed=5,
        evaluation_concurrency=1,
    )

    assert means.shape == (2,)
    assert stds.shape == (2,)
    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(stds))
    assert parameter_samples.shape[1] == 2
    assert qois.shape[0] == 2
