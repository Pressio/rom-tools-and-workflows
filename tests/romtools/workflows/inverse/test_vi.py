import numpy as np
import pytest

import romtools.workflows
from romtools.workflows.inverse._inverse_utils import run_vi_iteration
from romtools.workflows.inverse import vi_drivers
from romtools.workflows.parameter_spaces import (
    GaussianParameterSpace,
    MultivariateGaussianParameterSpace,
    MonteCarloSampler,
)


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


class CountingLinearQoiModel(LinearQoiModel):
    def __init__(self, slope: float):
        super().__init__(slope=slope)
        self.run_model_calls = 0

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        self.run_model_calls += 1
        return 0


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


def test_arctan_transform_round_trip():
    parameter_mins = np.array([-2.0, -1.0])
    parameter_maxes = np.array([3.0, 2.0])
    optimizer_values = np.array([[-4.0, -0.5], [0.0, 0.8], [2.5, 3.0]])
    transformed = vi_drivers._transform_optimizer_to_parameter(
        optimizer_values,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin=1e-6,
        transform_map="arctan",
    )
    recovered = vi_drivers._transform_parameter_to_optimizer(
        transformed,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin=1e-6,
        transform_map="arctan",
    )
    assert np.allclose(recovered, optimizer_values, atol=1e-10, rtol=1e-10)


def test_run_vi_rejects_removed_legacy_kwargs():
    model = LinearQoiModel(slope=2.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )

    with pytest.raises(TypeError, match="unexpected keyword argument 'max_iterations'"):
        romtools.workflows.run_vi(
            model=model,
            prior_parameter_space=variational_parameter_space,
            observations=np.array([0.0]),
            observations_covariance=np.eye(1),
            max_iterations=10,
        )


@pytest.mark.mpi_skip
def test_run_vi_iteration_avoids_extra_mean_evaluation(tmp_path):
    model = CountingLinearQoiModel(slope=2.0)
    observations = np.array([1.2])
    parameter_names = ["theta"]
    parameter_samples = np.array([[-0.4], [0.1], [0.7], [1.1]])

    results = run_vi_iteration(
        model=model,
        observations=observations,
        run_directory_base=f"{tmp_path}/run_",
        parameter_names=parameter_names,
        parameter_samples=parameter_samples,
        evaluation_concurrency=1,
    )

    assert model.run_model_calls == parameter_samples.shape[0]
    assert results["qois"].shape == (1, parameter_samples.shape[0])
    assert results["errors"].shape == (1, parameter_samples.shape[0])
    assert np.allclose(results["mean-qoi"], np.mean(results["qois"], axis=1))


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "optimizer_method, optimizer_config",
    [
        pytest.param(
            "gradient",
            romtools.workflows.VIGradientOptimizerConfig(
                gradient_method="natural",
                gradient_norm_tolerance=5e-3,
                max_iterations=30,
                min_variational_std=1e-4,
                max_variational_std=2.0,
            ),
            id="natural_gradient",
        ),
        pytest.param(
            "newton",
            romtools.workflows.VINewtonOptimizerConfig(
                gradient_norm_tolerance=5e-3,
                max_iterations=30,
                min_variational_std=1e-4,
                max_variational_std=2.0,
                newton_metric="natural",
            ),
            id="newton_natural",
        ),
        pytest.param(
            "newton",
            romtools.workflows.VINewtonOptimizerConfig(
                gradient_norm_tolerance=5e-3,
                max_iterations=30,
                min_variational_std=1e-4,
                max_variational_std=2.0,
                newton_metric="standard",
            ),
            id="newton_whitened_natural",
        ),
    ],
)
def test_run_vi_linear_problem(tmp_path, optimizer_method, optimizer_config):
    model = LinearQoiModel(slope=2.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )

    observed_theta = 0.6
    observations = np.array([2.0 * observed_theta])
    observations_covariance = np.array([[0.1**2]])
    initial_objective = 0.5 * (observations[0] - 2.0 * (-1.0))**2 / observations_covariance[0, 0]

    means, stds, _, qois = romtools.workflows.run_vi(
        model=model,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=np.array([-2.0]),
        parameter_maxes=np.array([2.0]),
        absolute_vi_directory=str(tmp_path),
        sample_size=48,
        optimizer_method=optimizer_method,
        optimizer_config=optimizer_config,
        line_search_method="stochastic_nonmonotone",
        line_search_config=romtools.workflows.VIStochasticNonmonotoneLineSearchConfig(
            initial_step_size=5e-2,
            step_size_growth_factor=1.05,
            step_size_decay_factor=2.0,
            max_step_size_decrease_trys=10,
            line_search_armijo_coefficient=1e-4,
            line_search_uncertainty_sigma=1.0,
        ),
        random_seed=3,
        evaluation_concurrency=1,
    )

    assert means.shape == (1,)
    assert stds.shape == (1,)
    assert qois.shape[0] == 1
    assert qois.shape[1] >= 48
    losses = 0.5 * (qois[0] - observations[0])**2 / observations_covariance[0, 0]
    assert np.isfinite(np.mean(losses))
    assert np.mean(losses) < initial_objective
    assert 1e-4 <= stds[0] <= 2.0
    restart_paths = sorted(tmp_path.glob("iteration_*/restart.npz"))
    assert restart_paths
    restart_file = restart_paths[-1]
    stats_file = restart_file.parent / "stats.txt"
    assert stats_file.exists()
    stats_text = stats_file.read_text()
    assert "elbo:" in stats_text
    assert "variational_mean:" in stats_text
    assert "variational_covariance:" in stats_text
    assert "mean_log_likelihood:" in stats_text
    assert "mean_log_prior:" in stats_text
    assert "mean_relative_mse:" in stats_text
    assert "cpu_time_seconds:" in stats_text
    with np.load(restart_file) as restart:
        assert "log_likelihoods" in restart
        assert "log_priors" in restart
        assert "mean_relative_mse" in restart
        assert "prior_mean" in restart
        assert "prior_covariance" in restart
        assert "variational_mean" in restart
        assert "variational_mean_coordinates" in restart
        assert str(restart["variational_mean_coordinates"].item()) == "physical"
        assert "variational_log_std" in restart
        assert "iteration" in restart
        assert "step_size" in restart
        assert "rng_state" in restart
        physical_variational_mean = restart["variational_mean"].copy()
    assert np.all(np.isfinite(physical_variational_mean))
    assert physical_variational_mean.shape == (1,)
    assert np.all(physical_variational_mean >= -2.0)
    assert np.all(physical_variational_mean <= 2.0)
    history_file = tmp_path / "history.npz"
    assert history_file.exists()
    with np.load(history_file) as history:
        assert "vi_history_variational_mean" in history
        assert "vi_history_variational_covariance" in history
        assert "vi_history_relative_mse" in history
        assert "vi_history_loglikelihood" in history
        assert "vi_history_cpu_time_seconds" in history
        num_entries = history["vi_history_cpu_time_seconds"].shape[0]
        assert num_entries >= 1
        assert history["vi_history_variational_mean"].shape[0] == num_entries
        assert history["vi_history_variational_covariance"].shape[0] == num_entries
        assert history["vi_history_relative_mse"].shape[0] == num_entries
        assert history["vi_history_loglikelihood"].shape[0] == num_entries


@pytest.mark.mpi_skip
def test_run_vi_uses_distinct_prior_and_initial_variational_spaces(tmp_path):
    model = LinearQoiModel(slope=1.0)
    prior_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.2]),
        stds=np.array([0.4]),
        sampler=MonteCarloSampler,
    )
    initial_variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([-0.6]),
        stds=np.array([0.1]),
        sampler=MonteCarloSampler,
    )

    romtools.workflows.run_vi(
        model=model,
        prior_parameter_space=prior_parameter_space,
        initial_variational_parameter_space=initial_variational_parameter_space,
        observations=np.array([0.0]),
        observations_covariance=np.array([[0.2**2]]),
        absolute_vi_directory=str(tmp_path),
        sample_size=4,
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=0,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
        ),
        bounded_parameter_handling="clip",
        random_seed=2,
        evaluation_concurrency=1,
    )

    with np.load(tmp_path / "iteration_0" / "restart.npz") as restart:
        assert np.allclose(restart["prior_mean"], np.array([0.2]))
        assert np.allclose(restart["prior_covariance"], np.array([[0.16]]))
        assert np.allclose(restart["variational_mean"], np.array([-0.6]))


@pytest.mark.mpi_skip
def test_run_vi_accepts_full_newton_hessian_option(tmp_path):
    model = LinearQoiModel(slope=1.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )

    means, stds, parameter_samples, qois = romtools.workflows.run_vi(
        model=model,
        prior_parameter_space=variational_parameter_space,
        observations=np.array([0.0]),
        observations_covariance=np.array([[0.2**2]]),
        absolute_vi_directory=str(tmp_path),
        sample_size=8,
        optimizer_method="newton",
        optimizer_config=romtools.workflows.VINewtonOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=1,
            newton_hessian_type="full",
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
        ),
        bounded_parameter_handling="clip",
        random_seed=4,
        evaluation_concurrency=1,
    )

    assert means.shape == (1,)
    assert stds.shape == (1,)
    assert parameter_samples.shape[1] == 1
    assert qois.shape[0] == 1


@pytest.mark.mpi_skip
def test_run_vi_multivariate_newton_supported(tmp_path):
    model = TwoParameterLinearQoiModel()
    variational_parameter_space = MultivariateGaussianParameterSpace(
        parameter_names=["theta0", "theta1"],
        means=np.array([0.0, 0.0]),
        covariance=np.array([[0.4, 0.1], [0.1, 0.3]]),
        sampler=MonteCarloSampler,
    )
    observations = np.array([0.2, -0.1])
    observations_covariance = np.diag([0.05**2, 0.08**2])

    means, stds, parameter_samples, qois = romtools.workflows.run_vi(
        model=model,
        prior_parameter_space=variational_parameter_space,
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


@pytest.mark.mpi_skip
def test_run_vi_saves_elbo_relative_tolerance_in_restart(tmp_path):
    model = LinearQoiModel(slope=2.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )
    observations = np.array([0.3])
    observations_covariance = np.array([[0.1**2]])
    elbo_relative_tolerance = 1e-3

    romtools.workflows.run_vi(
        model=model,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        absolute_vi_directory=str(tmp_path),
        sample_size=8,
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=1,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
        ),
        bounded_parameter_handling="clip",
        random_seed=7,
        sampling_method="rqmc",
        evaluation_concurrency=1,
        elbo_relative_tolerance=elbo_relative_tolerance,
    )

    restart_file = tmp_path / "iteration_0" / "restart.npz"
    assert restart_file.exists()
    with np.load(restart_file) as restart:
        assert "elbo_relative_tolerance" in restart
        assert np.isclose(float(restart["elbo_relative_tolerance"]), elbo_relative_tolerance)
        assert "sampling_method" in restart
        assert str(restart["sampling_method"].item()) == "rqmc"


@pytest.mark.mpi_skip
def test_run_vi_accepts_arctan_transform_map(tmp_path):
    model = LinearQoiModel(slope=1.5)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([0.7]),
        sampler=MonteCarloSampler,
    )
    observations = np.array([0.5])
    observations_covariance = np.array([[0.15**2]])

    means, stds, parameter_samples, qois = romtools.workflows.run_vi(
        model=model,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=np.array([-2.0]),
        parameter_maxes=np.array([2.0]),
        absolute_vi_directory=str(tmp_path),
        sample_size=10,
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=1,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
        ),
        bounded_parameter_handling="transform",
        transform_map="arctan",
        random_seed=9,
        evaluation_concurrency=1,
    )

    assert means.shape == (1,)
    assert stds.shape == (1,)
    assert parameter_samples.shape[1] == 1
    assert qois.shape[0] == 1
    assert np.all(np.isfinite(parameter_samples))


@pytest.mark.mpi_skip
def test_run_vi_restart_continues_optimization_with_physical_restart_mean(tmp_path):
    model = LinearQoiModel(slope=1.5)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([0.7]),
        sampler=MonteCarloSampler,
    )
    observations = np.array([0.5])
    observations_covariance = np.array([[0.15**2]])

    romtools.workflows.run_vi(
        model=model,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=np.array([-2.0]),
        parameter_maxes=np.array([2.0]),
        absolute_vi_directory=str(tmp_path),
        sample_size=10,
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=1,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
        ),
        bounded_parameter_handling="transform",
        random_seed=9,
        evaluation_concurrency=1,
    )

    restart_file = tmp_path / "iteration_0" / "restart.npz"
    with np.load(restart_file) as restart:
        assert str(restart["variational_mean_coordinates"].item()) == "physical"
        assert np.all(restart["variational_mean"] >= -2.0)
        assert np.all(restart["variational_mean"] <= 2.0)

    means, stds, parameter_samples, qois = romtools.workflows.run_vi(
        model=model,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=np.array([-2.0]),
        parameter_maxes=np.array([2.0]),
        absolute_vi_directory=str(tmp_path),
        restart_file=str(restart_file),
        sample_size=10,
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=2,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
        ),
        bounded_parameter_handling="transform",
        random_seed=9,
        evaluation_concurrency=1,
    )

    assert means.shape == (1,)
    assert stds.shape == (1,)
    assert parameter_samples.shape[1] == 1
    assert qois.shape[0] == 1
    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(stds))
    with np.load(tmp_path / "iteration_1" / "restart.npz") as restart:
        assert str(restart["variational_mean_coordinates"].item()) == "physical"
        assert np.all(np.isfinite(restart["variational_mean"]))
        assert np.all(restart["variational_mean"] >= -2.0)
        assert np.all(restart["variational_mean"] <= 2.0)


@pytest.mark.mpi_skip
def test_run_vi_limits_number_of_restart_files(tmp_path):
    model = LinearQoiModel(slope=1.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([0.5]),
        sampler=MonteCarloSampler,
    )
    observations = np.array([0.1])
    observations_covariance = np.array([[0.2**2]])

    romtools.workflows.run_vi(
        model=model,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        absolute_vi_directory=str(tmp_path),
        sample_size=8,
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=6,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
        ),
        bounded_parameter_handling="clip",
        random_seed=3,
        evaluation_concurrency=1,
        restart_files_to_keep=2,
    )

    restart_paths = sorted(tmp_path.glob("iteration_*/restart.npz"))
    assert len(restart_paths) <= 2
    remaining_iterations = sorted(int(path.parent.name.split("_")[1]) for path in restart_paths)
    if len(remaining_iterations) == 2:
        assert remaining_iterations[1] - remaining_iterations[0] == 1
