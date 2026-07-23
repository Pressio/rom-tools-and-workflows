import numpy as np
import pytest

import romtools.workflows
from romtools.workflows.inverse import mf_vi_drivers
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


class SingleParameterSpace(romtools.workflows.ParameterSpace):
    def __init__(self, lower: float, upper: float):
        self._lower = float(lower)
        self._upper = float(upper)

    def get_names(self):
        return ["theta"]

    def get_dimensionality(self) -> int:
        return 1

    def generate_samples(self, number_of_samples: int, seed=None) -> np.ndarray:
        _ = seed
        return np.random.uniform(self._lower, self._upper, size=(number_of_samples, 1))


class LinearQoiRomBuilderWithTrainingData:
    def __init__(self, slope: float):
        self._model = LinearQoiModel(slope=slope)

    def build_from_training_dirs(self, offline_data_dir, training_data_dirs, training_parameters, training_qois):
        _ = offline_data_dir
        _ = training_data_dirs
        _ = training_parameters
        _ = training_qois
        return self._model


def test_compute_mfmc_alpha_matrix_mode_captures_cross_component_coupling():
    low_terms = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0],
        [2.0, 2.0],
    ])
    true_alpha = np.array([
        [1.2, -0.5],
        [0.8, 1.5],
    ])
    high_terms = low_terms @ true_alpha

    alpha_componentwise = mf_vi_drivers._compute_mfmc_alpha(
        high_terms,
        low_terms,
        mode="componentwise",
    )
    alpha_matrix = mf_vi_drivers._compute_mfmc_alpha(
        high_terms,
        low_terms,
        mode="matrix",
    )

    delta = np.array([0.4, -0.7])
    exact_delta = delta @ true_alpha
    componentwise_delta = delta * alpha_componentwise
    matrix_delta = delta @ alpha_matrix
    componentwise_error = np.linalg.norm(componentwise_delta - exact_delta)
    matrix_error = np.linalg.norm(matrix_delta - exact_delta)

    assert alpha_matrix.shape == (2, 2)
    assert matrix_error < componentwise_error
    assert matrix_error < 1e-10


def test_compute_mfmc_alpha_scalar_mode_matches_isotropic_target():
    low_terms = np.array([
        [1.0, -2.0],
        [0.5, 1.5],
        [-1.0, 0.0],
        [2.0, -1.0],
        [1.5, 2.0],
        [-0.5, -1.5],
    ])
    true_alpha = 1.7
    high_terms = true_alpha * low_terms

    alpha_scalar = mf_vi_drivers._compute_mfmc_alpha(
        high_terms,
        low_terms,
        mode="scalar",
    )
    assert np.isclose(float(alpha_scalar), true_alpha, atol=1e-10)


def test_run_mf_vi_rejects_removed_legacy_kwargs():
    model = LinearQoiModel(slope=2.0)
    rom_builder = LinearQoiRomBuilderWithTrainingData(slope=2.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )

    with pytest.raises(TypeError, match="unexpected keyword argument 'max_iterations'"):
        romtools.workflows.run_mf_vi(
            model=model,
            rom_model_builder=rom_builder,
            prior_parameter_space=variational_parameter_space,
            observations=np.array([0.0]),
            observations_covariance=np.eye(1),
            max_iterations=10,
        )


@pytest.mark.mpi_skip
def test_mf_vi_with_auto_rom_forwards_prior_and_initializer(monkeypatch):
    captured = {}

    def fake_run_mf_vi(**kwargs):
        captured.update(kwargs)
        return "ok", None, None, None

    monkeypatch.setattr(mf_vi_drivers, "run_mf_vi", fake_run_mf_vi)

    prior_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )
    initial_variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.5]),
        stds=np.array([0.2]),
        sampler=MonteCarloSampler,
    )

    mf_vi_drivers.mf_vi_with_auto_rom(
        model=LinearQoiModel(slope=1.0),
        prior_parameter_space=prior_parameter_space,
        initial_variational_parameter_space=initial_variational_parameter_space,
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
    )

    assert captured["prior_parameter_space"] is prior_parameter_space
    assert captured["initial_variational_parameter_space"] is initial_variational_parameter_space
    assert captured["rom_model_builder"].parameter_names == ["theta"]


@pytest.mark.mpi_skip
def test_run_mf_vi_accepts_full_newton_hessian_option(tmp_path):
    model = LinearQoiModel(slope=2.0)
    rom_builder = LinearQoiRomBuilderWithTrainingData(slope=2.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )

    means, stds, parameter_samples, qois = romtools.workflows.run_mf_vi(
        model=model,
        rom_model_builder=rom_builder,
        prior_parameter_space=variational_parameter_space,
        observations=np.array([0.0]),
        observations_covariance=np.array([[0.2**2]]),
        parameter_mins=np.array([-2.0]),
        parameter_maxes=np.array([2.0]),
        absolute_vi_directory=str(tmp_path),
        fom_sample_size=6,
        rom_extra_sample_size=0,
        rom_tolerance=0.0,
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
        random_seed=7,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
    )

    assert means.shape == (1,)
    assert stds.shape == (1,)
    assert parameter_samples.shape[1] == 1
    assert qois.shape[0] == 1


@pytest.mark.mpi_skip
def test_run_mf_vi_restart_continues_optimization_with_minimal_restart(tmp_path):
    model = LinearQoiModel(slope=2.0)
    rom_builder = LinearQoiRomBuilderWithTrainingData(slope=2.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )
    observations = np.array([1.3])
    observations_covariance = np.array([[0.1**2]])
    optimizer_config = romtools.workflows.VIGradientOptimizerConfig(
        gradient_method="standard",
        gradient_norm_tolerance=0.0,
        max_iterations=20,
        min_variational_std=1e-4,
        max_variational_std=2.0,
    )
    line_search_config = romtools.workflows.VILegacyLineSearchConfig(
        initial_step_size=1e-2,
        max_step_size=1e-2,
        step_size_growth_factor=1.0,
        step_size_decay_factor=2.0,
        max_step_size_decrease_trys=20,
        relaxation_parameter=10.0,
        line_search_sample_growth_factor=1.0,
    )

    split_dir = tmp_path / "split"

    optimizer_config_10 = romtools.workflows.VIGradientOptimizerConfig(
        gradient_method="standard",
        gradient_norm_tolerance=0.0,
        max_iterations=10,
        min_variational_std=1e-4,
        max_variational_std=2.0,
    )
    romtools.workflows.run_mf_vi(
        model=model,
        rom_model_builder=rom_builder,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=np.array([-2.0]),
        parameter_maxes=np.array([2.0]),
        absolute_vi_directory=str(split_dir),
        fom_sample_size=6,
        rom_extra_sample_size=8,
        rom_tolerance=0.0,
        max_rom_training_history=2,
        optimizer_method="gradient",
        optimizer_config=optimizer_config_10,
        line_search_method="legacy",
        line_search_config=line_search_config,
        bounded_parameter_handling="clip",
        random_seed=11,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
    )

    restart_file = split_dir / "iteration_9" / "restart.npz"
    assert restart_file.exists()
    stats_file = split_dir / "iteration_9" / "stats.txt"
    assert stats_file.exists()
    stats_text = stats_file.read_text()
    assert "elbo:" in stats_text
    assert "variational_mean:" in stats_text
    assert "variational_covariance:" in stats_text
    assert "mean_log_likelihood:" in stats_text
    assert "mean_log_prior:" in stats_text
    assert "mean_relative_mse:" in stats_text
    assert "cpu_time_seconds:" in stats_text

    split_mean, split_std, split_parameter_samples, split_qois = romtools.workflows.run_mf_vi(
        model=model,
        rom_model_builder=rom_builder,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=np.array([-2.0]),
        parameter_maxes=np.array([2.0]),
        absolute_vi_directory=str(split_dir),
        restart_file=str(restart_file),
        fom_sample_size=6,
        rom_extra_sample_size=8,
        rom_tolerance=0.0,
        max_rom_training_history=2,
        optimizer_method="gradient",
        optimizer_config=optimizer_config,
        line_search_method="legacy",
        line_search_config=line_search_config,
        bounded_parameter_handling="clip",
        random_seed=11,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
    )

    assert (split_dir / "iteration_19" / "restart.npz").exists()
    assert split_mean.shape == (1,)
    assert split_std.shape == (1,)
    assert split_parameter_samples.shape[1] == 1
    assert split_qois.shape[0] == 1
    assert np.all(np.isfinite(split_mean))
    assert np.all(np.isfinite(split_std))

    with np.load(split_dir / "iteration_19" / "restart.npz", allow_pickle=True) as split_restart:
        for key in (
            "log_likelihoods",
            "log_priors",
            "mean_relative_mse",
            "prior_mean",
            "prior_covariance",
            "variational_mean",
            "variational_mean_coordinates",
            "variational_log_std",
            "iteration",
            "step_size",
            "training_directories",
            "rom_training_directories",
            "training_parameters",
            "training_qois",
            "rom_training_parameters",
            "rom_training_qois",
            "rng_state",
        ):
            assert key in split_restart
        assert str(split_restart["variational_mean_coordinates"].item()) == "physical"
    history_file = split_dir / "history.npz"
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
def test_run_mf_vi_accepts_matrix_control_variate_mode(tmp_path):
    model = LinearQoiModel(slope=2.0)
    rom_builder = LinearQoiRomBuilderWithTrainingData(slope=2.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )
    observations = np.array([0.6])
    observations_covariance = np.array([[0.1**2]])

    romtools.workflows.run_mf_vi(
        model=model,
        rom_model_builder=rom_builder,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=np.array([-2.0]),
        parameter_maxes=np.array([2.0]),
        absolute_vi_directory=str(tmp_path),
        fom_sample_size=4,
        rom_extra_sample_size=4,
        rom_tolerance=0.0,
        max_rom_training_history=1,
        optimizer_method="gradient",
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=1,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
            relaxation_parameter=10.0,
            line_search_sample_growth_factor=1.0,
        ),
        use_mfmc_control_variate=True,
        mfmc_control_variate_mode="matrix",
        bounded_parameter_handling="transform",
        random_seed=17,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
    )

    restart_file = tmp_path / "iteration_0" / "restart.npz"
    assert restart_file.exists()
    with np.load(restart_file, allow_pickle=True) as restart:
        assert "mfmc_control_variate_mode" in restart
        assert str(restart["mfmc_control_variate_mode"].item()) == "matrix"


@pytest.mark.mpi_skip
def test_run_mf_vi_accepts_scalar_control_variate_mode(tmp_path):
    model = LinearQoiModel(slope=2.0)
    rom_builder = LinearQoiRomBuilderWithTrainingData(slope=2.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )
    observations = np.array([0.6])
    observations_covariance = np.array([[0.1**2]])

    romtools.workflows.run_mf_vi(
        model=model,
        rom_model_builder=rom_builder,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=np.array([-2.0]),
        parameter_maxes=np.array([2.0]),
        absolute_vi_directory=str(tmp_path),
        fom_sample_size=4,
        rom_extra_sample_size=4,
        rom_tolerance=0.0,
        max_rom_training_history=1,
        optimizer_method="gradient",
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=1,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
            relaxation_parameter=10.0,
            line_search_sample_growth_factor=1.0,
        ),
        use_mfmc_control_variate=True,
        mfmc_control_variate_mode="scalar",
        bounded_parameter_handling="transform",
        random_seed=19,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
    )

    restart_file = tmp_path / "iteration_0" / "restart.npz"
    assert restart_file.exists()
    with np.load(restart_file, allow_pickle=True) as restart:
        assert "mfmc_control_variate_mode" in restart
        assert str(restart["mfmc_control_variate_mode"].item()) == "scalar"
    assert not (tmp_path / "iteration_0" / "run_fom_sample_set_0_mean").exists()
    assert not (tmp_path / "iteration_0" / "run_rom_sample_set_0_mean").exists()


@pytest.mark.mpi_skip
def test_run_mf_vi_restart_continues_with_physical_restart_mean(tmp_path):
    model = LinearQoiModel(slope=2.0)
    rom_builder = LinearQoiRomBuilderWithTrainingData(slope=2.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )
    observations = np.array([0.7])
    observations_covariance = np.array([[0.1**2]])

    romtools.workflows.run_mf_vi(
        model=model,
        rom_model_builder=rom_builder,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=np.array([-2.0]),
        parameter_maxes=np.array([2.0]),
        absolute_vi_directory=str(tmp_path),
        fom_sample_size=4,
        rom_extra_sample_size=4,
        rom_tolerance=0.0,
        max_rom_training_history=1,
        optimizer_method="gradient",
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=1,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
            relaxation_parameter=10.0,
            line_search_sample_growth_factor=1.0,
        ),
        bounded_parameter_handling="transform",
        random_seed=23,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
    )

    restart_file = tmp_path / "iteration_0" / "restart.npz"
    with np.load(restart_file, allow_pickle=True) as restart:
        assert str(restart["variational_mean_coordinates"].item()) == "physical"
        assert np.all(restart["variational_mean"] >= -2.0)
        assert np.all(restart["variational_mean"] <= 2.0)

    means, stds, parameter_samples, qois = romtools.workflows.run_mf_vi(
        model=model,
        rom_model_builder=rom_builder,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=np.array([-2.0]),
        parameter_maxes=np.array([2.0]),
        absolute_vi_directory=str(tmp_path),
        restart_file=str(restart_file),
        fom_sample_size=4,
        rom_extra_sample_size=4,
        rom_tolerance=0.0,
        max_rom_training_history=1,
        optimizer_method="gradient",
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=2,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
            relaxation_parameter=10.0,
            line_search_sample_growth_factor=1.0,
        ),
        bounded_parameter_handling="transform",
        random_seed=23,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
    )

    assert means.shape == (1,)
    assert stds.shape == (1,)
    assert parameter_samples.shape[1] == 1
    assert qois.shape[0] == 1
    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(stds))
    with np.load(tmp_path / "iteration_1" / "restart.npz", allow_pickle=True) as restart:
        assert str(restart["variational_mean_coordinates"].item()) == "physical"
        assert np.all(np.isfinite(restart["variational_mean"]))
        assert np.all(restart["variational_mean"] >= -2.0)
        assert np.all(restart["variational_mean"] <= 2.0)


@pytest.mark.mpi_skip
def test_run_mf_vi_prints_when_rom_is_retrained(tmp_path, capsys):
    model = LinearQoiModel(slope=2.0)
    rom_builder = LinearQoiRomBuilderWithTrainingData(slope=2.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )
    observations = np.array([0.7])
    observations_covariance = np.array([[0.1**2]])

    romtools.workflows.run_mf_vi(
        model=model,
        rom_model_builder=rom_builder,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=np.array([-2.0]),
        parameter_maxes=np.array([2.0]),
        absolute_vi_directory=str(tmp_path),
        fom_sample_size=4,
        rom_extra_sample_size=4,
        rom_tolerance=0.0,
        max_rom_training_history=1,
        optimizer_method="gradient",
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=1,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
            relaxation_parameter=10.0,
            line_search_sample_growth_factor=1.0,
        ),
        bounded_parameter_handling="transform",
        random_seed=29,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
    )

    stdout = capsys.readouterr().out
    assert "Retraining ROM:" in stdout
    assert "rom_tolerance=0.00000e+00" in stdout


@pytest.mark.mpi_skip
def test_run_mf_vi_accepts_arctan_transform_map(tmp_path):
    model = LinearQoiModel(slope=2.0)
    rom_builder = LinearQoiRomBuilderWithTrainingData(slope=2.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )
    observations = np.array([0.7])
    observations_covariance = np.array([[0.1**2]])

    romtools.workflows.run_mf_vi(
        model=model,
        rom_model_builder=rom_builder,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=np.array([-2.0]),
        parameter_maxes=np.array([2.0]),
        absolute_vi_directory=str(tmp_path),
        fom_sample_size=4,
        rom_extra_sample_size=4,
        rom_tolerance=0.0,
        max_rom_training_history=1,
        optimizer_method="gradient",
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=1,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
            relaxation_parameter=10.0,
            line_search_sample_growth_factor=1.0,
        ),
        bounded_parameter_handling="transform",
        transform_map="arctan",
        random_seed=23,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
    )

    restart_file = tmp_path / "iteration_0" / "restart.npz"
    assert restart_file.exists()
    with np.load(restart_file, allow_pickle=True) as restart:
        assert "transform_map" in restart
        assert str(restart["transform_map"].item()) == "arctan"


@pytest.mark.mpi_skip
def test_run_mf_vi_history_stores_physical_mean_for_transform_bounds(tmp_path):
    model = LinearQoiModel(slope=2.0)
    rom_builder = LinearQoiRomBuilderWithTrainingData(slope=2.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )
    observations = np.array([0.7])
    observations_covariance = np.array([[0.1**2]])

    romtools.workflows.run_mf_vi(
        model=model,
        rom_model_builder=rom_builder,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=np.array([-2.0]),
        parameter_maxes=np.array([2.0]),
        absolute_vi_directory=str(tmp_path),
        fom_sample_size=4,
        rom_extra_sample_size=4,
        rom_tolerance=0.0,
        max_rom_training_history=1,
        optimizer_method="gradient",
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=1,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
            relaxation_parameter=10.0,
            line_search_sample_growth_factor=1.0,
        ),
        bounded_parameter_handling="transform",
        random_seed=21,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
    )

    with np.load(tmp_path / "iteration_0" / "restart.npz", allow_pickle=True) as restart:
        assert str(restart["variational_mean_coordinates"].item()) == "physical"
        assert np.all(restart["variational_mean"] >= -2.0)
        assert np.all(restart["variational_mean"] <= 2.0)


@pytest.mark.mpi_skip
def test_run_mf_vi_saves_elbo_relative_tolerance_in_restart(tmp_path):
    model = LinearQoiModel(slope=2.0)
    rom_builder = LinearQoiRomBuilderWithTrainingData(slope=2.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )
    observations = np.array([1.0])
    observations_covariance = np.array([[0.1**2]])
    elbo_relative_tolerance = 2e-3

    romtools.workflows.run_mf_vi(
        model=model,
        rom_model_builder=rom_builder,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        absolute_vi_directory=str(tmp_path),
        fom_sample_size=4,
        rom_extra_sample_size=4,
        rom_tolerance=0.0,
        max_rom_training_history=1,
        optimizer_method="gradient",
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=1,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
            relaxation_parameter=10.0,
            line_search_sample_growth_factor=1.0,
        ),
        bounded_parameter_handling="clip",
        random_seed=13,
        sampling_method="rqmc",
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        elbo_relative_tolerance=elbo_relative_tolerance,
    )

    restart_file = tmp_path / "iteration_0" / "restart.npz"
    assert restart_file.exists()
    with np.load(restart_file, allow_pickle=True) as restart:
        assert "elbo_relative_tolerance" in restart
        assert np.isclose(float(restart["elbo_relative_tolerance"]), elbo_relative_tolerance)
        assert "sampling_method" in restart
        assert str(restart["sampling_method"].item()) == "rqmc"


@pytest.mark.mpi_skip
def test_run_mf_vi_limits_number_of_restart_files(tmp_path):
    model = LinearQoiModel(slope=2.0)
    rom_builder = LinearQoiRomBuilderWithTrainingData(slope=2.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )
    observations = np.array([0.4])
    observations_covariance = np.array([[0.1**2]])

    romtools.workflows.run_mf_vi(
        model=model,
        rom_model_builder=rom_builder,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        absolute_vi_directory=str(tmp_path),
        fom_sample_size=4,
        rom_extra_sample_size=4,
        rom_tolerance=0.0,
        max_rom_training_history=1,
        optimizer_method="gradient",
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=6,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
            relaxation_parameter=10.0,
            line_search_sample_growth_factor=1.0,
        ),
        bounded_parameter_handling="clip",
        random_seed=13,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        restart_files_to_keep=2,
    )

    restart_paths = sorted(tmp_path.glob("iteration_*/restart.npz"))
    assert len(restart_paths) <= 2
    remaining_iterations = sorted(int(path.parent.name.split("_")[1]) for path in restart_paths)
    if len(remaining_iterations) == 2:
        assert remaining_iterations[1] - remaining_iterations[0] == 1


@pytest.mark.mpi_skip
def test_run_mf_vi_kfold_correlation_estimator_runs(tmp_path):
    model = LinearQoiModel(slope=2.0)
    rom_builder = LinearQoiRomBuilderWithTrainingData(slope=2.0)
    variational_parameter_space = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )
    observations = np.array([0.5])
    observations_covariance = np.array([[0.1**2]])

    means, stds, parameter_samples, qois = romtools.workflows.run_mf_vi(
        model=model,
        rom_model_builder=rom_builder,
        prior_parameter_space=variational_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        absolute_vi_directory=str(tmp_path),
        fom_sample_size=6,
        rom_extra_sample_size=4,
        rom_tolerance=0.0,
        max_rom_training_history=1,
        optimizer_method="gradient",
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=2,
        ),
        line_search_method="legacy",
        line_search_config=romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-2,
            max_step_size=1e-2,
            step_size_growth_factor=1.0,
            relaxation_parameter=10.0,
            line_search_sample_growth_factor=1.0,
        ),
        bounded_parameter_handling="clip",
        random_seed=17,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        correlation_estimator="kfold",
        correlation_k_folds=3,
    )

    assert means.shape == (1,)
    assert stds.shape == (1,)
    assert parameter_samples.shape[1] == 1
    assert qois.shape[0] == 1
