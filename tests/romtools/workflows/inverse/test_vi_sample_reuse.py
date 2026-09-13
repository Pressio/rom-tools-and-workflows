import numpy as np
import pytest

import romtools.workflows
from romtools.workflows.inverse import (
    VISampleReuseConfig,
    mf_vi_drivers,
    vi_drivers,
)
from romtools.workflows.inverse import vi_sample_reuse
from romtools.workflows.inverse import vi_sample_reuse_hessian
from romtools.workflows.inverse.vi_optimization_methods import (
    VIAdamOptimizerConfig,
    VINewtonOptimizerConfig,
    VILegacyLineSearchConfig,
)
from romtools.workflows.parameter_spaces import GaussianParameterSpace, MonteCarloSampler


class CountingLinearQoiModel:
    def __init__(self, slope=1.0):
        self._slope = float(slope)
        self.run_model_calls = 0

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        return

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        self.run_model_calls += 1
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        return np.array([self._slope * float(parameter_sample["theta"])])


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


def _parameter_space():
    return GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )


def _reuse_config():
    return VISampleReuseConfig(
        history_batches=4,
        ess_threshold=1.0e-12,
        use_score_diagnostic=False,
        periodic_refresh=None,
    )


def _newton_config(curvature_strategy="same_sample", hessian_samples=None):
    return VINewtonOptimizerConfig(
        gradient_norm_tolerance=0.0,
        max_iterations=2,
        newton_regularization=0.1,
        newton_hessian_type="full",
        newton_curvature_strategy=curvature_strategy,
        newton_hessian_num_samples=hessian_samples,
    )


def _small_newton_line_search():
    return VILegacyLineSearchConfig(
        initial_step_size=1.0e-3,
        max_step_size=1.0e-3,
        step_size_growth_factor=1.0,
        step_size_decay_factor=2.0,
        max_step_size_decrease_trys=2,
        relaxation_parameter=3.0,
    )


def test_deterministic_mixture_weights_are_one_for_current_proposal():
    config = _reuse_config()
    archive = vi_sample_reuse._ReuseArchive(config)
    samples = np.array([[-1.0], [0.0], [1.0]])
    batch = vi_sample_reuse._ReuseBatch(
        optimizer_samples=samples,
        parameter_samples=samples.copy(),
        qois=samples.T.copy(),
        errors=samples.T.copy(),
        variational_mean=np.array([0.0]),
        variational_log_std=np.array([0.0]),
        variational_correlation_cholesky=None,
        iteration=0,
    )
    archive.append(batch)

    weights, origin_weights = vi_sample_reuse._compute_importance_weights(
        archive,
        np.array([0.0]),
        np.array([0.0]),
        None,
    )

    np.testing.assert_allclose(weights, np.ones(3))
    np.testing.assert_allclose(origin_weights, np.ones(3))
    assert np.isclose(vi_sample_reuse._effective_sample_size(weights), 3.0)


def test_weighted_loo_reduces_to_standard_loo_for_unit_weights():
    values = np.array([1.0, 2.0, 4.0, 8.0])
    expected = vi_drivers._compute_leave_one_out_baseline(values)
    actual = vi_sample_reuse._weighted_loo_baseline(values, np.ones(values.size))
    np.testing.assert_allclose(actual, expected)


def test_reused_hessian_reduces_to_standard_hessian_for_unit_weights():
    samples = np.array([[-1.5], [-0.25], [0.5], [1.75]])
    mean = np.array([0.1])
    log_std = np.log(np.array([0.9]))
    values = np.array([-1.2, -0.4, -0.8, -2.0])
    expected = vi_drivers._compute_reinforce_hessian_full(
        samples,
        mean,
        np.exp(log_std),
        values,
        "loo",
        None,
    )
    actual = vi_sample_reuse_hessian._hessian_from_archive(
        samples,
        values,
        mean,
        log_std,
        None,
        np.ones(samples.shape[0]),
        np.ones(samples.shape[0]),
        "loo",
    )
    np.testing.assert_allclose(actual, expected)


def test_reused_mf_hessian_reduces_to_standard_mf_hessian_for_unit_weights():
    samples_fom = np.array([[-1.25], [0.1], [1.4]])
    samples_extra = np.array([[-0.8], [0.4], [1.0], [1.8]])
    mean = np.array([0.0])
    log_std = np.log(np.array([1.1]))
    fom_values = np.array([-1.4, -0.5, -1.8])
    rom_base_values = np.array([-1.2, -0.45, -1.6])
    rom_extra_values = np.array([-0.9, -0.55, -1.1, -2.0])
    expected = mf_vi_drivers._compute_mfmc_reinforce_hessian_full(
        samples_fom,
        samples_fom,
        samples_extra,
        mean,
        np.exp(log_std),
        fom_values,
        rom_base_values,
        rom_extra_values,
        "loo",
        True,
        "componentwise",
        None,
        1.0,
    )
    actual = vi_sample_reuse_hessian._mf_hessian_from_reuse(
        samples_fom,
        samples_extra,
        mean,
        log_std,
        None,
        fom_values,
        rom_base_values,
        rom_extra_values,
        np.ones(samples_fom.shape[0]),
        np.ones(samples_fom.shape[0]),
        "loo",
        1.0,
        True,
        "componentwise",
    )
    np.testing.assert_allclose(actual, expected)


@pytest.mark.mpi_skip
def test_run_vi_reuses_high_fidelity_model_evaluations(tmp_path):
    model = CountingLinearQoiModel()
    result = vi_drivers.run_vi(
        model=model,
        prior_parameter_space=_parameter_space(),
        observations=np.array([0.5]),
        observations_covariance=np.array([[0.25]]),
        absolute_work_dir=str(tmp_path / "vi_reuse"),
        sample_size=6,
        optimizer_method="adam",
        optimizer_config=VIAdamOptimizerConfig(
            learning_rate=0.01,
            gradient_norm_tolerance=0.0,
            max_iterations=3,
        ),
        bounded_parameter_handling="clip",
        random_seed=3,
        evaluation_concurrency=1,
        sample_reuse_config=_reuse_config(),
    )

    assert model.run_model_calls == 6
    assert np.all(np.isfinite(result[0]))
    assert np.all(np.isfinite(result[1]))


@pytest.mark.mpi_skip
def test_run_mf_vi_reuses_fom_and_keeps_fresh_rom_enrichment(tmp_path):
    fom = CountingLinearQoiModel()
    result = mf_vi_drivers.run_mf_vi(
        model=fom,
        rom_model_builder=LinearQoiRomBuilderWithTrainingData(),
        prior_parameter_space=_parameter_space(),
        observations=np.array([0.5]),
        observations_covariance=np.array([[0.25]]),
        absolute_work_dir=str(tmp_path / "mf_vi_reuse"),
        fom_sample_size=5,
        rom_extra_sample_size=7,
        rom_tolerance=1.0,
        optimizer_method="adam",
        optimizer_config=VIAdamOptimizerConfig(
            learning_rate=0.01,
            gradient_norm_tolerance=0.0,
            max_iterations=3,
        ),
        bounded_parameter_handling="clip",
        random_seed=3,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        sample_reuse_config=_reuse_config(),
    )

    assert fom.run_model_calls == 5
    assert np.all(np.isfinite(result[0]))
    assert np.all(np.isfinite(result[1]))


@pytest.mark.mpi_skip
def test_run_vi_newton_reuses_hessian_fom_evaluations(tmp_path):
    model = CountingLinearQoiModel()
    result = vi_drivers.run_vi(
        model=model,
        prior_parameter_space=_parameter_space(),
        observations=np.array([0.5]),
        observations_covariance=np.array([[0.25]]),
        absolute_work_dir=str(tmp_path / "vi_newton_reuse"),
        sample_size=6,
        optimizer_method="newton",
        optimizer_config=_newton_config(),
        line_search_method="legacy",
        line_search_config=_small_newton_line_search(),
        bounded_parameter_handling="clip",
        random_seed=4,
        evaluation_concurrency=1,
        sample_reuse_config=_reuse_config(),
    )

    assert model.run_model_calls == 6
    assert np.all(np.isfinite(result[0]))
    assert np.all(np.isfinite(result[1]))


@pytest.mark.mpi_skip
def test_run_vi_newton_independent_curvature_uses_separate_reuse_archive(tmp_path):
    model = CountingLinearQoiModel()
    result = vi_drivers.run_vi(
        model=model,
        prior_parameter_space=_parameter_space(),
        observations=np.array([0.5]),
        observations_covariance=np.array([[0.25]]),
        absolute_work_dir=str(tmp_path / "vi_newton_independent_reuse"),
        sample_size=6,
        optimizer_method="newton",
        optimizer_config=_newton_config(
            curvature_strategy="independent", hessian_samples=4
        ),
        line_search_method="legacy",
        line_search_config=_small_newton_line_search(),
        bounded_parameter_handling="clip",
        random_seed=5,
        evaluation_concurrency=1,
        sample_reuse_config=_reuse_config(),
    )

    # Six gradient samples plus four samples seeding the independent Hessian archive.
    assert model.run_model_calls == 10
    assert np.all(np.isfinite(result[0]))
    assert np.all(np.isfinite(result[1]))


@pytest.mark.mpi_skip
def test_run_mf_vi_newton_reuses_hessian_fom_evaluations(tmp_path):
    fom = CountingLinearQoiModel()
    result = mf_vi_drivers.run_mf_vi(
        model=fom,
        rom_model_builder=LinearQoiRomBuilderWithTrainingData(),
        prior_parameter_space=_parameter_space(),
        observations=np.array([0.5]),
        observations_covariance=np.array([[0.25]]),
        absolute_work_dir=str(tmp_path / "mf_vi_newton_reuse"),
        fom_sample_size=5,
        rom_extra_sample_size=7,
        rom_tolerance=1.0,
        optimizer_method="newton",
        optimizer_config=_newton_config(),
        line_search_method="legacy",
        line_search_config=_small_newton_line_search(),
        bounded_parameter_handling="clip",
        random_seed=6,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        sample_reuse_config=_reuse_config(),
    )

    assert fom.run_model_calls == 5
    assert np.all(np.isfinite(result[0]))
    assert np.all(np.isfinite(result[1]))


def test_sample_reuse_rejects_rqmc(tmp_path):
    common = dict(
        model=LinearQoiModel(),
        prior_parameter_space=_parameter_space(),
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
        absolute_work_dir=str(tmp_path / "unsupported"),
        bounded_parameter_handling="clip",
        sample_reuse_config=_reuse_config(),
    )
    with pytest.raises(NotImplementedError, match="sampling_method='mc'"):
        vi_drivers.run_vi(sampling_method="rqmc", **common)
