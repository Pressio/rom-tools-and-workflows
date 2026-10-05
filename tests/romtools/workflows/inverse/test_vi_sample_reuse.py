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
        use_hessian_score_diagnostic=False,
        hessian_relative_standard_error_threshold=None,
    )


def _strict_hessian_variance_config():
    return VISampleReuseConfig(
        history_batches=4,
        ess_threshold=1.0e-12,
        use_score_diagnostic=False,
        periodic_refresh=None,
        use_hessian_score_diagnostic=False,
        hessian_relative_standard_error_threshold=1.0e-12,
    )


def _always_refresh_config():
    return VISampleReuseConfig(
        history_batches=4,
        ess_threshold=1.0e12,
        use_score_diagnostic=False,
        periodic_refresh=None,
        use_hessian_score_diagnostic=False,
        hessian_relative_standard_error_threshold=None,
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


def _reuse_batch(samples, iteration=0):
    samples = np.asarray(samples, dtype=float).reshape(-1, 1)
    return vi_sample_reuse._ReuseBatch(
        optimizer_samples=samples,
        parameter_samples=samples.copy(),
        qois=samples.T.copy(),
        errors=samples.T.copy(),
        variational_mean=np.array([0.0]),
        variational_log_std=np.array([0.0]),
        variational_correlation_cholesky=None,
        iteration=iteration,
    )


def test_newton_hessian_reuse_safeguards_are_enabled_by_default():
    config = VISampleReuseConfig()
    assert config.use_hessian_score_diagnostic
    assert np.isclose(config.hessian_score_error_scale, 2.0)
    assert np.isclose(config.hessian_relative_standard_error_threshold, 0.25)


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


def test_importance_log_ratios_are_clipped_to_fifty():
    config = _reuse_config()
    archive = vi_sample_reuse._ReuseArchive(config)
    archive.append(_reuse_batch([20.0]))

    weights, origin_weights = vi_sample_reuse._compute_importance_weights(
        archive,
        np.array([20.0]),
        np.array([0.0]),
        None,
    )

    # The helper batch was generated by N(0, 1), while the current proposal is
    # N(20, 1); both ratios would otherwise have log value 200.
    np.testing.assert_allclose(weights, np.exp(50.0))
    np.testing.assert_allclose(origin_weights, np.exp(50.0))


def test_score_diagnostic_uses_inverse_fisher_metric(monkeypatch):
    archive = vi_sample_reuse._ReuseArchive(_reuse_config())
    samples = np.array([[1.0, -2.0], [3.0, 1.0]])
    archive.append(vi_sample_reuse._ReuseBatch(
        optimizer_samples=samples,
        parameter_samples=samples.copy(),
        qois=np.zeros((1, 2)),
        errors=np.zeros((1, 2)),
        variational_mean=np.zeros(2),
        variational_log_std=np.zeros(2),
        variational_correlation_cholesky=None,
        iteration=0,
    ))
    mean = np.array([0.5, -0.25])
    log_std = np.log(np.array([2.0, 0.5]))
    reference_samples = np.array([[0.0, 0.25], [2.0, -1.0]])
    monkeypatch.setattr(
        vi_sample_reuse,
        "_draw_optimizer_only",
        lambda *_args, **_kwargs: reference_samples,
    )

    actual_recycled, actual_reference = vi_sample_reuse._score_diagnostics(
        archive, mean, log_std, None, np.ones(2), 2
    )

    archive_scores = np.hstack(vi_sample_reuse._score_functions(
        archive.batches[0].optimizer_samples, mean, log_std, None
    ))
    reference_scores = np.hstack(vi_sample_reuse._score_functions(
        reference_samples, mean, log_std, None
    ))
    metric = np.array([4.0, 0.25, 0.5, 0.5])
    expected_recycled = np.sqrt(np.sum(metric * np.mean(archive_scores, axis=0) ** 2))
    expected_reference = np.sqrt(np.sum(metric * np.mean(reference_scores, axis=0) ** 2))
    assert np.isclose(actual_recycled, expected_recycled)
    assert np.isclose(actual_reference, expected_reference)


def test_ess_threshold_is_inclusive():
    config = VISampleReuseConfig(
        ess_threshold=3.0,
        use_score_diagnostic=False,
        periodic_refresh=None,
        use_hessian_score_diagnostic=False,
        hessian_relative_standard_error_threshold=None,
    )
    archive = vi_sample_reuse._ReuseArchive(config)
    archive.append(_reuse_batch([-1.0, 0.0, 1.0]))

    refresh, reasons, _, ess = vi_sample_reuse._refresh_decision(
        archive, np.array([0.0]), np.array([0.0]), None, 3
    )

    assert refresh
    assert reasons == ("ess",)
    assert np.isclose(ess, 3.0)


def test_dynamic_archive_limit_and_absolute_periodic_refresh():
    config = VISampleReuseConfig(
        history_batches=4,
        periodic_refresh=2,
        use_score_diagnostic=False,
        use_hessian_score_diagnostic=False,
        hessian_relative_standard_error_threshold=None,
    )
    assert vi_sample_reuse._iteration_archive_limit(config, 0) == 1
    assert vi_sample_reuse._iteration_archive_limit(config, 2) == 3
    assert vi_sample_reuse._iteration_archive_limit(config, 10) == 4

    controller = vi_sample_reuse._VIReuseController(config)
    assert not controller._periodic_due(0)
    assert controller._periodic_due(2)
    controller._periodic_refresh_iterations.add(2)
    assert not controller._periodic_due(2)
    assert not controller._periodic_due(3)
    assert controller._periodic_due(4)


def test_archive_append_respects_dynamic_capacity():
    archive = vi_sample_reuse._ReuseArchive(_reuse_config())
    archive.append(_reuse_batch([0.0], iteration=0), capacity=1)
    archive.append(_reuse_batch([1.0], iteration=1), capacity=2)
    archive.append(_reuse_batch([2.0], iteration=1), capacity=2)
    assert [batch.iteration for batch in archive.batches] == [1, 1]
    np.testing.assert_allclose(
        np.vstack([batch.optimizer_samples for batch in archive.batches]),
        np.array([[1.0], [2.0]]),
    )


def test_weighted_loo_reduces_to_standard_loo_for_unit_weights():
    values = np.array([1.0, 2.0, 4.0, 8.0])
    expected = vi_drivers._compute_leave_one_out_baseline(values)
    actual = vi_sample_reuse._weighted_loo_baseline(values, np.ones(values.size))
    np.testing.assert_allclose(actual, expected)


def test_archive_gradient_matches_abris_deterministic_mixture_equations():
    archive = vi_sample_reuse._ReuseArchive(_reuse_config())
    first = _reuse_batch([-1.0, 0.0], iteration=0)
    first.variational_mean[:] = -1.0
    second = _reuse_batch([1.0, 2.0, 3.0], iteration=1)
    second.variational_mean[:] = 2.0
    second.variational_log_std[:] = np.log(0.5)
    archive.append(first)
    archive.append(second)
    mean = np.array([0.5])
    std = 1.2
    log_std = np.log(np.array([std]))

    weights, origin_weights = vi_sample_reuse._compute_importance_weights(
        archive, mean, log_std, None
    )
    samples = np.vstack([first.optimizer_samples, second.optimizer_samples])[:, 0]

    def normal_pdf(x, location, scale):
        return np.exp(-0.5 * ((x - location) / scale) ** 2) / (
            np.sqrt(2.0 * np.pi) * scale
        )

    current_density = normal_pdf(samples, 0.5, std)
    mixture_density = (
        (2.0 / 5.0) * normal_pdf(samples, -1.0, 1.0)
        + (3.0 / 5.0) * normal_pdf(samples, 2.0, 0.5)
    )
    expected_weights = current_density / mixture_density
    np.testing.assert_allclose(weights, expected_weights)

    values = np.array([-2.0, -0.5, 0.25, 1.0, 1.5])
    gradient = vi_sample_reuse._gradient_from_archive(
        samples[:, None],
        values,
        mean,
        log_std,
        None,
        weights,
        origin_weights,
        "none",
        1.0,
        "analytic",
    )
    score_mean = (samples - mean[0]) / std**2
    score_log_std = ((samples - mean[0]) / std) ** 2 - 1.0
    np.testing.assert_allclose(gradient[0], np.mean(weights * values * score_mean))
    np.testing.assert_allclose(
        gradient[1], np.mean(weights * values * score_log_std) + 1.0
    )


def test_batch_aware_hessian_standard_error_uses_archive_strata():
    archive = vi_sample_reuse._ReuseArchive(_reuse_config())
    archive.append(_reuse_batch([-1.0, 1.0], iteration=0))
    archive.append(_reuse_batch([-2.0, 2.0], iteration=1))
    sample_terms = np.array([1.0, 3.0, 2.0, 6.0]).reshape(-1, 1, 1)

    actual = vi_sample_reuse_hessian._batch_aware_hessian_standard_error(
        sample_terms, archive
    )
    expected = np.sqrt(1.25)
    assert np.isclose(actual, expected)


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
    prior_parameter_space = _parameter_space()
    result = vi_drivers.run_vi(
        model=model,
        prior_parameter_space=prior_parameter_space,
        initial_variational_parameter_space=prior_parameter_space,
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
def test_run_vi_refresh_loop_uses_dynamic_budget_and_archive_update(
    tmp_path, monkeypatch
):
    model = CountingLinearQoiModel()
    prior_parameter_space = _parameter_space()
    observed_states = []
    original_annotate = vi_sample_reuse._annotate_reuse_state

    def record_state(*args, **kwargs):
        state = original_annotate(*args, **kwargs)
        observed_states.append(state.copy())
        return state

    monkeypatch.setattr(vi_sample_reuse, "_annotate_reuse_state", record_state)
    vi_drivers.run_vi(
        model=model,
        prior_parameter_space=prior_parameter_space,
        initial_variational_parameter_space=prior_parameter_space,
        observations=np.array([0.5]),
        observations_covariance=np.array([[0.25]]),
        absolute_work_dir=str(tmp_path / "vi_adaptive_refresh"),
        sample_size=3,
        optimizer_method="adam",
        optimizer_config=VIAdamOptimizerConfig(
            learning_rate=0.01,
            gradient_norm_tolerance=0.0,
            max_iterations=3,
        ),
        bounded_parameter_handling="clip",
        random_seed=9,
        evaluation_concurrency=1,
        sample_reuse_config=_always_refresh_config(),
    )

    # Iterations 0, 1, and 2 permit one, two, and three fresh batches.
    assert model.run_model_calls == 3 * (1 + 2 + 3)
    assert [state["sample_reuse_refresh_count"] for state in observed_states] == [1, 2, 3]
    assert [state["sample_reuse_archive_batches"] for state in observed_states] == [1, 2, 3]
    assert all(state["sample_reuse_used"] for state in observed_states)
    assert all(state["sample_reuse_final_update_used_archive"] for state in observed_states)
    assert all(state["sample_reuse_refresh_limit_reached"] for state in observed_states)
    assert observed_states[-1]["optimizer_samples"].shape[0] == 9

    refresh_directories = sorted(
        path.name
        for path in (tmp_path / "vi_adaptive_refresh" / "iteration_2").iterdir()
        if "abris_refresh" in path.name
    )
    assert refresh_directories


@pytest.mark.mpi_skip
def test_run_mf_vi_reuses_fom_and_keeps_fresh_rom_enrichment(tmp_path):
    fom = CountingLinearQoiModel()
    prior_parameter_space = _parameter_space()
    result = mf_vi_drivers.run_mf_vi(
        model=fom,
        rom_model_builder=LinearQoiRomBuilderWithTrainingData(),
        prior_parameter_space=prior_parameter_space,
        initial_variational_parameter_space=prior_parameter_space,
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
def test_run_mf_vi_refresh_loop_carries_training_state(tmp_path, monkeypatch):
    fom = CountingLinearQoiModel()
    prior_parameter_space = _parameter_space()
    observed_states = []
    original_annotate = vi_sample_reuse._annotate_reuse_state

    def record_state(*args, **kwargs):
        state = original_annotate(*args, **kwargs)
        observed_states.append(state.copy())
        return state

    monkeypatch.setattr(vi_sample_reuse, "_annotate_reuse_state", record_state)
    mf_vi_drivers.run_mf_vi(
        model=fom,
        rom_model_builder=LinearQoiRomBuilderWithTrainingData(),
        prior_parameter_space=prior_parameter_space,
        initial_variational_parameter_space=prior_parameter_space,
        observations=np.array([0.5]),
        observations_covariance=np.array([[0.25]]),
        absolute_work_dir=str(tmp_path / "mf_vi_adaptive_refresh"),
        fom_sample_size=2,
        rom_extra_sample_size=3,
        rom_tolerance=1.0,
        optimizer_method="adam",
        optimizer_config=VIAdamOptimizerConfig(
            learning_rate=0.01,
            gradient_norm_tolerance=0.0,
            max_iterations=3,
        ),
        bounded_parameter_handling="clip",
        random_seed=10,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        sample_reuse_config=_always_refresh_config(),
    )

    assert fom.run_model_calls == 2 * (1 + 2 + 3)
    assert [state["sample_reuse_refresh_count"] for state in observed_states] == [1, 2, 3]
    assert observed_states[-1]["sample_reuse_archive_batches"] == 3
    assert observed_states[-1]["parameter_samples_fom"].shape[0] == 6
    # Each repeated refresh receives the training state returned by the prior
    # refresh, so all six FOM batches remain represented in driver state.
    assert len(observed_states[-1]["training_dirs"]) == 12
    assert observed_states[-1]["parameter_samples_rom_only"].shape[0] == 3


@pytest.mark.mpi_skip
def test_run_vi_newton_reuses_hessian_fom_evaluations(tmp_path):
    model = CountingLinearQoiModel()
    prior_parameter_space = _parameter_space()
    result = vi_drivers.run_vi(
        model=model,
        prior_parameter_space=prior_parameter_space,
        initial_variational_parameter_space=prior_parameter_space,
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
def test_run_vi_newton_hessian_variance_can_force_refresh(tmp_path, monkeypatch):
    model = CountingLinearQoiModel()
    prior_parameter_space = _parameter_space()
    observed_states = []
    original_annotate = vi_sample_reuse._annotate_reuse_state

    def record_state(*args, **kwargs):
        state = original_annotate(*args, **kwargs)
        observed_states.append(state.copy())
        return state

    monkeypatch.setattr(vi_sample_reuse, "_annotate_reuse_state", record_state)
    result = vi_drivers.run_vi(
        model=model,
        prior_parameter_space=prior_parameter_space,
        initial_variational_parameter_space=prior_parameter_space,
        observations=np.array([0.5]),
        observations_covariance=np.array([[0.25]]),
        absolute_work_dir=str(tmp_path / "vi_newton_variance_refresh"),
        sample_size=6,
        optimizer_method="newton",
        optimizer_config=_newton_config(),
        line_search_method="legacy",
        line_search_config=_small_newton_line_search(),
        bounded_parameter_handling="clip",
        random_seed=7,
        evaluation_concurrency=1,
        sample_reuse_config=_strict_hessian_variance_config(),
    )

    assert model.run_model_calls == 6 * (1 + 2)
    assert [state["sample_reuse_refresh_count"] for state in observed_states] == [1, 2]
    assert all(count <= limit for count, limit in zip(
        [state["sample_reuse_refresh_count"] for state in observed_states],
        [1, 2],
    ))
    assert all(state["sample_reuse_refresh_limit_reached"] for state in observed_states)
    assert "hessian_variance" in observed_states[-1]["sample_reuse_refresh_reasons"]
    assert np.all(np.isfinite(result[0]))
    assert np.all(np.isfinite(result[1]))


@pytest.mark.mpi_skip
def test_run_vi_newton_independent_curvature_uses_separate_reuse_archive(tmp_path):
    model = CountingLinearQoiModel()
    prior_parameter_space = _parameter_space()
    result = vi_drivers.run_vi(
        model=model,
        prior_parameter_space=prior_parameter_space,
        initial_variational_parameter_space=prior_parameter_space,
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
    prior_parameter_space = _parameter_space()
    result = mf_vi_drivers.run_mf_vi(
        model=fom,
        rom_model_builder=LinearQoiRomBuilderWithTrainingData(),
        prior_parameter_space=prior_parameter_space,
        initial_variational_parameter_space=prior_parameter_space,
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


@pytest.mark.mpi_skip
def test_run_mf_vi_newton_hessian_variance_can_force_refresh(tmp_path, monkeypatch):
    fom = CountingLinearQoiModel()
    prior_parameter_space = _parameter_space()
    observed_states = []
    original_annotate = vi_sample_reuse._annotate_reuse_state

    def record_state(*args, **kwargs):
        state = original_annotate(*args, **kwargs)
        observed_states.append(state.copy())
        return state

    monkeypatch.setattr(vi_sample_reuse, "_annotate_reuse_state", record_state)
    result = mf_vi_drivers.run_mf_vi(
        model=fom,
        rom_model_builder=LinearQoiRomBuilderWithTrainingData(),
        prior_parameter_space=prior_parameter_space,
        initial_variational_parameter_space=prior_parameter_space,
        observations=np.array([0.5]),
        observations_covariance=np.array([[0.25]]),
        absolute_work_dir=str(tmp_path / "mf_vi_newton_variance_refresh"),
        fom_sample_size=5,
        rom_extra_sample_size=7,
        rom_tolerance=1.0,
        optimizer_method="newton",
        optimizer_config=_newton_config(),
        line_search_method="legacy",
        line_search_config=_small_newton_line_search(),
        bounded_parameter_handling="clip",
        random_seed=8,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        sample_reuse_config=_strict_hessian_variance_config(),
    )

    assert fom.run_model_calls == 5 * (1 + 2)
    assert [state["sample_reuse_refresh_count"] for state in observed_states] == [1, 2]
    assert all(state["sample_reuse_refresh_limit_reached"] for state in observed_states)
    assert "hessian_variance" in observed_states[-1]["sample_reuse_refresh_reasons"]
    assert np.all(np.isfinite(result[0]))
    assert np.all(np.isfinite(result[1]))


def test_sample_reuse_rejects_rqmc(tmp_path):
    parameter_space = _parameter_space()
    common = dict(
        model=LinearQoiModel(),
        prior_parameter_space=parameter_space,
        initial_variational_parameter_space=parameter_space,
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
        absolute_work_dir=str(tmp_path / "unsupported"),
        bounded_parameter_handling="clip",
        sample_reuse_config=_reuse_config(),
    )
    with pytest.raises(NotImplementedError, match="sampling_method='mc'"):
        vi_drivers.run_vi(sampling_method="rqmc", **common)
