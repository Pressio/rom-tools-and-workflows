"""Tests for fixed-window ROM-only gradient/Hessian MFVI substeps."""
import numpy as np
import pytest

from romtools.workflows.inverse import mf_vi_rom_substeps as substeps
from romtools.workflows.inverse.vi_optimization_methods import (
    VINewtonOptimizerConfig,
)


@pytest.mark.parametrize(
    "outer,start,end,count,enabled",
    [
        (0, 1, 3, 2, False),
        (1, 1, 3, 2, True),
        (2, 1, 3, 2, True),
        (3, 1, 3, 2, False),
        (100, 1, None, 2, True),
        (2, 0, None, 0, False),
    ],
)
def test_rom_substep_window(outer, start, end, count, enabled):
    assert substeps.rom_substeps_enabled(outer, start, end, count) == enabled


@pytest.mark.parametrize(
    "start,end,count", [(-1, None, 2), (0, 1, -1), (3, 2, 1), (0, -1, 0)]
)
def test_invalid_rom_substep_settings(start, end, count):
    with pytest.raises(ValueError):
        substeps.validate_rom_substeps(start, end, count)


def test_rom_substep_restart_round_trip():
    settings = substeps.rom_substep_restart_data(1, None, 3)
    assert substeps.restore_rom_substeps(settings, 0, 5, 0) == (1, None, 3)
    assert substeps.restore_rom_substeps({}, 2, 6, 4) == (2, 6, 4)


@pytest.mark.mpi_skip
def test_diagonal_substeps_recompute_gradient_and_hessian(monkeypatch):
    from romtools.workflows.inverse import vi_sample_reuse as reuse

    observed = []
    rom = object()

    def fake_evaluate(**kwargs):
        assert kwargs["model"] is rom
        assert "rom_substep_" in kwargs["run_directory_base"]
        current = kwargs["variational_mean"].copy()
        observed.append(current)
        return {
            "gradient_mean": np.array([1.0 + current[0]]),
            "gradient_log_std": np.array([0.5]),
            "hessian_diagonal_mean": np.array([-2.0]),
            "hessian_diagonal_log_std": np.array([-2.0]),
            "hessian_full": -2.0 * np.eye(2),
        }

    monkeypatch.setattr(reuse, "_ORIGINAL_EVALUATE_VI_STATE", fake_evaluate)
    mean, log_std = substeps.apply_diagonal_rom_substeps(
        rom_model=rom,
        candidate_mean=np.array([0.0]),
        candidate_log_std=np.array([0.0]),
        outer_iteration=1,
        num_rom_substeps=2,
        step_size=0.1,
        outer_method="newton",
        optimizer_config=VINewtonOptimizerConfig(
            newton_hessian_type="full", newton_regularization=0.01
        ),
        max_mean_update_std=None,
        max_log_std_update=0.5,
        min_variational_std=1e-6,
        max_variational_std=10.0,
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
        parameter_names=["theta"],
        prior_mean=np.zeros(1),
        prior_precision_operator=np.eye(1),
        prior_covariance_log_det=0.0,
        sample_size=10,
        evaluation_concurrency=1,
        covariance_regularization=1e-8,
        baseline_method="loo",
        bounded_parameter_handling="clip",
        parameter_mins=None,
        parameter_maxes=None,
        transform_interior_margin=0.0,
        transform_map="sigmoid",
        min_physical_variational_std_fraction=0.0,
        variational_correlation_cholesky=None,
        elbo_scaling_factor=1.0,
        log_likelihood_precision_operator=np.eye(1),
        sampling_method="mc",
        score_function_entropy_strategy="analytic",
        directory="work/iteration_2",
        dispatcher=object(),
    )
    assert len(observed) == 2
    assert not np.array_equal(observed[0], observed[1])
    assert mean[0] > 0.0
    assert log_std[0] > 0.0


@pytest.mark.mpi_skip
def test_full_covariance_substeps_use_fresh_hessian_and_keep_spd(monkeypatch):
    from romtools.workflows.inverse import full_covariance_vi_drivers as fcvi
    from romtools.workflows.inverse import full_covariance_newton as fcnewton

    current_means = []
    hessian_inputs = []
    rom = object()

    def fake_evaluate(**kwargs):
        assert kwargs["model"] is rom
        current_means.append(kwargs["variational_mean"].copy())
        return {
            "optimizer_samples": np.ones((8, 1)),
            "log_joint_terms": np.ones(8),
            "gradient_mean": np.array([1.0]),
            "gradient_covariance_svec": np.array([0.2]),
        }

    def fake_hessian(samples, mean, cholesky, terms, baseline, scaling):
        hessian_inputs.append(mean.copy())
        return -2.0 * np.eye(2)

    monkeypatch.setattr(fcvi, "_evaluate_single_fidelity_state", fake_evaluate)
    monkeypatch.setattr(fcnewton, "estimate_ordinary_hessian", fake_hessian)
    mean, chol = substeps.apply_full_covariance_rom_substeps(
        rom_model=rom,
        candidate_mean=np.array([0.0]),
        candidate_cholesky=np.eye(1),
        outer_iteration=0,
        num_rom_substeps=2,
        step_size=0.1,
        outer_method="newton",
        optimizer_config=VINewtonOptimizerConfig(
            newton_hessian_type="full", newton_metric="standard"
        ),
        max_covariance_log_step=0.5,
        min_variational_std=1e-8,
        max_variational_std=10.0,
        min_physical_variational_std_fraction=0.0,
        observations=np.zeros(1),
        observations_covariance=np.eye(1),
        parameter_names=["theta"],
        prior_mean=np.zeros(1),
        prior_precision_operator=np.eye(1),
        prior_covariance_log_det=0.0,
        sample_size=8,
        evaluation_concurrency=1,
        covariance_regularization=1e-8,
        baseline_method="loo",
        bounded_parameter_handling="clip",
        parameter_mins=None,
        parameter_maxes=None,
        transform_interior_margin=0.0,
        transform_map="sigmoid",
        elbo_scaling_factor=1.0,
        log_likelihood_precision_operator=np.eye(1),
        sampling_method="mc",
        score_function_entropy_strategy="analytic",
        directory="work/iteration_1",
        dispatcher=object(),
    )
    assert len(current_means) == len(hessian_inputs) == 2
    assert not np.array_equal(current_means[0], current_means[1])
    assert mean[0] > 0.0
    assert np.linalg.eigvalsh(chol @ chol.T).min() > 0.0


class _LinearModel:
    def populate_run_directory(self, run_directory, parameter_sample):
        return None

    def run_model(self, run_directory, parameter_sample):
        return 0

    def compute_qoi(self, run_directory, parameter_sample):
        return np.array([parameter_sample["theta"]])


class _Builder:
    def build_from_training_dirs(self, offline_data_dir, training_data_dirs,
                                 training_parameters, training_qois):
        return _LinearModel()


def _mfvi_kwargs(tmp_path):
    from romtools.workflows.parameter_spaces import GaussianParameterSpace, MonteCarloSampler
    from romtools.workflows.inverse.vi_optimization_methods import (
        VIGradientOptimizerConfig, VILegacyLineSearchConfig,
    )
    q0 = GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.zeros(1),
        stds=np.ones(1),
        sampler=MonteCarloSampler,
    )
    return dict(
        model=_LinearModel(),
        rom_model_builder=_Builder(),
        prior_parameter_space=q0,
        initial_variational_parameter_space=q0,
        observations=np.array([0.25]),
        observations_covariance=np.eye(1),
        parameter_mins=np.array([-5.0]),
        parameter_maxes=np.array([5.0]),
        bounded_parameter_handling="transform",
        absolute_work_dir=str(tmp_path),
        fom_sample_size=4,
        rom_extra_sample_size=4,
        rom_tolerance=np.inf,
        random_seed=67,
        optimizer_method="gradient",
        optimizer_config=VIGradientOptimizerConfig(
            max_iterations=3, gradient_norm_tolerance=0.0
        ),
        line_search_method="legacy",
        line_search_config=VILegacyLineSearchConfig(
            initial_step_size=0.01,
            max_step_size=0.01,
            step_size_growth_factor=1.0,
            relaxation_parameter=100.0,
        ),
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
    )


@pytest.mark.mpi_skip
def test_disabled_rom_substeps_preserve_mfvi_results(tmp_path):
    from romtools.workflows.inverse import run_mf_vi
    ordinary = run_mf_vi(**_mfvi_kwargs(tmp_path / "original"))
    disabled = run_mf_vi(
        **_mfvi_kwargs(tmp_path / "disabled"),
        rom_substep_start_iteration=1,
        rom_substep_end_iteration=2,
        num_rom_substeps=0,
    )
    for old, new in zip(ordinary, disabled):
        assert np.array_equal(old, new)


@pytest.mark.mpi_skip
def test_mfvi_substep_schedule_calls_only_requested_outer_iterations(
    tmp_path, monkeypatch
):
    from romtools.workflows.inverse import mf_vi_drivers
    seen = []

    def fake_substeps(**kwargs):
        seen.append(kwargs["outer_iteration"])
        return kwargs["candidate_mean"], kwargs["candidate_log_std"]

    monkeypatch.setattr(
        mf_vi_drivers, "apply_diagonal_rom_substeps", fake_substeps
    )
    mf_vi_drivers.run_mf_vi(
        **_mfvi_kwargs(tmp_path / "window"),
        rom_substep_start_iteration=1,
        rom_substep_end_iteration=2,
        num_rom_substeps=2,
    )
    assert seen == [1]

def _diagonal_inner_kwargs(method, config):
    """Arguments for fast mocked ROM-only unit tests."""
    return dict(
        rom_model=object(),
        candidate_mean=np.zeros(1),
        candidate_log_std=np.zeros(1),
        outer_iteration=0,
        num_rom_substeps=2,
        step_size=1.0,
        outer_method=method,
        optimizer_config=config,
        max_mean_update_std=None,
        max_log_std_update=0.5,
        min_variational_std=1e-8,
        max_variational_std=10.0,
        observations=np.zeros(1),
        observations_covariance=np.eye(1),
        parameter_names=["theta"],
        prior_mean=np.zeros(1),
        prior_precision_operator=np.eye(1),
        prior_covariance_log_det=0.0,
        sample_size=8,
        evaluation_concurrency=1,
        covariance_regularization=0.0,
        baseline_method="loo",
        bounded_parameter_handling="clip",
        parameter_mins=None,
        parameter_maxes=None,
        transform_interior_margin=0.0,
        transform_map="sigmoid",
        min_physical_variational_std_fraction=0.0,
        variational_correlation_cholesky=None,
        elbo_scaling_factor=1.0,
        log_likelihood_precision_operator=np.eye(1),
        sampling_method="mc",
        score_function_entropy_strategy="analytic",
        directory="work/iteration_1",
        dispatcher=object(),
    )


def _full_covariance_inner_kwargs(method, config):
    """Arguments for fast mocked full-covariance ROM-only unit tests."""
    args = _diagonal_inner_kwargs(method, config)
    args["candidate_cholesky"] = np.eye(1)
    args["max_covariance_log_step"] = 0.5
    del args["candidate_log_std"]
    del args["max_mean_update_std"]
    del args["max_log_std_update"]
    del args["variational_correlation_cholesky"]
    return args


@pytest.mark.mpi_skip
def test_diagonal_rom_gradient_inherits_natural_gradient(monkeypatch):
    from romtools.workflows.inverse import vi_sample_reuse as reuse
    from romtools.workflows.inverse.vi_optimization_methods import VIGradientOptimizerConfig

    seen_methods = []

    def evaluate(**kwargs):
        seen_methods.append(kwargs["gradient_method"])
        return {
            "gradient_mean": np.array([1.0]),
            "gradient_log_std": np.array([0.0]),
            # Contrasting ordinary vs. natural update: use latter.
            "update_direction_mean": np.array([4.0]),
            "update_direction_log_std": np.array([0.0]),
        }

    monkeypatch.setattr(reuse, "_ORIGINAL_EVALUATE_VI_STATE", evaluate)
    args = _diagonal_inner_kwargs(
        "gradient", VIGradientOptimizerConfig(gradient_method="natural")
    )
    args["num_rom_substeps"] = 1
    args["step_size"] = 0.1
    mean, _ = substeps.apply_diagonal_rom_substeps(**args)
    assert seen_methods == ["natural"]
    assert mean[0] == pytest.approx(0.4)


@pytest.mark.mpi_skip
def test_diagonal_rom_adam_uses_frozen_outer_moments(monkeypatch):
    from romtools.workflows.inverse import vi_sample_reuse as reuse
    from romtools.workflows.inverse.vi_optimization_methods import (
        AdamSolver, VIAdamOptimizerConfig,
    )

    config = VIAdamOptimizerConfig(
        gradient_method="standard", learning_rate=0.1, beta1=0.8, beta2=0.9
    )
    outer = AdamSolver.from_config(config)
    outer.step(np.array([2.0, -0.5]))
    outer.step(np.array([-0.7, 0.2]))
    outer_state = outer.restart_state_dict()
    gradients = [np.array([-2.0, 0.3]), np.array([1.0, -0.4])]
    seen = []

    def evaluate(**kwargs):
        seen.append(kwargs["variational_mean"].copy())
        assert kwargs["gradient_method"] == "standard"
        grad = gradients[(len(seen) - 1) % len(gradients)]
        return {"gradient_mean": grad[:1], "gradient_log_std": grad[1:]}

    monkeypatch.setattr(reuse, "_ORIGINAL_EVALUATE_VI_STATE", evaluate)
    args = _diagonal_inner_kwargs("adam", config)
    args["outer_adam_solver"] = outer
    args["step_size"] = 0.2

    # Each hypothetical step must be computed from the SAME original state,
    # not from an uninitialized solver or the previous ROM gradient's moments.
    expected_directions = []
    for gradient in gradients:
        shadow = AdamSolver.from_config(config)
        shadow.load_restart_state_dict(outer_state)
        expected_directions.append(shadow.step(gradient))
    expected = 0.2 * np.sum(expected_directions, axis=0)

    first_mean, first_log_std = substeps.apply_diagonal_rom_substeps(**args)
    second_mean, second_log_std = substeps.apply_diagonal_rom_substeps(**args)
    assert first_mean[0] == pytest.approx(expected[0])
    assert first_log_std[0] == pytest.approx(expected[1])
    np.testing.assert_allclose(second_mean, first_mean)
    np.testing.assert_allclose(second_log_std, first_log_std)
    assert len(seen) == 4
    assert not np.array_equal(seen[0], seen[1])
    assert np.array_equal(seen[0], seen[2])
    assert outer.iteration == outer_state["adam_iteration"]
    np.testing.assert_array_equal(
        outer.first_moment, outer_state["adam_first_moment"]
    )
    np.testing.assert_array_equal(
        outer.second_moment, outer_state["adam_second_moment"]
    )


@pytest.mark.mpi_skip
def test_rom_adam_requires_outer_optimizer_state():
    from romtools.workflows.inverse.vi_optimization_methods import VIAdamOptimizerConfig

    with pytest.raises(TypeError, match="outer AdamSolver"):
        substeps.apply_diagonal_rom_substeps(**_diagonal_inner_kwargs(
            "adam", VIAdamOptimizerConfig(gradient_method="standard")
        ))


@pytest.mark.mpi_skip
def test_full_covariance_rom_gradient_inherits_natural_gradient(monkeypatch):
    from romtools.workflows.inverse import full_covariance_vi_drivers as fcvi
    from romtools.workflows.inverse.vi_optimization_methods import VIGradientOptimizerConfig

    seen_methods = []

    def evaluate(**kwargs):
        seen_methods.append(kwargs["gradient_method"])
        return {
            "gradient_mean": np.array([1.0]),
            "gradient_covariance_svec": np.array([0.0]),
            "update_direction_mean": np.array([3.0]),
            "update_direction_covariance_svec": np.array([0.0]),
        }

    monkeypatch.setattr(fcvi, "_evaluate_single_fidelity_state", evaluate)
    args = _full_covariance_inner_kwargs(
        "gradient", VIGradientOptimizerConfig(gradient_method="natural")
    )
    args["num_rom_substeps"] = 1
    args["step_size"] = 0.1
    mean, chol = substeps.apply_full_covariance_rom_substeps(**args)
    assert seen_methods == ["natural"]
    assert mean[0] == pytest.approx(0.3)
    assert np.linalg.eigvalsh(chol @ chol.T).min() > 0.0


@pytest.mark.mpi_skip
def test_full_covariance_rom_adam_uses_frozen_outer_moments(monkeypatch):
    from romtools.workflows.inverse import full_covariance_vi_drivers as fcvi
    from romtools.workflows.inverse.vi_optimization_methods import (
        AdamSolver, VIAdamOptimizerConfig,
    )

    config = VIAdamOptimizerConfig(
        gradient_method="natural", learning_rate=0.1, beta1=0.8, beta2=0.9
    )
    outer = AdamSolver.from_config(config)
    outer.parameter_dimension = 1
    outer.step(np.array([2.0, 0.0]))
    outer.step(np.array([-0.5, 0.0]))
    outer_state = outer.restart_state_dict()
    gradients = [1.0, -2.0]
    observed = []

    def evaluate(**kwargs):
        assert kwargs["gradient_method"] == "natural"
        observed.append(kwargs["variational_mean"].copy())
        value = gradients[(len(observed) - 1) % len(gradients)]
        return {
            "gradient_mean": np.array([value]),
            "gradient_covariance_svec": np.zeros(1),
            "update_direction_mean": np.array([value]),
            "update_direction_covariance_svec": np.zeros(1),
        }

    monkeypatch.setattr(fcvi, "_evaluate_single_fidelity_state", evaluate)
    args = _full_covariance_inner_kwargs("adam", config)
    args["outer_adam_solver"] = outer
    args["step_size"] = 0.2

    expected = 0.0
    for value in gradients:
        shadow = AdamSolver.from_config(config)
        shadow.parameter_dimension = 1
        shadow.load_restart_state_dict(outer_state)
        expected += 0.2 * shadow.step(np.array([value, 0.0]))[0]

    a, chol_a = substeps.apply_full_covariance_rom_substeps(**args)
    b, chol_b = substeps.apply_full_covariance_rom_substeps(**args)
    assert a[0] == pytest.approx(expected)
    assert b[0] == pytest.approx(expected)
    assert len(observed) == 4
    assert not np.array_equal(observed[0], observed[1])
    assert np.array_equal(observed[0], observed[2])
    assert outer.iteration == outer_state["adam_iteration"]
    np.testing.assert_array_equal(
        outer.first_moment, outer_state["adam_first_moment"]
    )
    np.testing.assert_array_equal(
        outer.second_moment, outer_state["adam_second_moment"]
    )
    assert np.linalg.eigvalsh(chol_a @ chol_a.T).min() > 0.0
    assert np.linalg.eigvalsh(chol_b @ chol_b.T).min() > 0.0
