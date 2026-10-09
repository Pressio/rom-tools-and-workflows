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
