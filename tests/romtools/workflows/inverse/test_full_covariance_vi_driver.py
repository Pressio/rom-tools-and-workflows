import numpy as np
import pytest

import romtools.workflows
from romtools.workflows.inverse.full_covariance_mf_vi_drivers import _upgrade_mf_state
from romtools.workflows.inverse.full_covariance_vi import (
    covariance_from_cholesky,
    natural_score_terms,
    svec,
)
from romtools.workflows.parameter_spaces import (
    GaussianParameterSpace,
    MultivariateGaussianParameterSpace,
    MonteCarloSampler,
)


class IdentityTwoParameterModel:
    def populate_run_directory(self, run_directory, parameter_sample):
        return

    def run_model(self, run_directory, parameter_sample):
        return 0

    def compute_qoi(self, run_directory, parameter_sample):
        return np.array([
            float(parameter_sample["theta0"]),
            float(parameter_sample["theta1"]),
        ])


def _diagonal_prior():
    return GaussianParameterSpace(
        parameter_names=["theta0", "theta1"],
        means=np.array([0.0, 0.0]),
        stds=np.array([1.0, 1.0]),
        sampler=MonteCarloSampler,
    )


def _correlated_prior():
    return MultivariateGaussianParameterSpace(
        parameter_names=["theta0", "theta1"],
        means=np.array([0.0, 0.0]),
        covariance=np.array([[1.0, 0.35], [0.35, 0.8]]),
        sampler=MonteCarloSampler,
    )


@pytest.mark.mpi_skip
def test_full_covariance_vi_is_explicit_and_legacy_multivariate_remains_available(tmp_path):
    # Omitting variational_distribution preserves the historical multivariate
    # path, including its current Newton support.
    means, stds, _, _ = romtools.workflows.run_vi(
        model=IdentityTwoParameterModel(),
        prior_parameter_space=_correlated_prior(),
        observations=np.zeros(2),
        observations_covariance=np.eye(2),
        absolute_work_dir=str(tmp_path / "legacy"),
        sample_size=6,
        optimizer_method="newton",
        optimizer_config=romtools.workflows.VINewtonOptimizerConfig(
            gradient_norm_tolerance=0.0,
            max_iterations=1,
        ),
        bounded_parameter_handling="clip",
        evaluation_concurrency=1,
    )
    assert means.shape == (2,)
    assert stds.shape == (2,)

    # The new family is selected explicitly and intentionally excludes Newton.
    with pytest.raises(NotImplementedError, match="Full-covariance Newton"):
        romtools.workflows.run_vi(
            model=IdentityTwoParameterModel(),
            prior_parameter_space=_correlated_prior(),
            observations=np.zeros(2),
            observations_covariance=np.eye(2),
            absolute_work_dir=str(tmp_path / "full_newton"),
            sample_size=6,
            optimizer_method="newton",
            optimizer_config=romtools.workflows.VINewtonOptimizerConfig(
                max_iterations=1,
            ),
            variational_distribution="full_covariance",
            bounded_parameter_handling="clip",
            evaluation_concurrency=1,
        )


@pytest.mark.mpi_skip
def test_full_covariance_natural_adam_rotates_initially_diagonal_covariance(tmp_path):
    work_dir = tmp_path / "full_adam"
    means, stds, parameter_samples, qois = romtools.workflows.run_vi(
        model=IdentityTwoParameterModel(),
        prior_parameter_space=_diagonal_prior(),
        observations=np.array([0.4, -0.3]),
        observations_covariance=np.array([[0.15, 0.11], [0.11, 0.18]]),
        absolute_work_dir=str(work_dir),
        sample_size=128,
        optimizer_method="adam",
        optimizer_config=romtools.workflows.VIAdamOptimizerConfig(
            gradient_method="natural",
            learning_rate=0.03,
            gradient_norm_tolerance=0.0,
            max_iterations=3,
        ),
        variational_distribution="full_covariance",
        bounded_parameter_handling="clip",
        baseline_method="loo",
        random_seed=398,
        evaluation_concurrency=1,
        create_run_directories=False,
    )

    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(stds))
    assert parameter_samples.shape[1] == 2
    assert qois.shape[0] == 2

    with np.load(work_dir / "history.npz", allow_pickle=True) as history:
        covariance_history = history["vi_history_variational_covariance"]
        assert covariance_history.shape[1:] == (2, 2)
        assert np.isclose(covariance_history[0, 0, 1], 0.0)
        assert abs(covariance_history[-1, 0, 1]) > 1e-5
        for covariance in covariance_history:
            assert np.all(np.linalg.eigvalsh(covariance) > 0.0)

    restart_paths = sorted(work_dir.glob("iteration_*/restart.npz"))
    assert restart_paths
    with np.load(restart_paths[-1], allow_pickle=True) as restart:
        assert str(restart["variational_distribution"].item()) == "full_covariance"
        cholesky = restart["variational_cholesky_optimizer"]
        assert np.allclose(cholesky, np.tril(cholesky))
        assert np.all(np.diag(cholesky) > 0.0)
        covariance = covariance_from_cholesky(cholesky)
        assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


@pytest.mark.mpi_skip
def test_full_covariance_restart_continues_with_complete_cholesky_state(tmp_path):
    work_dir = tmp_path / "restart"
    common = dict(
        model=IdentityTwoParameterModel(),
        prior_parameter_space=_correlated_prior(),
        observations=np.array([0.2, -0.1]),
        observations_covariance=np.array([[0.2, 0.07], [0.07, 0.25]]),
        absolute_work_dir=str(work_dir),
        sample_size=32,
        optimizer_method="adam",
        variational_distribution="full_covariance",
        bounded_parameter_handling="clip",
        random_seed=19,
        evaluation_concurrency=1,
        create_run_directories=False,
    )
    romtools.workflows.run_vi(
        optimizer_config=romtools.workflows.VIAdamOptimizerConfig(
            learning_rate=0.02,
            gradient_norm_tolerance=0.0,
            max_iterations=2,
        ),
        **common,
    )
    restart_file = work_dir / "iteration_1" / "restart.npz"
    assert restart_file.exists()
    with np.load(restart_file, allow_pickle=True) as restart:
        first_cholesky = restart["variational_cholesky_optimizer"].copy()
        assert first_cholesky.shape == (2, 2)

    result = romtools.workflows.run_vi(
        restart_file=str(restart_file),
        optimizer_config=romtools.workflows.VIAdamOptimizerConfig(
            learning_rate=0.02,
            gradient_norm_tolerance=0.0,
            max_iterations=3,
        ),
        **common,
    )
    assert all(np.all(np.isfinite(value)) for value in result[:2])
    with np.load(work_dir / "iteration_2" / "restart.npz", allow_pickle=True) as restart:
        second_cholesky = restart["variational_cholesky_optimizer"]
        assert second_cholesky.shape == first_cholesky.shape
        assert np.all(np.diag(second_cholesky) > 0.0)


def test_full_covariance_mf_control_variate_uses_complete_packed_natural_score():
    mean = np.array([0.1, -0.2])
    covariance = np.array([[1.2, 0.25], [0.25, 0.7]])
    cholesky = np.linalg.cholesky(covariance)
    optimizer_fom = np.array([[0.4, -0.5], [-0.3, 0.2]])
    optimizer_base = optimizer_fom.copy()
    optimizer_extra = np.array([[0.0, -0.1], [0.8, -0.4]])
    optimizer_samples = np.vstack([optimizer_fom, optimizer_base, optimizer_extra])
    high_signal = np.array([1.5, -0.5])
    low_base_signal = np.array([1.0, -0.25])
    low_extra_signal = np.array([0.75, 0.4])

    state = {
        "parameter_samples_fom": np.zeros((2, 2)),
        "parameter_samples_rom_base": np.zeros((2, 2)),
        "parameter_samples_rom_only": np.zeros((2, 2)),
        "optimizer_samples": optimizer_samples,
        "log_joint_terms_fom": high_signal,
        "log_joint_terms_rom_base": low_base_signal,
        "log_joint_terms_rom_only": low_extra_signal,
    }
    upgraded = _upgrade_mf_state(
        state,
        mean,
        cholesky,
        baseline_method="none",
        use_mfmc_control_variate=False,
        mfmc_control_variate_mode="componentwise",
        gradient_method="natural",
        elbo_scaling_factor=1.0,
        score_function_entropy_strategy="analytic",
    )

    score_fom = np.hstack(natural_score_terms(optimizer_fom, mean, cholesky))
    score_base = np.hstack(natural_score_terms(optimizer_base, mean, cholesky))
    score_extra = np.hstack(natural_score_terms(optimizer_extra, mean, cholesky))
    high_terms = high_signal[:, None] * score_fom
    low_base_terms = low_base_signal[:, None] * score_base
    low_extra_terms = low_extra_signal[:, None] * score_extra
    low_full_terms = np.vstack([low_base_terms, low_extra_terms])
    expected = (
        np.mean(high_terms, axis=0)
        + np.mean(low_full_terms, axis=0)
        - np.mean(low_base_terms, axis=0)
    )
    expected[mean.size:] += svec(covariance)
    actual = np.concatenate([
        upgraded["natural_gradient_mean"],
        upgraded["natural_gradient_covariance_svec"],
    ])
    np.testing.assert_allclose(actual, expected)
