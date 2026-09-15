import numpy as np
import pytest

import romtools.workflows
from romtools.workflows.inverse.full_covariance_newton import (
    _fisher_whitening_map,
    _ordinary_score_and_logq_hessian_terms,
    _symmetric_basis,
)
from romtools.workflows.inverse.full_covariance_vi import (
    covariance_from_cholesky,
    svec,
    svec_size,
)
from romtools.workflows.inverse.vi_optimization_methods import (
    VINewtonOptimizerConfig,
    VIStochasticNonmonotoneLineSearchConfig,
)
from romtools.workflows.parameter_spaces import (
    GaussianParameterSpace,
    MonteCarloSampler,
    MultivariateGaussianParameterSpace,
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


def _prior():
    return GaussianParameterSpace(
        parameter_names=["theta0", "theta1"],
        means=np.zeros(2),
        stds=np.ones(2),
        sampler=MonteCarloSampler,
    )


def _full_initializer():
    return MultivariateGaussianParameterSpace(
        parameter_names=["theta0", "theta1"],
        means=np.zeros(2),
        covariance=np.eye(2),
        sampler=MonteCarloSampler,
    )


def _ordinary_score(sample, mean, covariance):
    precision = np.linalg.inv(covariance)
    score_mean = precision @ (sample - mean)
    score_covariance = 0.5 * (
        np.outer(score_mean, score_mean) - precision
    )
    return np.concatenate([score_mean, svec(score_covariance)])


def test_full_covariance_logq_hessian_matches_finite_difference():
    mean = np.array([0.2, -0.1])
    covariance = np.array([[1.3, 0.25], [0.25, 0.8]])
    cholesky = np.linalg.cholesky(covariance)
    sample = np.array([0.7, -0.4])
    _, hessians = _ordinary_score_and_logq_hessian_terms(
        sample[None, :], mean, cholesky
    )
    analytic = hessians[0]

    dimensionality = mean.size
    basis = _symmetric_basis(dimensionality)
    total_size = dimensionality + svec_size(dimensionality)
    finite_difference = np.zeros((total_size, total_size))
    epsilon = 1e-6
    for column in range(total_size):
        mean_plus = mean.copy()
        mean_minus = mean.copy()
        covariance_plus = covariance.copy()
        covariance_minus = covariance.copy()
        if column < dimensionality:
            mean_plus[column] += epsilon
            mean_minus[column] -= epsilon
        else:
            direction = basis[column - dimensionality]
            covariance_plus += epsilon * direction
            covariance_minus -= epsilon * direction
        finite_difference[:, column] = (
            _ordinary_score(sample, mean_plus, covariance_plus)
            - _ordinary_score(sample, mean_minus, covariance_minus)
        ) / (2.0 * epsilon)

    assert np.allclose(analytic, finite_difference, rtol=2e-6, atol=2e-7)


def test_full_covariance_fisher_whitening_is_identity():
    covariance = np.array([[1.4, 0.3], [0.3, 0.9]])
    cholesky = np.linalg.cholesky(covariance)
    precision = np.linalg.inv(covariance)
    dimensionality = covariance.shape[0]
    basis = _symmetric_basis(dimensionality)
    covariance_size = len(basis)
    fisher = np.zeros(
        (dimensionality + covariance_size, dimensionality + covariance_size)
    )
    fisher[:dimensionality, :dimensionality] = precision
    for row, row_basis in enumerate(basis):
        for column, column_basis in enumerate(basis):
            fisher[dimensionality + row, dimensionality + column] = (
                0.5
                * np.trace(
                    precision
                    @ row_basis
                    @ precision
                    @ column_basis
                )
            )

    whitening = _fisher_whitening_map(cholesky)
    identity = whitening.T @ fisher @ whitening
    assert np.allclose(identity, np.eye(identity.shape[0]), rtol=1e-12, atol=1e-12)


@pytest.mark.mpi_skip
def test_public_full_covariance_natural_newton_runs(tmp_path):
    work_dir = tmp_path / "full_newton"
    means, stds, parameter_samples, qois = romtools.workflows.run_vi(
        model=IdentityTwoParameterModel(),
        prior_parameter_space=_prior(),
        initial_variational_parameter_space=_full_initializer(),
        observations=np.array([0.35, -0.25]),
        observations_covariance=np.array([[0.18, 0.10], [0.10, 0.22]]),
        absolute_work_dir=str(work_dir),
        sample_size=16,
        optimizer_method="newton",
        optimizer_config=VINewtonOptimizerConfig(
            newton_metric="natural",
            newton_regularization=1e-4,
            gradient_norm_tolerance=0.0,
            max_iterations=2,
        ),
        line_search_method="stochastic_nonmonotone",
        line_search_config=VIStochasticNonmonotoneLineSearchConfig(
            max_step_size=1.0,
        ),
        baseline_method="loo",
        score_function_entropy_strategy="joint",
        bounded_parameter_handling="clip",
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
        for covariance in covariance_history:
            assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
