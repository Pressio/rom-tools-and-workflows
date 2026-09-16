import numpy as np
import pytest

import romtools.workflows
from romtools.workflows.inverse.full_covariance_newton import (
    _exponential_retraction_hessian_correction,
    _fisher_whitening_map,
    _newton_step_from_hessian,
    _ordinary_score_and_logq_hessian_terms,
    _precision_from_cholesky,
    _symmetric_basis,
)
from romtools.workflows.inverse.full_covariance_vi import (
    covariance_from_cholesky,
    svec,
    svec_inverse,
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


def _exact_gaussian_elbo_gradient_hessian(
    mean,
    covariance,
    target_mean,
    target_covariance,
):
    """Return exact ELBO derivatives in ``(mu, svec(Sigma))`` coordinates."""
    dimensionality = mean.size
    basis = _symmetric_basis(dimensionality)
    target_precision = np.linalg.inv(target_covariance)
    precision = np.linalg.inv(covariance)

    gradient_mean = -target_precision @ (mean - target_mean)
    gradient_covariance = 0.5 * (precision - target_precision)
    gradient = np.concatenate([gradient_mean, svec(gradient_covariance)])

    hessian = np.zeros(
        (dimensionality + len(basis), dimensionality + len(basis)),
        dtype=float,
    )
    hessian[:dimensionality, :dimensionality] = -target_precision
    for column, basis_matrix in enumerate(basis):
        derivative = -0.5 * precision @ basis_matrix @ precision
        hessian[dimensionality:, dimensionality + column] = svec(derivative)
    return gradient, 0.5 * (hessian + hessian.T)


def _symmetric_matrix_exponential(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (eigenvectors * np.exp(eigenvalues)[None, :]) @ eigenvectors.T


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


def test_full_covariance_precision_remains_symmetric_when_ill_conditioned():
    dimensionality = 7
    indices = np.arange(dimensionality)
    covariance = 1.0 / (indices[:, None] + indices[None, :] + 1.0)
    cholesky = np.linalg.cholesky(covariance)
    precision = _precision_from_cholesky(cholesky)

    assert np.array_equal(precision, precision.T)
    scores, hessians = _ordinary_score_and_logq_hessian_terms(
        np.linspace(-0.3, 0.4, dimensionality)[None, :],
        np.zeros(dimensionality),
        cholesky,
    )
    assert np.all(np.isfinite(scores))
    assert np.all(np.isfinite(hessians))
    assert np.allclose(hessians[0], hessians[0].T)


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


def test_exponential_retraction_pullback_hessian_matches_finite_difference():
    mean = np.array([0.25, -0.15])
    covariance = np.array([[1.2, 0.2], [0.2, 0.8]])
    target_mean = np.array([0.4, -0.3])
    target_covariance = np.array([[0.35, 0.08], [0.08, 0.5]])
    cholesky = np.linalg.cholesky(covariance)
    target_precision = np.linalg.inv(target_covariance)
    dimensionality = mean.size
    basis = _symmetric_basis(dimensionality)

    gradient, ordinary_hessian = _exact_gaussian_elbo_gradient_hessian(
        mean,
        covariance,
        target_mean,
        target_covariance,
    )
    transform = _fisher_whitening_map(cholesky)
    pullback_hessian = transform.T @ ordinary_hessian @ transform
    pullback_hessian[dimensionality:, dimensionality:] += (
        _exponential_retraction_hessian_correction(
            gradient[dimensionality:],
            cholesky,
        )
    )
    pullback_hessian = 0.5 * (pullback_hessian + pullback_hessian.T)

    def local_elbo(local_coordinates):
        mean_coordinates = local_coordinates[:dimensionality]
        covariance_coordinates = local_coordinates[dimensionality:]
        local_basis_matrix = np.zeros_like(covariance)
        for coefficient, basis_matrix in zip(covariance_coordinates, basis):
            local_basis_matrix += coefficient * basis_matrix
        local_mean = mean + cholesky @ mean_coordinates
        local_covariance = cholesky @ _symmetric_matrix_exponential(
            np.sqrt(2.0) * local_basis_matrix
        ) @ cholesky.T
        delta = local_mean - target_mean
        return (
            -0.5 * delta @ target_precision @ delta
            -0.5 * np.trace(target_precision @ local_covariance)
            +0.5 * np.linalg.slogdet(local_covariance)[1]
        )

    total_size = pullback_hessian.shape[0]
    finite_difference = np.zeros_like(pullback_hessian)
    origin = np.zeros(total_size)
    objective_at_origin = local_elbo(origin)
    epsilon = 2e-4
    for row in range(total_size):
        row_direction = np.zeros(total_size)
        row_direction[row] = epsilon
        finite_difference[row, row] = (
            local_elbo(row_direction)
            - 2.0 * objective_at_origin
            + local_elbo(-row_direction)
        ) / epsilon**2
        for column in range(row):
            column_direction = np.zeros(total_size)
            column_direction[column] = epsilon
            value = (
                local_elbo(row_direction + column_direction)
                - local_elbo(row_direction - column_direction)
                - local_elbo(-row_direction + column_direction)
                + local_elbo(-row_direction - column_direction)
            ) / (4.0 * epsilon**2)
            finite_difference[row, column] = value
            finite_difference[column, row] = value

    assert np.allclose(
        pullback_hessian,
        finite_difference,
        rtol=2e-6,
        atol=1e-6,
    )


def test_full_covariance_newton_reduces_to_mean_field_on_diagonal_sine_target():
    dimensionality = 7
    observation_count = 31
    observation_locations = np.arange(1, observation_count + 1) / (
        observation_count + 1
    )
    modes = np.arange(1, dimensionality + 1, dtype=float)
    observation_matrix = -np.sin(
        np.pi * np.outer(observation_locations, modes)
    ) / (np.pi * modes[None, :]) ** 2
    noise_std = 5.0e-3
    prior_std = 0.75
    covariance = np.eye(dimensionality) * prior_std**2
    cholesky = np.linalg.cholesky(covariance)
    posterior_precision = (
        np.eye(dimensionality) / prior_std**2
        + observation_matrix.T @ observation_matrix / noise_std**2
    )
    assert np.allclose(
        posterior_precision,
        np.diag(np.diag(posterior_precision)),
        rtol=0.0,
        atol=5e-13,
    )

    precision = np.linalg.inv(covariance)
    gradient_covariance = 0.5 * (precision - posterior_precision)
    gradient = np.concatenate(
        [np.zeros(dimensionality), svec(gradient_covariance)]
    )
    basis = _symmetric_basis(dimensionality)
    ordinary_hessian = np.zeros(
        (dimensionality + len(basis), dimensionality + len(basis)),
        dtype=float,
    )
    ordinary_hessian[:dimensionality, :dimensionality] = -posterior_precision
    for column, basis_matrix in enumerate(basis):
        ordinary_hessian[dimensionality:, dimensionality + column] = svec(
            -0.5 * precision @ basis_matrix @ precision
        )
    ordinary_hessian = 0.5 * (ordinary_hessian + ordinary_hessian.T)

    state = {
        "gradient_mean": gradient[:dimensionality],
        "gradient_covariance_svec": gradient[dimensionality:],
    }
    config = VINewtonOptimizerConfig(
        newton_metric="natural",
        newton_hessian_type="full",
        newton_regularization=1e-12,
    )
    direction = _newton_step_from_hessian(
        state,
        cholesky,
        config,
        ordinary_hessian,
    )
    covariance_direction = svec_inverse(
        direction[dimensionality:],
        dimensionality,
    )
    whitened_covariance_direction = (
        np.linalg.solve(cholesky, covariance_direction)
        @ np.linalg.inv(cholesky.T)
    )
    equivalent_log_std_direction = 0.5 * np.diag(
        whitened_covariance_direction
    )
    target_curvature = np.diag(posterior_precision) * prior_std**2
    mean_field_log_std_direction = (
        1.0 - target_curvature
    ) / (2.0 * target_curvature)

    assert np.allclose(
        equivalent_log_std_direction,
        mean_field_log_std_direction,
        rtol=2e-12,
        atol=2e-12,
    )
    off_diagonal = whitened_covariance_direction - np.diag(
        np.diag(whitened_covariance_direction)
    )
    assert np.max(np.abs(off_diagonal)) < 2e-12


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
