import numpy as np

from romtools.workflows.inverse.full_covariance_vi import (
    covariance_from_cholesky,
    entropy,
    estimate_natural_gradient,
    log_density,
    natural_score_terms,
    retract_covariance,
    robust_cholesky,
    svec,
    svec_inverse,
)


def test_svec_preserves_frobenius_inner_product():
    a = np.array([[1.2, -0.3, 0.4], [-0.3, 2.0, 0.7], [0.4, 0.7, -0.2]])
    b = np.array([[0.5, 0.8, -0.1], [0.8, -1.0, 0.2], [-0.1, 0.2, 1.4]])
    assert np.allclose(np.dot(svec(a), svec(b)), np.sum(a * b))
    assert np.allclose(svec_inverse(svec(a), 3), a)


def test_full_covariance_natural_scores_match_gaussian_fisher_inverse():
    mean = np.array([0.3, -0.7])
    covariance = np.array([[1.4, 0.45], [0.45, 0.8]])
    cholesky = np.linalg.cholesky(covariance)
    samples = np.array([[1.1, -0.2], [-0.4, -1.0], [0.2, 0.1]])
    mean_scores, covariance_scores = natural_score_terms(samples, mean, cholesky)
    precision = np.linalg.inv(covariance)

    for index, sample in enumerate(samples):
        delta = sample - mean
        ordinary_mean_score = precision @ delta
        ordinary_covariance_score = 0.5 * (
            precision @ np.outer(delta, delta) @ precision - precision
        )
        expected_mean_natural = covariance @ ordinary_mean_score
        expected_covariance_natural = (
            2.0 * covariance @ ordinary_covariance_score @ covariance
        )
        assert np.allclose(mean_scores[index], expected_mean_natural)
        assert np.allclose(
            svec_inverse(covariance_scores[index], 2),
            expected_covariance_natural,
        )


def test_joint_estimator_is_zero_at_exact_full_gaussian_fixed_point_with_loo():
    rng = np.random.default_rng(398)
    mean = np.array([0.2, -0.4])
    covariance = np.array([[1.1, 0.55], [0.55, 0.9]])
    cholesky = np.linalg.cholesky(covariance)
    samples = mean + rng.normal(size=(64, 2)) @ cholesky.T
    log_q = log_density(samples, mean, cholesky)
    result = estimate_natural_gradient(
        samples,
        mean,
        cholesky,
        scaled_log_joint_terms=log_q + 2.75,
        baseline_method="loo",
        entropy_strategy="joint",
    )
    packed = np.concatenate([
        result["natural_mean"], result["natural_covariance_svec"]
    ])
    assert np.linalg.norm(packed) < 1e-13


def test_analytic_entropy_natural_covariance_gradient_is_covariance():
    rng = np.random.default_rng(9)
    mean = np.array([0.0, 0.0])
    covariance = np.array([[1.5, -0.35], [-0.35, 0.7]])
    cholesky = np.linalg.cholesky(covariance)
    # Antithetic standard-normal samples make the stochastic score average
    # exactly zero for the zero log-joint signal; only entropy remains.
    z = rng.normal(size=(128, 2))
    samples = mean + np.vstack([z, -z]) @ cholesky.T
    result = estimate_natural_gradient(
        samples,
        mean,
        cholesky,
        scaled_log_joint_terms=np.zeros(samples.shape[0]),
        baseline_method="none",
        elbo_scaling_factor=1.7,
        entropy_strategy="analytic",
    )
    assert np.allclose(result["natural_mean"], 0.0)
    assert np.allclose(
        svec_inverse(result["natural_covariance_svec"], 2),
        1.7 * covariance,
    )


def test_covariance_retraction_is_spd_and_has_correct_first_order_direction():
    covariance = np.array([[1.0, 0.2], [0.2, 0.7]])
    cholesky = np.linalg.cholesky(covariance)
    direction = np.array([[0.15, -0.22], [-0.22, 0.05]])
    epsilon = 1e-7
    updated_cholesky, applied_scale = retract_covariance(
        cholesky,
        direction,
        epsilon,
        max_covariance_log_step=None,
    )
    updated = covariance_from_cholesky(updated_cholesky)
    finite_difference = (updated - covariance) / epsilon
    assert applied_scale == 1.0
    assert np.all(np.linalg.eigvalsh(updated) > 0.0)
    assert np.allclose(finite_difference, direction, rtol=2e-6, atol=2e-7)


def test_covariance_log_step_limit_preserves_direction_with_uniform_scaling():
    covariance = np.array([[1.0, 0.0], [0.0, 1.0]])
    cholesky = np.eye(2)
    direction = np.array([[8.0, 1.5], [1.5, -4.0]])
    updated_cholesky, applied_scale = retract_covariance(
        cholesky,
        direction,
        step_size=1.0,
        max_covariance_log_step=0.25,
    )
    updated = covariance_from_cholesky(updated_cholesky)
    assert 0.0 < applied_scale < 1.0
    assert np.all(np.linalg.eigvalsh(updated) > 0.0)
    whitened = direction
    maximum_log_step = np.max(np.abs(applied_scale * np.linalg.eigvalsh(whitened)))
    assert maximum_log_step <= 0.25 * (1.0 + 1e-12)


def test_robust_cholesky_reconstructs_positive_definite_covariance():
    covariance = np.array([[2.0, 0.7], [0.7, 1.0]])
    cholesky = robust_cholesky(covariance)
    assert np.allclose(covariance_from_cholesky(cholesky), covariance)


def test_full_covariance_entropy_matches_slogdet_formula():
    covariance = np.array([[1.8, 0.3], [0.3, 0.6]])
    cholesky = np.linalg.cholesky(covariance)
    _, log_det = np.linalg.slogdet(covariance)
    expected = 0.5 * (2.0 * (1.0 + np.log(2.0 * np.pi)) + log_det)
    assert np.isclose(entropy(cholesky), expected)
