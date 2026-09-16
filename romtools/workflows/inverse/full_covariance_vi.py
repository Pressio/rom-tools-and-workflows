"""Utilities for full-covariance Gaussian variational inference.

The full-covariance variational family is represented in optimizer coordinates
as ``q(x) = N(mu, Sigma)`` with ``Sigma = L L.T``.  The Cholesky factor is a
numerical representation; natural-gradient directions are expressed in the
mean and covariance tangent spaces.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def svec_size(dimensionality: int) -> int:
    """Return the number of independent entries in a symmetric matrix."""
    if dimensionality < 1:
        raise ValueError("dimensionality must be positive")
    return dimensionality * (dimensionality + 1) // 2


def infer_svec_dimension(size: int) -> int:
    """Infer the matrix dimension associated with an ``svec`` length."""
    if size < 1:
        raise ValueError("svec size must be positive")
    dimensionality = int((math.sqrt(8 * size + 1) - 1) / 2)
    if svec_size(dimensionality) != size:
        raise ValueError(f"{size} is not a valid symmetric-vector size")
    return dimensionality


def svec(matrix: np.ndarray) -> np.ndarray:
    """Isometrically vectorize a symmetric matrix.

    Lower-triangular entries are stored in NumPy's ``tril_indices`` order.
    Off-diagonal entries are multiplied by ``sqrt(2)`` so Euclidean inner
    products of packed vectors equal Frobenius inner products of the matrices.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("svec expects a square 2D matrix")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
        raise ValueError("svec expects a symmetric matrix")
    rows, cols = np.tril_indices(matrix.shape[0])
    packed = matrix[rows, cols].copy()
    packed[rows != cols] *= np.sqrt(2.0)
    return packed


def svec_inverse(vector: np.ndarray, dimensionality: Optional[int] = None) -> np.ndarray:
    """Invert :func:`svec`."""
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1:
        raise ValueError("svec_inverse expects a 1D vector")
    if dimensionality is None:
        dimensionality = infer_svec_dimension(vector.size)
    if svec_size(dimensionality) != vector.size:
        raise ValueError("vector size does not match dimensionality")
    rows, cols = np.tril_indices(dimensionality)
    values = vector.copy()
    values[rows != cols] /= np.sqrt(2.0)
    matrix = np.zeros((dimensionality, dimensionality), dtype=float)
    matrix[rows, cols] = values
    matrix[cols, rows] = values
    return matrix


def covariance_from_cholesky(cholesky: np.ndarray) -> np.ndarray:
    """Return the symmetric covariance represented by a lower Cholesky factor."""
    cholesky = np.asarray(cholesky, dtype=float)
    if cholesky.ndim != 2 or cholesky.shape[0] != cholesky.shape[1]:
        raise ValueError("cholesky must be a square 2D array")
    covariance = cholesky @ cholesky.T
    return 0.5 * (covariance + covariance.T)


def robust_cholesky(
    covariance: np.ndarray,
    initial_jitter: float = 0.0,
    max_jitter: float = 1e-6,
) -> np.ndarray:
    """Compute a Cholesky factor, adding only roundoff-scale jitter if needed."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    covariance = 0.5 * (covariance + covariance.T)
    dimensionality = covariance.shape[0]
    jitter = max(float(initial_jitter), 0.0)
    while True:
        try:
            return np.linalg.cholesky(covariance + jitter * np.eye(dimensionality))
        except np.linalg.LinAlgError as exc:
            if jitter >= max_jitter:
                raise ValueError("covariance must be positive definite") from exc
            jitter = max(1e-14, 10.0 * jitter)


def marginal_std_from_cholesky(cholesky: np.ndarray) -> np.ndarray:
    """Return marginal standard deviations for a covariance Cholesky factor."""
    covariance = covariance_from_cholesky(cholesky)
    return np.sqrt(np.maximum(np.diag(covariance), 0.0))


def correlation_cholesky_from_covariance(covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return marginal standard deviations and a correlation Cholesky factor."""
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if np.any(std <= 0.0):
        raise ValueError("full-covariance Gaussian requires positive marginal variances")
    correlation = covariance / (std[:, None] * std[None, :])
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    return std, robust_cholesky(correlation, initial_jitter=0.0, max_jitter=1e-8)


def rescale_cholesky_marginals(
    cholesky: np.ndarray,
    minimum_std: np.ndarray | float,
    maximum_std: np.ndarray | float,
) -> np.ndarray:
    """Rescale covariance marginals while preserving its correlation matrix."""
    cholesky = np.asarray(cholesky, dtype=float)
    current_std = marginal_std_from_cholesky(cholesky)
    minimum_std = np.broadcast_to(np.asarray(minimum_std, dtype=float), current_std.shape)
    maximum_std = np.broadcast_to(np.asarray(maximum_std, dtype=float), current_std.shape)
    if np.any(minimum_std <= 0.0):
        raise ValueError("minimum_std must be positive")
    if np.any(maximum_std < minimum_std):
        raise ValueError("maximum_std must be >= minimum_std")
    target_std = np.clip(current_std, minimum_std, maximum_std)
    scale = target_std / current_std
    # D @ L is lower triangular and represents D Sigma D.
    return scale[:, None] * cholesky


def log_density(
    optimizer_samples: np.ndarray,
    mean: np.ndarray,
    cholesky: np.ndarray,
) -> np.ndarray:
    """Evaluate a full-covariance Gaussian log density in optimizer coordinates."""
    optimizer_samples = np.asarray(optimizer_samples, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cholesky = np.asarray(cholesky, dtype=float)
    centered = optimizer_samples - mean[None, :]
    whitened = np.linalg.solve(cholesky, centered.T).T
    quadratic = np.sum(whitened**2, axis=1)
    log_det_covariance = 2.0 * np.sum(np.log(np.diag(cholesky)))
    normalizer = mean.size * np.log(2.0 * np.pi) + log_det_covariance
    return -0.5 * (quadratic + normalizer)


def entropy(cholesky: np.ndarray, scaling_factor: float = 1.0) -> float:
    """Return Gaussian entropy, optionally scaled with the ELBO."""
    cholesky = np.asarray(cholesky, dtype=float)
    dimensionality = cholesky.shape[0]
    log_det_covariance = 2.0 * np.sum(np.log(np.diag(cholesky)))
    value = 0.5 * (
        dimensionality * (1.0 + np.log(2.0 * np.pi)) + log_det_covariance
    )
    return float(scaling_factor * value)


def natural_score_terms(
    optimizer_samples: np.ndarray,
    mean: np.ndarray,
    cholesky: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-sample natural scores for mean and covariance.

    The covariance scores are returned in ``svec`` coordinates.
    """
    optimizer_samples = np.asarray(optimizer_samples, dtype=float)
    mean = np.asarray(mean, dtype=float)
    covariance = covariance_from_cholesky(cholesky)
    delta = optimizer_samples - mean[None, :]
    covariance_terms = np.empty((optimizer_samples.shape[0], svec_size(mean.size)))
    for sample_index, delta_i in enumerate(delta):
        covariance_terms[sample_index] = svec(
            np.outer(delta_i, delta_i) - covariance
        )
    return delta, covariance_terms


def _leave_one_out_baseline(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.zeros_like(values)
    return (np.sum(values) - values) / float(values.size - 1)


def _optimal_component_baseline(values: np.ndarray, scores: np.ndarray) -> np.ndarray:
    numerator = np.mean(values[:, None] * scores**2, axis=0)
    denominator = np.mean(scores**2, axis=0)
    baseline = np.zeros(scores.shape[1], dtype=float)
    nonzero = denominator > 0.0
    baseline[nonzero] = numerator[nonzero] / denominator[nonzero]
    return baseline


def _component_standard_error(terms: np.ndarray) -> np.ndarray:
    if terms.shape[0] <= 1:
        return np.zeros(terms.shape[1], dtype=float)
    return np.std(terms, axis=0, ddof=1) / np.sqrt(terms.shape[0])


def estimate_natural_gradient(
    optimizer_samples: np.ndarray,
    mean: np.ndarray,
    cholesky: np.ndarray,
    scaled_log_joint_terms: np.ndarray,
    baseline_method: str,
    elbo_scaling_factor: float = 1.0,
    entropy_strategy: str = "analytic",
) -> dict:
    """Estimate full-Gaussian natural and ordinary ELBO gradients.

    ``scaled_log_joint_terms`` must already include the ELBO scaling factor and,
    for transformed bounded parameters, the transform-Jacobian contribution.
    Analytic entropy is added exactly after the stochastic log-joint estimate.
    """
    optimizer_samples = np.asarray(optimizer_samples, dtype=float)
    scaled_log_joint_terms = np.asarray(scaled_log_joint_terms, dtype=float)
    if scaled_log_joint_terms.shape != (optimizer_samples.shape[0],):
        raise ValueError("scaled_log_joint_terms shape must match the sample count")
    mean_scores, covariance_scores = natural_score_terms(
        optimizer_samples, mean, cholesky
    )
    entropy_strategy = entropy_strategy.strip().lower()
    if entropy_strategy not in ("analytic", "joint"):
        raise ValueError("entropy_strategy must be 'analytic' or 'joint'")
    weights = scaled_log_joint_terms.copy()
    if entropy_strategy == "joint":
        weights -= elbo_scaling_factor * log_density(
            optimizer_samples, mean, cholesky
        )

    baseline_method = baseline_method.strip().lower()
    if baseline_method == "optimal":
        baseline_mean = _optimal_component_baseline(weights, mean_scores)
        baseline_covariance = _optimal_component_baseline(weights, covariance_scores)
        centered_mean = weights[:, None] - baseline_mean[None, :]
        centered_covariance = weights[:, None] - baseline_covariance[None, :]
    elif baseline_method == "loo":
        loo = _leave_one_out_baseline(weights)
        baseline_mean = np.zeros(mean_scores.shape[1], dtype=float)
        baseline_covariance = np.zeros(covariance_scores.shape[1], dtype=float)
        centered_mean = (weights - loo)[:, None]
        centered_covariance = centered_mean
    elif baseline_method == "none":
        baseline_mean = np.zeros(mean_scores.shape[1], dtype=float)
        baseline_covariance = np.zeros(covariance_scores.shape[1], dtype=float)
        centered_mean = weights[:, None]
        centered_covariance = weights[:, None]
    else:
        raise ValueError("baseline_method must be 'none', 'loo', or 'optimal'")

    mean_terms = centered_mean * mean_scores
    covariance_terms = centered_covariance * covariance_scores
    natural_mean = np.mean(mean_terms, axis=0)
    natural_covariance = np.mean(covariance_terms, axis=0)
    covariance = covariance_from_cholesky(cholesky)
    if entropy_strategy == "analytic":
        natural_covariance += elbo_scaling_factor * svec(covariance)

    natural_standard_error = np.concatenate([
        _component_standard_error(mean_terms),
        _component_standard_error(covariance_terms),
    ])
    natural_vector = np.concatenate([natural_mean, natural_covariance])
    noise_norm = float(np.linalg.norm(natural_standard_error))
    signal_norm = float(np.linalg.norm(natural_vector))
    snr = np.inf if noise_norm == 0.0 and signal_norm > 0.0 else (
        0.0 if noise_norm == 0.0 else signal_norm / noise_norm
    )

    # Recover ordinary gradients for directional-derivative diagnostics and
    # line-search slope calculations.  Never form Sigma^{-1} explicitly.
    ordinary_mean = np.linalg.solve(covariance, natural_mean)
    natural_covariance_matrix = svec_inverse(natural_covariance, mean.size)
    left_solve = np.linalg.solve(covariance, natural_covariance_matrix)
    ordinary_covariance_matrix = 0.5 * np.linalg.solve(
        covariance, left_solve.T
    ).T
    ordinary_covariance_matrix = 0.5 * (
        ordinary_covariance_matrix + ordinary_covariance_matrix.T
    )

    return {
        "natural_mean": natural_mean,
        "natural_covariance_svec": natural_covariance,
        "ordinary_mean": ordinary_mean,
        "ordinary_covariance_svec": svec(ordinary_covariance_matrix),
        "baseline_mean": baseline_mean,
        "baseline_covariance_svec": baseline_covariance,
        "gradient_standard_error": natural_standard_error,
        "gradient_signal_to_noise_ratio": float(snr),
        "mean_terms": mean_terms,
        "covariance_terms": covariance_terms,
        "weights": weights,
    }


def retract_covariance(
    cholesky: np.ndarray,
    covariance_direction: np.ndarray,
    step_size: float,
    max_covariance_log_step: Optional[float] = 1.0,
) -> Tuple[np.ndarray, float]:
    """Apply an SPD-preserving covariance retraction.

    The requested log-step is uniformly rescaled, rather than clipping
    individual eigenvalues, so the tangent direction is preserved.

    Returns
    -------
    new_cholesky, applied_scale
        ``applied_scale`` multiplies ``step_size`` after log-step limiting.
    """
    cholesky = np.asarray(cholesky, dtype=float)
    covariance_direction = np.asarray(covariance_direction, dtype=float)
    if covariance_direction.shape != cholesky.shape:
        raise ValueError("covariance_direction shape must match cholesky")
    covariance_direction = 0.5 * (
        covariance_direction + covariance_direction.T
    )
    left = np.linalg.solve(cholesky, covariance_direction)
    whitened = np.linalg.solve(cholesky, left.T).T
    whitened = 0.5 * (whitened + whitened.T)
    eigenvalues, eigenvectors = np.linalg.eigh(whitened)
    applied_scale = 1.0
    if max_covariance_log_step is not None:
        if max_covariance_log_step <= 0.0:
            raise ValueError("max_covariance_log_step must be positive or None")
        maximum_requested = float(np.max(np.abs(step_size * eigenvalues)))
        if maximum_requested > max_covariance_log_step:
            applied_scale = max_covariance_log_step / maximum_requested
    log_eigenvalues = step_size * applied_scale * eigenvalues
    exponential = (
        eigenvectors * np.exp(log_eigenvalues)[None, :]
    ) @ eigenvectors.T
    covariance_new = cholesky @ exponential @ cholesky.T
    covariance_new = 0.5 * (covariance_new + covariance_new.T)
    return robust_cholesky(covariance_new), float(applied_scale)


def packed_direction_to_covariance(
    packed_direction: np.ndarray,
    dimensionality: int,
) -> np.ndarray:
    """Convert an ``svec`` covariance direction back to matrix form."""
    return svec_inverse(packed_direction, dimensionality)
