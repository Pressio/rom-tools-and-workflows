"""Statistical estimators used by Monte Carlo UQ workflows."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MonteCarloStatistics:
    """Componentwise Monte Carlo statistics."""

    mean: np.ndarray
    variance: np.ndarray
    standard_deviation: np.ndarray
    standard_error: np.ndarray


@dataclass
class MultifidelityStatistics:
    """Componentwise two-level control-variate statistics."""

    mean: np.ndarray
    variance: np.ndarray
    standard_error: np.ndarray
    control_variate_coefficients: np.ndarray
    paired_correlations: np.ndarray


def _as_qoi_matrix(qoi_values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(qoi_values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array")
    if values.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if values.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one QoI component")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def compute_monte_carlo_statistics(qoi_values: np.ndarray) -> MonteCarloStatistics:
    """Compute componentwise Monte Carlo sample statistics.

    Args:
        qoi_values: Scalar samples with shape ``(N,)`` or flattened vector-QoI
            samples with shape ``(N, N_qoi)``. At least two are required.

    Returns:
        MonteCarloStatistics: Sample mean, unbiased sample variance
        (``ddof=1``), sample standard deviation, and standard error of the mean.
    """
    values = _as_qoi_matrix(qoi_values, "qoi_values")
    variance = np.var(values, axis=0, ddof=1)
    standard_deviation = np.sqrt(variance)
    return MonteCarloStatistics(
        mean=np.mean(values, axis=0),
        variance=variance,
        standard_deviation=standard_deviation,
        standard_error=standard_deviation / np.sqrt(values.shape[0]),
    )


def compute_paired_statistics(high_qois: np.ndarray, low_qois: np.ndarray):
    """Compute statistics from identically ordered high/low QoI pairs.

    Returns the high and low unbiased sample variances, covariance,
    control-variate coefficient ``covariance / low_variance``, and Pearson
    correlation. Components with numerically constant low-fidelity data receive
    zero coefficient and correlation.
    """
    high = _as_qoi_matrix(high_qois, "high_qois")
    low = _as_qoi_matrix(low_qois, "low_qois")
    if high.shape != low.shape:
        raise ValueError("paired high- and low-fidelity QoIs must have the same shape")

    high_centered = high - np.mean(high, axis=0)
    low_centered = low - np.mean(low, axis=0)
    denominator = high.shape[0] - 1
    high_variance = np.sum(high_centered * high_centered, axis=0) / denominator
    low_variance = np.sum(low_centered * low_centered, axis=0) / denominator
    covariance = np.sum(high_centered * low_centered, axis=0) / denominator

    scale = np.maximum(np.max(np.abs(low), axis=0), 1.0)
    low_tolerance = np.finfo(float).eps * scale * scale * 100.0
    valid_low_variance = low_variance > low_tolerance

    coefficients = np.zeros_like(covariance)
    coefficients[valid_low_variance] = (
        covariance[valid_low_variance] / low_variance[valid_low_variance]
    )

    correlations = np.zeros_like(covariance)
    correlation_denominator = np.sqrt(high_variance * low_variance)
    valid_correlation = correlation_denominator > 0.0
    correlations[valid_correlation] = (
        covariance[valid_correlation] / correlation_denominator[valid_correlation]
    )
    correlations = np.clip(correlations, -1.0, 1.0)
    return high_variance, low_variance, covariance, coefficients, correlations


def compute_multifidelity_statistics(
    high_fidelity_qois: np.ndarray,
    low_fidelity_qois: np.ndarray,
    control_variate_coefficients: Optional[np.ndarray] = None,
) -> MultifidelityStatistics:
    """Compute a componentwise two-level control-variate estimate.

    Args:
        high_fidelity_qois: High-fidelity samples with shape ``(N_H, N_qoi)``
            or ``(N_H,)``.
        low_fidelity_qois: Low-fidelity samples with shape ``(N_L, N_qoi)`` or
            ``(N_L,)``. Its first ``N_H`` rows must be paired with the
            high-fidelity samples and ``N_L`` must be at least ``N_H``.
        control_variate_coefficients: Optional scalar or length-``N_qoi``
            coefficient array. Paired samples estimate coefficients if omitted.

    Returns:
        MultifidelityStatistics: Mean estimate, estimated variance and standard
        error of that mean estimator, coefficients, and paired correlations.
    """
    high = _as_qoi_matrix(high_fidelity_qois, "high_fidelity_qois")
    low = _as_qoi_matrix(low_fidelity_qois, "low_fidelity_qois")
    if low.shape[0] < high.shape[0]:
        raise ValueError("low-fidelity sample count must be at least the high-fidelity count")
    if low.shape[1] != high.shape[1]:
        raise ValueError("high- and low-fidelity QoIs must have the same dimension")

    number_high = high.shape[0]
    number_low = low.shape[0]
    paired_low = low[:number_high]
    high_variance, low_variance, covariance, estimated_coefficients, correlations = (
        compute_paired_statistics(high, paired_low)
    )

    if control_variate_coefficients is None:
        coefficients = estimated_coefficients
    else:
        coefficients = np.asarray(control_variate_coefficients, dtype=float).reshape(-1)
        if coefficients.size == 1 and high.shape[1] > 1:
            coefficients = np.full(high.shape[1], coefficients.item())
        if coefficients.shape != (high.shape[1],):
            raise ValueError("control_variate_coefficients has the wrong QoI dimension")

    mean = (
        np.mean(high, axis=0)
        + coefficients * (np.mean(low, axis=0) - np.mean(paired_low, axis=0))
    )
    inverse_count_difference = 1.0 / number_high - 1.0 / number_low
    estimator_variance = (
        high_variance / number_high
        + inverse_count_difference
        * (coefficients * coefficients * low_variance - 2.0 * coefficients * covariance)
    )

    roundoff_scale = np.maximum(high_variance / number_high, 1.0)
    tolerance = np.finfo(float).eps * roundoff_scale * 1000.0
    if np.any(estimator_variance < -tolerance):
        raise ValueError("estimated multifidelity variance is negative")
    estimator_variance = np.maximum(estimator_variance, 0.0)

    return MultifidelityStatistics(
        mean=mean,
        variance=estimator_variance,
        standard_error=np.sqrt(estimator_variance),
        control_variate_coefficients=coefficients,
        paired_correlations=correlations,
    )
