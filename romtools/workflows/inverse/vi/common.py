"""Shared density, baseline, and uncertainty helpers for VI estimators."""

import numpy as np


def _compute_standard_error(samples: np.ndarray) -> float:
    sample_count = samples.size
    if sample_count <= 1:
        return 0.0
    return float(np.std(samples, ddof=1) / np.sqrt(sample_count))


def _compute_component_standard_error(samples: np.ndarray) -> np.ndarray:
    sample_count = samples.shape[0]
    if sample_count <= 1:
        return np.zeros(samples.shape[1], dtype=float)
    return np.std(samples, axis=0, ddof=1) / np.sqrt(sample_count)


def _compute_gradient_signal_to_noise_ratio(gradient: np.ndarray,
                                            gradient_standard_error: np.ndarray) -> float:
    signal_norm = float(np.linalg.norm(gradient))
    noise_norm = float(np.linalg.norm(gradient_standard_error))
    if noise_norm <= 0.0:
        if signal_norm <= 0.0:
            return 0.0
        return np.inf
    return signal_norm / noise_norm


def _compute_optimal_baseline(values: np.ndarray, score_functions: np.ndarray,
                              *, dtype=None) -> np.ndarray:
    numerator = np.mean(values[:, None] * (score_functions ** 2), axis=0)
    denominator = np.mean(score_functions ** 2, axis=0)
    baseline = np.zeros_like(numerator, dtype=dtype)
    nonzero_denominator = denominator > 0.0
    baseline[nonzero_denominator] = numerator[nonzero_denominator] / denominator[nonzero_denominator]
    return baseline


def _compute_leave_one_out_baseline(values: np.ndarray) -> np.ndarray:
    sample_count = values.size
    if sample_count <= 1:
        return np.zeros_like(values)
    return (np.sum(values) - values) / float(sample_count - 1)


def _normalize_baseline_method(baseline_method: str):
    method = baseline_method.strip().lower()
    if method not in ('none', 'loo', 'optimal'):
        raise ValueError(
            f"Unsupported baseline_method '{baseline_method}'. "
            "Supported options are 'none', 'loo', and 'optimal'."
        )
    return method


def _compute_variational_log_densities(
        optimizer_samples: np.ndarray,
        variational_mean: np.ndarray,
        variational_std: np.ndarray,
        variational_correlation_cholesky: np.ndarray = None) -> np.ndarray:
    """Evaluate the Gaussian variational density in optimizer coordinates."""
    normalized = (
        optimizer_samples - variational_mean[None, :]
    ) / variational_std[None, :]
    log_det_correlation = 0.0
    if variational_correlation_cholesky is None:
        quadratic_terms = np.sum(normalized ** 2, axis=1)
    else:
        whitened = np.linalg.solve(
            variational_correlation_cholesky,
            normalized.transpose(),
        ).transpose()
        quadratic_terms = np.sum(whitened ** 2, axis=1)
        log_det_correlation = 2.0 * np.sum(
            np.log(np.diag(variational_correlation_cholesky))
        )
    log_det_covariance = (
        2.0 * np.sum(np.log(variational_std)) + log_det_correlation
    )
    normalizer = variational_mean.size * np.log(2.0 * np.pi) + log_det_covariance
    return -0.5 * (quadratic_terms + normalizer)


def _normalize_mfmc_control_variate_mode(mfmc_control_variate_mode: str) -> str:
    mode = mfmc_control_variate_mode.strip().lower()
    if mode in ("componentwise", "component", "diag", "diagonal"):
        return "componentwise"
    if mode in ("scalar", "isotropic"):
        return "scalar"
    if mode in ("matrix", "joint", "full"):
        return "matrix"
    raise ValueError(
        f"Unsupported mfmc_control_variate_mode '{mfmc_control_variate_mode}'. "
        "Supported options are 'componentwise', 'scalar', and 'matrix'."
    )
