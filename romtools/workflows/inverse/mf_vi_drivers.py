import copy
import os
import time
from typing import Optional, Tuple

import numpy as np

from romtools.workflows.inverse.eki_utils import bound_samples, run_eki_iteration
from romtools.workflows.inverse.mf_eki_drivers import GaussianProcessQoiModelBuilderWithTrainingData
from romtools.workflows.inverse.vi_optimization_methods import (
    SteepestDescentSolver,
    VIGradientOptimizerConfig,
    VINewtonOptimizerConfig,
    VILegacyLineSearchConfig,
    VIStochasticNonmonotoneLineSearchConfig,
    _normalize_line_search_method as _normalize_line_search_method_shared,
    _normalize_newton_metric,
    _resolve_line_search_config,
    _resolve_optimizer_config,
    _raise_for_unused_legacy_kwargs,
)
from romtools.workflows.inverse.vi_drivers import (
    _compute_correlation_cholesky_from_samples,
    _clip_variational_log_std,
    _enforce_variational_log_std_bounds,
    _compute_gradient_norm,
    _compute_leave_one_out_baseline,
    _compute_log_likelihoods,
    _compute_log_likelihood_precision_operator,
    _compute_optimal_baseline,
    _compute_relative_mse,
    _compute_newton_metric_scale,
    _compute_newton_step,
    _compute_update_directions,
    _compute_variational_std,
    _draw_parameter_samples,
    _normalize_baseline_method,
    _normalize_bounded_parameter_handling,
    _normalize_variational_distribution,
    _print_vi_parameters,
    _resolve_elbo_scaling_factor,
    _resolve_parameter_bounds,
    _resolve_restart_file,
    _transform_parameter_to_optimizer,
)
from romtools.workflows.model_builders import QoiModelBuilderWithTrainingData
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace


def _compute_rom_relative_error(rom_qois: np.ndarray, fom_qois: np.ndarray, epsilon: float = 1e-16) -> float:
    denominator = np.linalg.norm(fom_qois)
    if denominator <= epsilon:
        return float(np.linalg.norm(rom_qois - fom_qois))
    return float(np.linalg.norm(rom_qois - fom_qois) / denominator)


def _build_iteration_training_data(run_directory_base: str,
                                   parameter_samples: np.ndarray,
                                   qoi_results) -> Tuple[list, np.ndarray, np.ndarray]:
    sample_count = parameter_samples.shape[0]
    training_dirs = [f'{run_directory_base}{sample_index}' for sample_index in range(sample_count)]
    training_dirs.append(f'{run_directory_base}mean')
    training_parameters = np.vstack([
        parameter_samples,
        np.mean(parameter_samples, axis=0)[None, :],
    ])
    training_qois = np.vstack([
        qoi_results['qois'].T,
        qoi_results['mean-qoi'][None, :],
    ])
    return training_dirs, training_parameters, training_qois


def _append_training_data(training_dirs: list,
                          training_parameters: np.ndarray,
                          training_qois: np.ndarray,
                          iteration_dirs: list,
                          iteration_parameters: np.ndarray,
                          iteration_qois: np.ndarray) -> Tuple[list, np.ndarray, np.ndarray]:
    if training_parameters is None or training_qois is None:
        return copy.deepcopy(iteration_dirs), iteration_parameters.copy(), iteration_qois.copy()
    return (
        list(training_dirs) + list(iteration_dirs),
        np.vstack([training_parameters, iteration_parameters]),
        np.vstack([training_qois, iteration_qois]),
    )


def _compute_mfmc_alpha(high_terms: np.ndarray,
                        low_terms: np.ndarray,
                        epsilon: float = 1e-12) -> np.ndarray:
    trailing_shape = high_terms.shape[1:]
    high_flat = high_terms.reshape(high_terms.shape[0], -1)
    low_flat = low_terms.reshape(low_terms.shape[0], -1)
    centered_high = high_flat - np.mean(high_flat, axis=0, keepdims=True)
    centered_low = low_flat - np.mean(low_flat, axis=0, keepdims=True)
    covariance = np.mean(centered_high * centered_low, axis=0)
    low_variance = np.mean(centered_low ** 2, axis=0)
    # Fall back to no control variate when low-fidelity variance is near zero.
    alphas = np.zeros_like(covariance)
    nonzero = low_variance > epsilon
    alphas[nonzero] = covariance[nonzero] / low_variance[nonzero]
    return alphas.reshape(trailing_shape)


def _mfmc_gradient_estimator(high_terms: np.ndarray,
                             low_terms_base: np.ndarray,
                             low_terms_extra: np.ndarray,
                             use_control_variate: bool) -> Tuple[np.ndarray, np.ndarray]:
    trailing_shape = high_terms.shape[1:]
    high_flat = high_terms.reshape(high_terms.shape[0], -1)
    low_base_flat = low_terms_base.reshape(low_terms_base.shape[0], -1)
    if low_terms_extra is None or low_terms_extra.shape[0] == 0:
        return (
            np.mean(high_flat, axis=0).reshape(trailing_shape),
            np.zeros(high_flat.shape[1]).reshape(trailing_shape),
        )

    if use_control_variate:
        alpha = _compute_mfmc_alpha(high_terms, low_terms_base)
    else:
        alpha = np.ones(trailing_shape)
    alpha_flat = alpha.reshape(-1)

    # Hierarchical low-fidelity correction using independent low-fidelity samples.
    # low_terms_full = [low_terms_base; low_terms_extra] with low_terms_base size
    # matched to the high-fidelity sample count.
    low_extra_flat = low_terms_extra.reshape(low_terms_extra.shape[0], -1)
    low_terms_full = np.vstack([low_base_flat, low_extra_flat])
    estimate_flat = (
        np.mean(high_flat, axis=0)
        + alpha_flat * (np.mean(low_terms_full, axis=0) - np.mean(low_base_flat, axis=0))
    )
    return estimate_flat.reshape(trailing_shape), alpha


def _compute_mfmc_reinforce_gradients(optimizer_samples_fom: np.ndarray,
                                      optimizer_samples_rom_base: np.ndarray,
                                      optimizer_samples_rom_extra: np.ndarray,
                                      variational_mean: np.ndarray,
                                      variational_std: np.ndarray,
                                      fom_log_likelihoods: np.ndarray,
                                      rom_log_likelihoods_base: np.ndarray,
                                      rom_log_likelihoods_extra: np.ndarray,
                                      baseline_method: str,
                                      use_mfmc_control_variate: bool,
                                      variational_correlation_cholesky: np.ndarray = None,
                                      elbo_scaling_factor: float = 1.0):
    # Build score-function terms for either diagonal or fixed-correlation Gaussian families.
    def _compute_scores(optimizer_samples: np.ndarray):
        centered = optimizer_samples - variational_mean[None, :]
        normalized = centered / variational_std[None, :]
        if variational_correlation_cholesky is None:
            score_mean = centered / (variational_std[None, :] ** 2)
            score_log_std = normalized ** 2 - 1.0
        else:
            correlation_solve = np.linalg.solve(
                variational_correlation_cholesky,
                normalized.transpose(),
            )
            correlation_inverse_times_normalized = np.linalg.solve(
                variational_correlation_cholesky.transpose(),
                correlation_solve,
            ).transpose()
            score_mean = correlation_inverse_times_normalized / variational_std[None, :]
            score_log_std = normalized * correlation_inverse_times_normalized - 1.0
        return score_mean, score_log_std

    score_mean_fom, score_log_std_fom = _compute_scores(optimizer_samples_fom)
    score_mean_rom_base, score_log_std_rom_base = _compute_scores(optimizer_samples_rom_base)

    # Additional low-fidelity samples are optional; if absent we fall back to single-level MC.
    if optimizer_samples_rom_extra is not None and optimizer_samples_rom_extra.shape[0] > 0:
        score_mean_rom_extra, score_log_std_rom_extra = _compute_scores(optimizer_samples_rom_extra)
    else:
        score_mean_rom_extra = None
        score_log_std_rom_extra = None

    # Keep objective scaling consistent with ELBO scaling used in other VI paths.
    scaled_fom_log_likelihoods = elbo_scaling_factor * fom_log_likelihoods
    scaled_rom_log_likelihoods_base = elbo_scaling_factor * rom_log_likelihoods_base
    scaled_rom_log_likelihoods_extra = elbo_scaling_factor * rom_log_likelihoods_extra

    # Compute shared baselines and centered likelihood weights.
    # Baselines are estimated from FOM statistics and reused for ROM terms.
    baseline_method = _normalize_baseline_method(baseline_method)
    if baseline_method == 'optimal':
        baseline_mean = _compute_optimal_baseline(scaled_fom_log_likelihoods, score_mean_fom)
        baseline_log_std = _compute_optimal_baseline(scaled_fom_log_likelihoods, score_log_std_fom)
        centered_fom_log_likelihoods = scaled_fom_log_likelihoods
        centered_rom_log_likelihoods_base = scaled_rom_log_likelihoods_base
        centered_rom_log_likelihoods_extra = scaled_rom_log_likelihoods_extra
    elif baseline_method == 'loo':
        baseline_mean = np.zeros_like(variational_mean)
        baseline_log_std = np.zeros_like(variational_mean)
        centered_fom_log_likelihoods = (
            scaled_fom_log_likelihoods - _compute_leave_one_out_baseline(scaled_fom_log_likelihoods)
        )
        centered_rom_log_likelihoods_base = (
            scaled_rom_log_likelihoods_base
            - _compute_leave_one_out_baseline(scaled_rom_log_likelihoods_base)
        )
        centered_rom_log_likelihoods_extra = (
            scaled_rom_log_likelihoods_extra
            - _compute_leave_one_out_baseline(scaled_rom_log_likelihoods_extra)
        )
    else:
        baseline_mean = np.zeros_like(variational_mean)
        baseline_log_std = np.zeros_like(variational_mean)
        centered_fom_log_likelihoods = scaled_fom_log_likelihoods
        centered_rom_log_likelihoods_base = scaled_rom_log_likelihoods_base
        centered_rom_log_likelihoods_extra = scaled_rom_log_likelihoods_extra

    # High terms use FOM samples; low base terms use an independent low-fidelity sample set.
    high_mean_terms = (centered_fom_log_likelihoods[:, None] - baseline_mean[None, :]) * score_mean_fom
    low_mean_terms_base = (
        centered_rom_log_likelihoods_base[:, None] - baseline_mean[None, :]
    ) * score_mean_rom_base
    high_log_std_terms = (
        centered_fom_log_likelihoods[:, None] - baseline_log_std[None, :]
    ) * score_log_std_fom
    low_log_std_terms_base = (
        centered_rom_log_likelihoods_base[:, None] - baseline_log_std[None, :]
    ) * score_log_std_rom_base

    if score_mean_rom_extra is None:
        low_mean_terms_extra = None
        low_log_std_terms_extra = None
    else:
        low_mean_terms_extra = (
            centered_rom_log_likelihoods_extra[:, None] - baseline_mean[None, :]
        ) * score_mean_rom_extra
        low_log_std_terms_extra = (
            (centered_rom_log_likelihoods_extra[:, None] - baseline_log_std[None, :]) * score_log_std_rom_extra
        )

    # Estimate E[FOM term] with a hierarchical low-fidelity correction:
    # mean(FOM) + alpha * (mean(ROM_full) - mean(ROM_base)).
    gradient_mean, alpha_mean = _mfmc_gradient_estimator(
        high_mean_terms,
        low_mean_terms_base,
        low_mean_terms_extra,
        use_mfmc_control_variate,
    )
    gradient_log_std, alpha_log_std = _mfmc_gradient_estimator(
        high_log_std_terms,
        low_log_std_terms_base,
        low_log_std_terms_extra,
        use_mfmc_control_variate,
    )
    # Entropy gradient contribution for log-std parameters.
    gradient_log_std += elbo_scaling_factor
    return gradient_mean, gradient_log_std, baseline_mean, baseline_log_std, alpha_mean, alpha_log_std


def _compute_mfmc_reinforce_hessian_diagonal(optimizer_samples_fom: np.ndarray,
                                             optimizer_samples_rom_base: np.ndarray,
                                             optimizer_samples_rom_extra: np.ndarray,
                                             variational_mean: np.ndarray,
                                             variational_std: np.ndarray,
                                             fom_log_likelihoods: np.ndarray,
                                             rom_log_likelihoods_base: np.ndarray,
                                             rom_log_likelihoods_extra: np.ndarray,
                                             baseline_method: str,
                                             use_mfmc_control_variate: bool,
                                             variational_correlation_cholesky: np.ndarray = None,
                                             elbo_scaling_factor: float = 1.0):
    hessian_full = _compute_mfmc_reinforce_hessian_full(
        optimizer_samples_fom,
        optimizer_samples_rom_base,
        optimizer_samples_rom_extra,
        variational_mean,
        variational_std,
        fom_log_likelihoods,
        rom_log_likelihoods_base,
        rom_log_likelihoods_extra,
        baseline_method,
        use_mfmc_control_variate,
        variational_correlation_cholesky,
        elbo_scaling_factor,
    )
    dimensionality = variational_mean.size
    hessian_diagonal_mean = np.diag(hessian_full[:dimensionality, :dimensionality])
    hessian_diagonal_log_std = np.diag(hessian_full[dimensionality:, dimensionality:])
    return hessian_diagonal_mean, hessian_diagonal_log_std


def _compute_mfmc_reinforce_hessian_full(optimizer_samples_fom: np.ndarray,
                                         optimizer_samples_rom_base: np.ndarray,
                                         optimizer_samples_rom_extra: np.ndarray,
                                         variational_mean: np.ndarray,
                                         variational_std: np.ndarray,
                                         fom_log_likelihoods: np.ndarray,
                                         rom_log_likelihoods_base: np.ndarray,
                                         rom_log_likelihoods_extra: np.ndarray,
                                         baseline_method: str,
                                         use_mfmc_control_variate: bool,
                                         variational_correlation_cholesky: np.ndarray = None,
                                         elbo_scaling_factor: float = 1.0):
    dimensionality = variational_mean.size
    correlation_inverse = None
    second_mean_template = None
    if variational_correlation_cholesky is not None:
        identity = np.eye(dimensionality)
        correlation_solve = np.linalg.solve(variational_correlation_cholesky, identity)
        correlation_inverse = np.linalg.solve(
            variational_correlation_cholesky.transpose(),
            correlation_solve,
        )
        inverse_std_outer = 1.0 / (variational_std[:, None] * variational_std[None, :])
        second_mean_template = -correlation_inverse * inverse_std_outer

    def _compute_hessian_scores(optimizer_samples: np.ndarray):
        centered = optimizer_samples - variational_mean[None, :]
        normalized = centered / variational_std[None, :]
        if variational_correlation_cholesky is None:
            score_mean = centered / (variational_std[None, :] ** 2)
            score_log_std = normalized ** 2 - 1.0
        else:
            correlation_solve_samples = np.linalg.solve(
                variational_correlation_cholesky,
                normalized.transpose(),
            )
            correlation_inverse_times_normalized = np.linalg.solve(
                variational_correlation_cholesky.transpose(),
                correlation_solve_samples,
            ).transpose()
            score_mean = correlation_inverse_times_normalized / variational_std[None, :]
            score_log_std = normalized * correlation_inverse_times_normalized - 1.0

        score_mean_outer = score_mean[:, :, None] * score_mean[:, None, :]
        score_log_std_outer = score_log_std[:, :, None] * score_log_std[:, None, :]
        score_cross_outer = score_mean[:, :, None] * score_log_std[:, None, :]
        if variational_correlation_cholesky is None:
            second_mean = np.zeros_like(score_mean_outer)
            second_log_std = np.zeros_like(score_log_std_outer)
            second_cross = np.zeros_like(score_cross_outer)
            index = np.arange(dimensionality)
            second_mean[:, index, index] = -1.0 / (variational_std ** 2)
            second_log_std[:, index, index] = -2.0 * (normalized ** 2)
            second_cross[:, index, index] = -2.0 * score_mean
        else:
            sample_count = optimizer_samples.shape[0]
            second_mean = np.broadcast_to(
                second_mean_template,
                (sample_count, dimensionality, dimensionality),
            ).copy()

            outer_normalized = normalized[:, :, None] * normalized[:, None, :]
            second_log_std = -(outer_normalized * correlation_inverse[None, :, :])
            index = np.arange(dimensionality)
            second_log_std[:, index, index] -= (
                normalized * correlation_inverse_times_normalized
            )

            second_cross = -(
                (normalized[:, None, :] * correlation_inverse[None, :, :])
                / variational_std[None, :, None]
            )
            second_cross[:, index, index] -= score_mean

        mean_scores = score_mean_outer + second_mean
        log_std_scores = score_log_std_outer + second_log_std
        cross_scores = score_cross_outer + second_cross
        return mean_scores, log_std_scores, cross_scores

    scaled_fom_log_likelihoods = elbo_scaling_factor * fom_log_likelihoods
    scaled_rom_log_likelihoods_base = elbo_scaling_factor * rom_log_likelihoods_base
    scaled_rom_log_likelihoods_extra = elbo_scaling_factor * rom_log_likelihoods_extra

    high_mean_scores, high_log_std_scores, high_cross_scores = _compute_hessian_scores(
        optimizer_samples_fom,
    )
    low_mean_scores_base, low_log_std_scores_base, low_cross_scores_base = _compute_hessian_scores(
        optimizer_samples_rom_base,
    )
    baseline_method = _normalize_baseline_method(baseline_method)
    if baseline_method == 'optimal':
        dimensionality = variational_mean.size
        baseline_hessian_mean = _compute_optimal_baseline(
            scaled_fom_log_likelihoods,
            high_mean_scores.reshape(high_mean_scores.shape[0], -1),
        ).reshape(dimensionality, dimensionality)
        baseline_hessian_log_std = _compute_optimal_baseline(
            scaled_fom_log_likelihoods,
            high_log_std_scores.reshape(high_log_std_scores.shape[0], -1),
        ).reshape(dimensionality, dimensionality)
        baseline_hessian_cross = _compute_optimal_baseline(
            scaled_fom_log_likelihoods,
            high_cross_scores.reshape(high_cross_scores.shape[0], -1),
        ).reshape(dimensionality, dimensionality)
        centered_fom_weights_mean = (
            scaled_fom_log_likelihoods[:, None, None] - baseline_hessian_mean[None, :, :]
        )
        centered_fom_weights_log_std = (
            scaled_fom_log_likelihoods[:, None, None] - baseline_hessian_log_std[None, :, :]
        )
        centered_fom_weights_cross = (
            scaled_fom_log_likelihoods[:, None, None] - baseline_hessian_cross[None, :, :]
        )
        centered_rom_base_weights_mean = (
            scaled_rom_log_likelihoods_base[:, None, None] - baseline_hessian_mean[None, :, :]
        )
        centered_rom_base_weights_log_std = (
            scaled_rom_log_likelihoods_base[:, None, None] - baseline_hessian_log_std[None, :, :]
        )
        centered_rom_base_weights_cross = (
            scaled_rom_log_likelihoods_base[:, None, None] - baseline_hessian_cross[None, :, :]
        )
        centered_rom_extra_weights_mean = (
            scaled_rom_log_likelihoods_extra[:, None, None] - baseline_hessian_mean[None, :, :]
        )
        centered_rom_extra_weights_log_std = (
            scaled_rom_log_likelihoods_extra[:, None, None] - baseline_hessian_log_std[None, :, :]
        )
        centered_rom_extra_weights_cross = (
            scaled_rom_log_likelihoods_extra[:, None, None] - baseline_hessian_cross[None, :, :]
        )
    elif baseline_method == 'loo':
        centered_fom_weights = (
            scaled_fom_log_likelihoods - _compute_leave_one_out_baseline(scaled_fom_log_likelihoods)
        )
        centered_rom_base_weights = (
            scaled_rom_log_likelihoods_base
            - _compute_leave_one_out_baseline(scaled_rom_log_likelihoods_base)
        )
        centered_rom_extra_weights = (
            scaled_rom_log_likelihoods_extra
            - _compute_leave_one_out_baseline(scaled_rom_log_likelihoods_extra)
        )
        centered_fom_weights_mean = centered_fom_weights[:, None, None]
        centered_fom_weights_log_std = centered_fom_weights[:, None, None]
        centered_fom_weights_cross = centered_fom_weights[:, None, None]
        centered_rom_base_weights_mean = centered_rom_base_weights[:, None, None]
        centered_rom_base_weights_log_std = centered_rom_base_weights[:, None, None]
        centered_rom_base_weights_cross = centered_rom_base_weights[:, None, None]
        centered_rom_extra_weights_mean = centered_rom_extra_weights[:, None, None]
        centered_rom_extra_weights_log_std = centered_rom_extra_weights[:, None, None]
        centered_rom_extra_weights_cross = centered_rom_extra_weights[:, None, None]
    else:
        centered_fom_weights_mean = scaled_fom_log_likelihoods[:, None, None]
        centered_fom_weights_log_std = scaled_fom_log_likelihoods[:, None, None]
        centered_fom_weights_cross = scaled_fom_log_likelihoods[:, None, None]
        centered_rom_base_weights_mean = scaled_rom_log_likelihoods_base[:, None, None]
        centered_rom_base_weights_log_std = scaled_rom_log_likelihoods_base[:, None, None]
        centered_rom_base_weights_cross = scaled_rom_log_likelihoods_base[:, None, None]
        centered_rom_extra_weights_mean = scaled_rom_log_likelihoods_extra[:, None, None]
        centered_rom_extra_weights_log_std = scaled_rom_log_likelihoods_extra[:, None, None]
        centered_rom_extra_weights_cross = scaled_rom_log_likelihoods_extra[:, None, None]

    high_mean_terms = centered_fom_weights_mean * high_mean_scores
    low_mean_terms_base = centered_rom_base_weights_mean * low_mean_scores_base
    high_log_std_terms = centered_fom_weights_log_std * high_log_std_scores
    low_log_std_terms_base = centered_rom_base_weights_log_std * low_log_std_scores_base
    high_cross_terms = centered_fom_weights_cross * high_cross_scores
    low_cross_terms_base = centered_rom_base_weights_cross * low_cross_scores_base
    if optimizer_samples_rom_extra is not None and optimizer_samples_rom_extra.shape[0] > 0:
        low_mean_scores_extra, low_log_std_scores_extra, low_cross_scores_extra = _compute_hessian_scores(
            optimizer_samples_rom_extra,
        )
        low_mean_terms_extra = centered_rom_extra_weights_mean * low_mean_scores_extra
        low_log_std_terms_extra = centered_rom_extra_weights_log_std * low_log_std_scores_extra
        low_cross_terms_extra = centered_rom_extra_weights_cross * low_cross_scores_extra
    else:
        low_mean_terms_extra = None
        low_log_std_terms_extra = None
        low_cross_terms_extra = None

    hessian_diagonal_mean, _ = _mfmc_gradient_estimator(
        high_mean_terms,
        low_mean_terms_base,
        low_mean_terms_extra,
        use_mfmc_control_variate,
    )
    hessian_diagonal_log_std, _ = _mfmc_gradient_estimator(
        high_log_std_terms,
        low_log_std_terms_base,
        low_log_std_terms_extra,
        use_mfmc_control_variate,
    )
    hessian_cross, _ = _mfmc_gradient_estimator(
        high_cross_terms,
        low_cross_terms_base,
        low_cross_terms_extra,
        use_mfmc_control_variate,
    )

    hessian_full = np.block([
        [hessian_diagonal_mean, hessian_cross],
        [hessian_cross.transpose(), hessian_diagonal_log_std],
    ])
    hessian_full = 0.5 * (hessian_full + hessian_full.transpose())
    return np.nan_to_num(hessian_full, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_rom_base_sampling_strategy(rom_base_sampling_strategy: str) -> str:
    strategy = rom_base_sampling_strategy.strip().lower()
    if strategy in ('coupled', 'shared', 'same_as_fom'):
        return 'coupled'
    if strategy in ('separate', 'independent'):
        return 'separate'
    raise ValueError(
        f"Unsupported rom_base_sampling_strategy '{rom_base_sampling_strategy}'. "
        "Supported options are 'coupled' and 'separate'."
    )


def _normalize_line_search_method(line_search_method: str) -> str:
    return _normalize_line_search_method_shared(line_search_method)


def _compute_standard_error(samples: np.ndarray) -> float:
    sample_count = samples.size
    if sample_count <= 1:
        return 0.0
    return float(np.std(samples, ddof=1) / np.sqrt(sample_count))


def _compute_mfmc_log_likelihood_standard_error(fom_log_likelihoods: np.ndarray,
                                                rom_log_likelihoods_base: np.ndarray,
                                                rom_log_likelihoods_extra: np.ndarray) -> float:
    fom_standard_error = _compute_standard_error(fom_log_likelihoods)
    if rom_log_likelihoods_extra.size == 0:
        return fom_standard_error
    low_fidelity_full = np.concatenate([rom_log_likelihoods_base, rom_log_likelihoods_extra])
    low_fidelity_full_standard_error = _compute_standard_error(low_fidelity_full)
    low_fidelity_base_standard_error = _compute_standard_error(rom_log_likelihoods_base)
    return float(np.sqrt(
        fom_standard_error ** 2
        + low_fidelity_full_standard_error ** 2
        + low_fidelity_base_standard_error ** 2
    ))


def _compute_state_elbo_standard_error(state, elbo_scaling_factor: float) -> float:
    mfmc_log_likelihood_standard_error = _compute_mfmc_log_likelihood_standard_error(
        state['log_likelihoods_fom'],
        state['log_likelihoods_rom_base'],
        state['log_likelihoods_rom_only'],
    )
    return abs(elbo_scaling_factor) * mfmc_log_likelihood_standard_error


def _save_mf_vi_restart(restart_path: str,
                        state,
                        variational_mean: np.ndarray,
                        variational_log_std: np.ndarray,
                        variational_distribution: str,
                        variational_correlation_cholesky: np.ndarray,
                        elbo_scaling_factor: float,
                        iteration: int,
                        step_size: float,
                        bounded_parameter_handling: str,
                        optimization_method: str):
    save_data = dict(
        optimizer_samples=state['optimizer_samples'],
        parameter_samples=state['parameter_samples'],
        qois=state['qois_fom'],
        mean_qoi=state['mean_qoi_fom'],
        errors=state['errors_fom'],
        log_likelihoods=state['log_likelihoods_fom'],
        variational_mean=variational_mean,
        variational_log_std=variational_log_std,
        variational_distribution=variational_distribution,
        elbo_scaling_factor=elbo_scaling_factor,
        iteration=iteration,
        step_size=step_size,
        bounded_parameter_handling=bounded_parameter_handling,
        optimization_method=optimization_method,
        rom_error=state['rom_error'],
        mfmc_alpha_mean=state['mfmc_alpha_mean'],
        mfmc_alpha_log_std=state['mfmc_alpha_log_std'],
        training_directories=np.array(state['training_dirs']),
        rom_training_directories=np.array(state['rom_training_dirs']),
        training_parameters=state['training_parameters'],
        training_qois=state['training_qois'],
        rom_training_parameters=state['rom_training_parameters'],
        rom_training_qois=state['rom_training_qois'],
    )
    if variational_correlation_cholesky is not None:
        save_data['variational_correlation_cholesky'] = variational_correlation_cholesky
    np.savez(restart_path, **save_data)


def _evaluate_mf_vi_state(model: QoiModel,
                          rom_model,
                          rom_model_builder: QoiModelBuilderWithTrainingData,
                          observations: np.ndarray,
                          observations_covariance: np.ndarray,
                          iteration_directory: str,
                          parameter_names,
                          variational_mean: np.ndarray,
                          variational_log_std: np.ndarray,
                          fom_sample_size: int,
                          rom_extra_sample_size: int,
                          fom_evaluation_concurrency: int,
                          rom_evaluation_concurrency: int,
                          covariance_regularization: float,
                          baseline_method: str,
                          use_mfmc_control_variate: bool,
                          variational_correlation_cholesky: np.ndarray,
                          elbo_scaling_factor: float,
                          gradient_method: str,
                          bounded_parameter_handling: str,
                          min_variational_std: float,
                          max_variational_std: float,
                          rom_base_sampling_strategy: str,
                          parameter_mins: np.ndarray,
                          parameter_maxes: np.ndarray,
                          transform_interior_margin: float,
                          rom_tolerance: float,
                          max_rom_training_dirs: int,
                          training_dirs: list,
                          training_parameters: np.ndarray,
                          training_qois: np.ndarray,
                          rom_training_dirs: list,
                          rom_training_parameters: np.ndarray,
                          rom_training_qois: np.ndarray):
    run_directory_base = f'{iteration_directory}/run_fom_sample_set_0_'
    optimizer_samples_fom, parameter_samples_fom = _draw_parameter_samples(
        variational_mean,
        variational_log_std,
        fom_sample_size,
        min_variational_std,
        max_variational_std,
        bounded_parameter_handling,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin=transform_interior_margin,
        variational_correlation_cholesky=variational_correlation_cholesky,
    )

    if rom_base_sampling_strategy == 'coupled':
        optimizer_samples_rom_base = optimizer_samples_fom.copy()
        parameter_samples_rom_base = parameter_samples_fom.copy()
    else:
        run_directory_base = f'{iteration_directory}/run_rom_sample_set_0_'
        optimizer_samples_rom_base, parameter_samples_rom_base = _draw_parameter_samples(
            variational_mean,
            variational_log_std,
            fom_sample_size,
            min_variational_std,
            max_variational_std,
            bounded_parameter_handling,
            parameter_mins,
            parameter_maxes,
            transform_interior_margin=transform_interior_margin,
            variational_correlation_cholesky=variational_correlation_cholesky,
        )

    if rom_extra_sample_size > 0:
        optimizer_samples_rom_extra, parameter_samples_rom_extra = _draw_parameter_samples(
            variational_mean,
            variational_log_std,
            rom_extra_sample_size,
            min_variational_std,
            max_variational_std,
            bounded_parameter_handling,
            parameter_mins,
            parameter_maxes,
            transform_interior_margin=transform_interior_margin,
            variational_correlation_cholesky=variational_correlation_cholesky,
        )
    else:
        optimizer_samples_rom_extra = np.zeros((0, variational_mean.size))
        parameter_samples_rom_extra = np.zeros((0, variational_mean.size))

    run_directory_base = f'{iteration_directory}/run_fom_sample_set_0_'
    fom_results = run_eki_iteration(
        model,
        observations,
        run_directory_base,
        parameter_names,
        parameter_samples_fom,
        fom_evaluation_concurrency,
    )
    iteration_training_dirs, iteration_training_parameters, iteration_training_qois = _build_iteration_training_data(
        run_directory_base,
        parameter_samples_fom,
        fom_results,
    )
    candidate_training_dirs, candidate_training_parameters, candidate_training_qois = _append_training_data(
        training_dirs,
        training_parameters,
        training_qois,
        iteration_training_dirs,
        iteration_training_parameters,
        iteration_training_qois,
    )

    rom_model_candidate = rom_model
    candidate_rom_training_dirs = copy.deepcopy(rom_training_dirs)
    candidate_rom_training_parameters = (
        None if rom_training_parameters is None else rom_training_parameters.copy()
    )
    candidate_rom_training_qois = None if rom_training_qois is None else rom_training_qois.copy()

    if rom_model_candidate is None:
        candidate_rom_training_dirs = candidate_training_dirs[-max_rom_training_dirs:]
        candidate_rom_training_parameters = candidate_training_parameters[-max_rom_training_dirs:]
        candidate_rom_training_qois = candidate_training_qois[-max_rom_training_dirs:]
        rom_model_candidate = rom_model_builder.build_from_training_dirs(
            iteration_directory,
            candidate_rom_training_dirs,
            candidate_rom_training_parameters,
            candidate_rom_training_qois,
        )

    run_directory_base = f'{iteration_directory}/run_rom_sample_set_0_'
    rom_results_base = run_eki_iteration(
        rom_model_candidate,
        observations,
        run_directory_base,
        parameter_names,
        parameter_samples_rom_base,
        rom_evaluation_concurrency,
    )
    rom_error = _compute_rom_relative_error(
        rom_results_base['mean-qoi'][:, None],
        fom_results['mean-qoi'][:, None],
    )
    rom_rebuilt_this_iteration = False

    if rom_error >= rom_tolerance:
        candidate_rom_training_dirs = candidate_training_dirs[-max_rom_training_dirs:]
        candidate_rom_training_parameters = candidate_training_parameters[-max_rom_training_dirs:]
        candidate_rom_training_qois = candidate_training_qois[-max_rom_training_dirs:]
        rom_model_candidate = rom_model_builder.build_from_training_dirs(
            iteration_directory,
            candidate_rom_training_dirs,
            candidate_rom_training_parameters,
            candidate_rom_training_qois,
        )
        rom_rebuilt_this_iteration = True
        rom_results_base = run_eki_iteration(
            rom_model_candidate,
            observations,
            run_directory_base,
            parameter_names,
            parameter_samples_rom_base,
            rom_evaluation_concurrency,
        )
        rom_error = _compute_rom_relative_error(
            rom_results_base['mean-qoi'][:, None],
            fom_results['mean-qoi'][:, None],
        )

    if rom_extra_sample_size > 0:
        run_directory_base = f'{iteration_directory}/run_rom_sample_set_1_'
        rom_results_extra = run_eki_iteration(
            rom_model_candidate,
            observations,
            run_directory_base,
            parameter_names,
            parameter_samples_rom_extra,
            rom_evaluation_concurrency,
        )
    else:
        rom_results_extra = {
            'qois': np.zeros((fom_results['qois'].shape[0], 0)),
            'mean-qoi': fom_results['mean-qoi'],
            'errors': np.zeros((fom_results['errors'].shape[0], 0)),
        }

    log_likelihood_precision_operator = _compute_log_likelihood_precision_operator(
        observations_covariance,
        covariance_regularization,
    )

    fom_log_likelihoods, fom_misfits = _compute_log_likelihoods(
        fom_results['errors'],
        observations_covariance,
        covariance_regularization,
        precision_operator=log_likelihood_precision_operator,
    )
    rom_log_likelihoods_base, _ = _compute_log_likelihoods(
        rom_results_base['errors'],
        observations_covariance,
        covariance_regularization,
        precision_operator=log_likelihood_precision_operator,
    )
    if rom_extra_sample_size > 0:
        rom_log_likelihoods_extra, _ = _compute_log_likelihoods(
            rom_results_extra['errors'],
            observations_covariance,
            covariance_regularization,
            precision_operator=log_likelihood_precision_operator,
        )
    else:
        rom_log_likelihoods_extra = np.zeros(0)

    variational_std, variational_log_std = _compute_variational_std(
        variational_log_std,
        min_variational_std,
        max_variational_std,
    )
    gradient_mean, gradient_log_std, baseline_mean, baseline_log_std, alpha_mean, alpha_log_std = (
        _compute_mfmc_reinforce_gradients(
            optimizer_samples_fom,
            optimizer_samples_rom_base,
            optimizer_samples_rom_extra,
            variational_mean,
            variational_std,
            fom_log_likelihoods,
            rom_log_likelihoods_base,
            rom_log_likelihoods_extra,
            baseline_method,
            use_mfmc_control_variate,
            variational_correlation_cholesky,
            elbo_scaling_factor,
        )
    )
    hessian_diagonal_mean, hessian_diagonal_log_std = _compute_mfmc_reinforce_hessian_diagonal(
        optimizer_samples_fom,
        optimizer_samples_rom_base,
        optimizer_samples_rom_extra,
        variational_mean,
        variational_std,
        fom_log_likelihoods,
        rom_log_likelihoods_base,
        rom_log_likelihoods_extra,
        baseline_method,
        use_mfmc_control_variate,
        variational_correlation_cholesky,
        elbo_scaling_factor,
    )
    hessian_full = _compute_mfmc_reinforce_hessian_full(
        optimizer_samples_fom,
        optimizer_samples_rom_base,
        optimizer_samples_rom_extra,
        variational_mean,
        variational_std,
        fom_log_likelihoods,
        rom_log_likelihoods_base,
        rom_log_likelihoods_extra,
        baseline_method,
        use_mfmc_control_variate,
        variational_correlation_cholesky,
        elbo_scaling_factor,
    )
    update_direction_mean, update_direction_log_std, normalized_gradient_method = _compute_update_directions(
        gradient_mean,
        gradient_log_std,
        variational_std,
        gradient_method,
    )

    dimensionality = variational_mean.size
    entropy = np.sum(variational_log_std) + 0.5 * dimensionality * (1.0 + np.log(2.0 * np.pi))
    if variational_correlation_cholesky is not None:
        entropy += np.sum(np.log(np.diag(variational_correlation_cholesky)))
    entropy *= elbo_scaling_factor
    if rom_extra_sample_size > 0:
        low_fidelity_full_log_likelihoods = np.concatenate(
            [rom_log_likelihoods_base, rom_log_likelihoods_extra]
        )
        mfmc_mean_log_likelihood = (
            np.mean(fom_log_likelihoods)
            + np.mean(low_fidelity_full_log_likelihoods)
            - np.mean(rom_log_likelihoods_base)
        )
    else:
        mfmc_mean_log_likelihood = np.mean(fom_log_likelihoods)
    elbo = elbo_scaling_factor * mfmc_mean_log_likelihood + entropy

    optimizer_samples = np.vstack([
        optimizer_samples_fom,
        optimizer_samples_rom_base,
        optimizer_samples_rom_extra,
    ])
    parameter_samples = np.vstack([
        parameter_samples_fom,
        parameter_samples_rom_base,
        parameter_samples_rom_extra,
    ])
    state = {
        'optimizer_samples': optimizer_samples,
        'parameter_samples': parameter_samples,
        'parameter_samples_fom': parameter_samples_fom,
        'parameter_samples_rom_base': parameter_samples_rom_base,
        'parameter_samples_rom_only': parameter_samples_rom_extra,
        'qois_fom': fom_results['qois'],
        'mean_qoi_fom': fom_results['mean-qoi'],
        'errors_fom': fom_results['errors'],
        'qois_rom_base': rom_results_base['qois'],
        'qois_rom_coupled': rom_results_base['qois'],
        'qois_rom_only': rom_results_extra['qois'],
        'log_likelihoods_fom': fom_log_likelihoods,
        'log_likelihoods_rom_base': rom_log_likelihoods_base,
        'log_likelihoods_rom_coupled': rom_log_likelihoods_base,
        'log_likelihoods_rom_only': rom_log_likelihoods_extra,
        'mean_misfit': np.mean(fom_misfits),
        'mean_relative_mse': np.mean(_compute_relative_mse(fom_results['errors'], observations)),
        'entropy': entropy,
        'elbo': elbo,
        'gradient_mean': gradient_mean,
        'gradient_log_std': gradient_log_std,
        'hessian_diagonal_mean': hessian_diagonal_mean,
        'hessian_diagonal_log_std': hessian_diagonal_log_std,
        'hessian_full': hessian_full,
        'update_direction_mean': update_direction_mean,
        'update_direction_log_std': update_direction_log_std,
        'gradient_method': normalized_gradient_method,
        'baseline_mean': baseline_mean,
        'baseline_log_std': baseline_log_std,
        'mfmc_alpha_mean': alpha_mean,
        'mfmc_alpha_log_std': alpha_log_std,
        'rom_error': rom_error,
        'rom_rebuilt_this_iteration': rom_rebuilt_this_iteration,
        'rom_model': rom_model_candidate,
        'training_dirs': candidate_training_dirs,
        'training_parameters': candidate_training_parameters,
        'training_qois': candidate_training_qois,
        'rom_training_dirs': candidate_rom_training_dirs,
        'rom_training_parameters': candidate_rom_training_parameters,
        'rom_training_qois': candidate_rom_training_qois,
    }
    return state


def _validate_run_mf_vi_inputs(restart_file: str,
                               absolute_vi_directory: str,
                               fom_sample_size: int,
                               rom_extra_sample_size: int,
                               max_step_size: float,
                               step_size_growth_factor: float,
                               step_size_decay_factor: float,
                               line_search_sample_growth_factor: float,
                               relaxation_parameter: float,
                               line_search_method: str,
                               line_search_nonmonotone_window: int,
                               line_search_armijo_coefficient: float,
                               line_search_uncertainty_sigma: float,
                               min_variational_std: float,
                               max_variational_std: float,
                               max_log_std_update: float,
                               newton_regularization: float,
                               covariance_regularization: float,
                               observations_covariance: np.ndarray,
                               observations: np.ndarray,
                               elbo_scaling_factor: float,
                               max_rom_training_history: int,
                               parameter_space: ParameterSpace,
                               parameter_mins: np.ndarray,
                               parameter_maxes: np.ndarray,
                               transform_interior_margin: float,
                               min_physical_variational_std_fraction: float,
                               bounded_parameter_handling: str) -> None:
    assert restart_file is None, "run_mf_vi currently does not support restart_file."
    assert os.path.isabs(absolute_vi_directory), (
        f"absolute_vi_directory is not an absolute path ({absolute_vi_directory})"
    )
    assert fom_sample_size > 1, "fom_sample_size must be greater than 1"
    assert rom_extra_sample_size >= 0, "rom_extra_sample_size must be non-negative"
    assert max_step_size > 0.0, "max_step_size must be positive"
    assert step_size_growth_factor >= 1.0, "step_size_growth_factor must be greater than 1.0"
    assert step_size_decay_factor >= 1.0, "step_size_decay_factor must be greater than 1.0"
    assert line_search_sample_growth_factor >= 1.0, (
        "line_search_sample_growth_factor must be greater than or equal to 1.0"
    )
    assert relaxation_parameter >= 1.0, "relaxation_parameter must be >= 1.0"
    if line_search_method == 'stochastic_nonmonotone':
        assert line_search_nonmonotone_window >= 1, "line_search_nonmonotone_window must be >= 1"
        assert 0.0 <= line_search_armijo_coefficient <= 1.0, (
            "line_search_armijo_coefficient must be in [0,1]"
        )
        assert line_search_uncertainty_sigma >= 0.0, "line_search_uncertainty_sigma must be non-negative"
    assert min_variational_std > 0.0, "min_variational_std must be positive"
    assert max_variational_std > min_variational_std, (
        "max_variational_std must be greater than min_variational_std"
    )
    assert min_physical_variational_std_fraction >= 0.0, (
        "min_physical_variational_std_fraction must be non-negative"
    )
    assert max_log_std_update > 0.0, "max_log_std_update must be positive"
    assert newton_regularization > 0.0, "newton_regularization must be positive"
    assert covariance_regularization >= 0.0, "covariance_regularization must be non-negative"
    assert observations_covariance.shape[0] == observations_covariance.shape[1], (
        "observations_covariance must be square"
    )
    assert observations_covariance.shape[0] == observations.size, (
        "observations_covariance shape must match observations size"
    )
    assert elbo_scaling_factor > 0.0, "elbo_scaling_factor must be positive"
    assert max_rom_training_history >= 1, "max_rom_training_history must be >= 1"

    parameter_dimensionality = parameter_space.get_dimensionality()
    if parameter_mins is not None:
        assert np.size(parameter_mins) == parameter_dimensionality, (
            f"parameter_mins of size {np.size(parameter_mins)} is inconsistent with "
            f"the parameter_space of size {parameter_dimensionality}"
        )
    if parameter_maxes is not None:
        assert np.size(parameter_maxes) == parameter_dimensionality, (
            f"parameter_maxes of size {np.size(parameter_maxes)} is inconsistent with "
            f"the parameter_space of size {parameter_dimensionality}"
        )
    if bounded_parameter_handling == 'transform':
        assert 0.0 <= transform_interior_margin < 0.5, (
            "transform_interior_margin must be in [0.0, 0.5)"
        )
        assert parameter_mins is not None, "parameter_mins must be provided for bounded_parameter_handling='transform'"
        assert parameter_maxes is not None, "parameter_maxes must be provided for bounded_parameter_handling='transform'"
        assert np.size(parameter_mins) == parameter_dimensionality, (
            f"parameter_mins of size {np.size(parameter_mins)} is inconsistent with "
            f"the parameter_space of size {parameter_dimensionality}"
        )
        assert np.size(parameter_maxes) == parameter_dimensionality, (
            f"parameter_maxes of size {np.size(parameter_maxes)} is inconsistent with "
            f"the parameter_space of size {parameter_dimensionality}"
        )
        assert np.all(parameter_maxes > parameter_mins), (
            "All parameter_maxes entries must be greater than parameter_mins entries "
            "for bounded_parameter_handling='transform'"
        )


def run_mf_vi(model: QoiModel,
              rom_model_builder: QoiModelBuilderWithTrainingData,
              parameter_space: ParameterSpace,
              observations: np.ndarray,
              observations_covariance: np.ndarray,
              parameter_mins: np.ndarray = None,
              parameter_maxes: np.ndarray = None,
              restart_file: str = None,
              optimizer_method: str = 'gradient',
              optimizer_config=None,
              line_search_method: str = 'stochastic_nonmonotone',
              line_search_config=None,
              absolute_vi_directory: str = os.getcwd() + "/work/",
              fom_sample_size: int = 10,
              rom_extra_sample_size: int = 30,
              rom_tolerance: float = 0.005,
              max_rom_training_history: int = 1,
              random_seed: int = 1,
              fom_evaluation_concurrency=1,
              rom_evaluation_concurrency=1,
              covariance_regularization: float = 1e-7,
              elbo_scaling_factor='diag_mean',
              baseline_method: str = None,
              variational_distribution: str = 'diagonal',
              use_mfmc_control_variate: bool = True,
              rom_base_sampling_strategy: str = 'coupled',
              bounded_parameter_handling: str = 'transform',
              transform_interior_margin: float = 1e-8,
              min_physical_variational_std_fraction: float = 1e-8,
              **legacy_kwargs):
    """
    Run multi-fidelity VI with MFMC variance-reduced score-function gradients.

    Args:
        parameter_mins: Optional lower bounds on parameters.
        parameter_maxes: Optional upper bounds on parameters.
        restart_file: Optional restart file path.
        optimizer_method: Optimizer used for variational updates. Supported
            options are 'gradient' and 'newton'.
        optimizer_config: Method-specific optimizer config. Expected types are
            `VIGradientOptimizerConfig` for optimizer_method='gradient',
            `VINewtonOptimizerConfig` for optimizer_method='newton'.
        line_search_method: Line-search acceptance strategy. Supported options
            are 'legacy' and 'stochastic_nonmonotone'. Defaults to
            'stochastic_nonmonotone'.
        line_search_config: Method-specific line-search config. Expected types
            are `VILegacyLineSearchConfig` for line_search_method='legacy' and
            `VIStochasticNonmonotoneLineSearchConfig` for
            line_search_method='stochastic_nonmonotone'.
        rom_base_sampling_strategy: Strategy for ROM-base sample selection.
            'coupled' (default) reuses the FOM sample set for ROM-base
            evaluations. 'separate' draws an independent ROM-base sample set.
        transform_interior_margin: Margin used by bounded_parameter_handling='transform'
            to keep mapped samples away from exact bounds.
        min_physical_variational_std_fraction: Minimum physical-space
            variational standard deviation as a fraction of each parameter range
            when bounded_parameter_handling='transform'.
        legacy_kwargs: Optional scalar overrides for fields in
            `optimizer_config` and `line_search_config`.

    Returns:
        Tuple of (variational_mean, variational_std, fom_parameter_samples, fom_qois).
    """
    start_time = time.time()
    parameter_mins, parameter_maxes = _resolve_parameter_bounds(
        parameter_mins,
        parameter_maxes,
        legacy_kwargs,
    )
    restart_file = _resolve_restart_file(restart_file, legacy_kwargs)
    optimization_method, resolved_optimizer_config = _resolve_optimizer_config(
        optimizer_method,
        optimizer_config,
        legacy_kwargs,
        VIGradientOptimizerConfig(),
        VINewtonOptimizerConfig(newton_regularization=1e-8),
    )
    line_search_method, resolved_line_search_config = _resolve_line_search_config(
        line_search_method,
        line_search_config,
        legacy_kwargs,
        VILegacyLineSearchConfig(
            step_size_growth_factor=1.05,
            relaxation_parameter=10.0,
        ),
        VIStochasticNonmonotoneLineSearchConfig(
            step_size_growth_factor=1.05,
            relaxation_parameter=10.0,
        ),
    )
    _raise_for_unused_legacy_kwargs(legacy_kwargs, 'run_mf_vi')

    gradient_method = 'standard'
    if optimization_method == 'gradient':
        gradient_method = resolved_optimizer_config.gradient_method
    gradient_norm_tolerance = resolved_optimizer_config.gradient_norm_tolerance
    max_iterations = resolved_optimizer_config.max_iterations
    max_log_std_update = resolved_optimizer_config.max_log_std_update
    min_variational_std = resolved_optimizer_config.min_variational_std
    max_variational_std = resolved_optimizer_config.max_variational_std

    newton_defaults = VINewtonOptimizerConfig(newton_regularization=1e-8)
    newton_metric = _normalize_newton_metric(newton_defaults.newton_metric)
    newton_regularization = newton_defaults.newton_regularization
    if optimization_method == 'newton':
        newton_metric = _normalize_newton_metric(resolved_optimizer_config.newton_metric)
        newton_regularization = resolved_optimizer_config.newton_regularization

    initial_step_size = resolved_line_search_config.initial_step_size
    max_step_size = resolved_line_search_config.max_step_size
    step_size_growth_factor = resolved_line_search_config.step_size_growth_factor
    step_size_decay_factor = resolved_line_search_config.step_size_decay_factor
    max_step_size_decrease_trys = resolved_line_search_config.max_step_size_decrease_trys
    relaxation_parameter = resolved_line_search_config.relaxation_parameter
    line_search_sample_growth_factor = resolved_line_search_config.line_search_sample_growth_factor
    line_search_nonmonotone_window = 1
    line_search_armijo_coefficient = 0.0
    line_search_uncertainty_sigma = 0.0
    if line_search_method == 'stochastic_nonmonotone':
        line_search_nonmonotone_window = resolved_line_search_config.line_search_nonmonotone_window
        line_search_armijo_coefficient = resolved_line_search_config.line_search_armijo_coefficient
        line_search_uncertainty_sigma = resolved_line_search_config.line_search_uncertainty_sigma

    max_rom_training_dirs = int(max_rom_training_history * (fom_sample_size + 1))

    elbo_scaling_factor = _resolve_elbo_scaling_factor(
        elbo_scaling_factor,
        observations_covariance,
    )

    bounded_parameter_handling = _normalize_bounded_parameter_handling(bounded_parameter_handling)
    variational_distribution = _normalize_variational_distribution(variational_distribution)
    if baseline_method is None:
        baseline_method = 'loo'
    baseline_method = _normalize_baseline_method(baseline_method)
    rom_base_sampling_strategy = _normalize_rom_base_sampling_strategy(rom_base_sampling_strategy)
    if bounded_parameter_handling == 'transform':
        if parameter_mins is None and hasattr(parameter_space, 'parameter_mins'):
            parameter_mins = np.asarray(parameter_space.parameter_mins)
        if parameter_maxes is None and hasattr(parameter_space, 'parameter_maxes'):
            parameter_maxes = np.asarray(parameter_space.parameter_maxes)
    _validate_run_mf_vi_inputs(
        restart_file=restart_file,
        absolute_vi_directory=absolute_vi_directory,
        fom_sample_size=fom_sample_size,
        rom_extra_sample_size=rom_extra_sample_size,
        max_step_size=max_step_size,
        step_size_growth_factor=step_size_growth_factor,
        step_size_decay_factor=step_size_decay_factor,
        line_search_sample_growth_factor=line_search_sample_growth_factor,
        relaxation_parameter=relaxation_parameter,
        line_search_method=line_search_method,
        line_search_nonmonotone_window=line_search_nonmonotone_window,
        line_search_armijo_coefficient=line_search_armijo_coefficient,
        line_search_uncertainty_sigma=line_search_uncertainty_sigma,
        min_variational_std=min_variational_std,
        max_variational_std=max_variational_std,
        max_log_std_update=max_log_std_update,
        newton_regularization=newton_regularization,
        covariance_regularization=covariance_regularization,
        observations_covariance=observations_covariance,
        observations=observations,
        elbo_scaling_factor=elbo_scaling_factor,
        max_rom_training_history=max_rom_training_history,
        parameter_space=parameter_space,
        parameter_mins=parameter_mins,
        parameter_maxes=parameter_maxes,
        transform_interior_margin=transform_interior_margin,
        min_physical_variational_std_fraction=min_physical_variational_std_fraction,
        bounded_parameter_handling=bounded_parameter_handling,
    )

    np.random.seed(random_seed)
    parameter_names = parameter_space.get_names()
    iteration = 0
    step_size = min(initial_step_size, max_step_size)

    total_initial_samples = fom_sample_size + max(rom_extra_sample_size, 1)
    initial_samples = parameter_space.generate_samples(total_initial_samples)
    initial_samples = bound_samples(initial_samples, parameter_mins, parameter_maxes)
    if bounded_parameter_handling == 'transform':
        optimizer_initial_samples = _transform_parameter_to_optimizer(
            initial_samples,
            parameter_mins,
            parameter_maxes,
            transform_interior_margin,
        )
    else:
        optimizer_initial_samples = initial_samples

    variational_mean = np.mean(optimizer_initial_samples, axis=0)
    variational_std = np.std(optimizer_initial_samples, axis=0, ddof=1)
    variational_std = np.maximum(variational_std, min_variational_std)
    variational_log_std = np.log(variational_std)
    variational_log_std = _clip_variational_log_std(
        variational_log_std,
        min_variational_std,
        max_variational_std,
    )
    variational_log_std = _enforce_variational_log_std_bounds(
        variational_mean,
        variational_log_std,
        min_variational_std,
        max_variational_std,
        bounded_parameter_handling,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin,
        min_physical_variational_std_fraction,
    )
    variational_correlation_cholesky = None
    if variational_distribution == 'multivariate':
        variational_correlation_cholesky = _compute_correlation_cholesky_from_samples(
            optimizer_initial_samples
        )

    state = _evaluate_mf_vi_state(
        model=model,
        rom_model=None,
        rom_model_builder=rom_model_builder,
        observations=observations,
        observations_covariance=observations_covariance,
        iteration_directory=f'{absolute_vi_directory}/iteration_{iteration}',
        parameter_names=parameter_names,
        variational_mean=variational_mean,
        variational_log_std=variational_log_std,
        fom_sample_size=fom_sample_size,
        rom_extra_sample_size=rom_extra_sample_size,
        fom_evaluation_concurrency=fom_evaluation_concurrency,
        rom_evaluation_concurrency=rom_evaluation_concurrency,
        covariance_regularization=covariance_regularization,
        baseline_method=baseline_method,
        use_mfmc_control_variate=use_mfmc_control_variate,
        variational_correlation_cholesky=variational_correlation_cholesky,
        elbo_scaling_factor=elbo_scaling_factor,
        gradient_method=gradient_method,
        bounded_parameter_handling=bounded_parameter_handling,
        min_variational_std=min_variational_std,
        max_variational_std=max_variational_std,
        rom_base_sampling_strategy=rom_base_sampling_strategy,
        parameter_mins=parameter_mins,
        parameter_maxes=parameter_maxes,
        transform_interior_margin=transform_interior_margin,
        rom_tolerance=rom_tolerance,
        max_rom_training_dirs=max_rom_training_dirs,
        training_dirs=[],
        training_parameters=None,
        training_qois=None,
        rom_training_dirs=[],
        rom_training_parameters=None,
        rom_training_qois=None,
    )

    _save_mf_vi_restart(
        f'{absolute_vi_directory}/iteration_{iteration}/restart.npz',
        state,
        variational_mean,
        variational_log_std,
        variational_distribution,
        variational_correlation_cholesky,
        elbo_scaling_factor,
        iteration,
        step_size,
        bounded_parameter_handling,
        optimization_method,
    )

    gradient_norm = _compute_gradient_norm(state, optimization_method)
    wall_time = time.time() - start_time
    alpha_mean_scalar = float(np.mean(state['mfmc_alpha_mean']))
    alpha_log_scalar = float(np.mean(state['mfmc_alpha_log_std']))
    print(
        f'Iteration: {iteration}, Relative MSE: {state["mean_relative_mse"]:.5f}, '
        f'ELBO: {state["elbo"]:.5f}, ROM err: {state["rom_error"]:.5f}, '
        f'alpha_mean: {alpha_mean_scalar:.5f}, alpha_logstd: {alpha_log_scalar:.5f}, '
        f'Step size: {step_size:.5f}, Gradient norm: {gradient_norm:.5f}, Wall time: {wall_time:.5f}'
    )
    _print_vi_parameters(
        variational_mean,
        variational_log_std,
        min_variational_std,
        max_variational_std,
        optimizer_samples=state['optimizer_samples'],
        parameter_samples=state['parameter_samples'],
    )

    iteration += 1
    step_failed_counter = 0
    steepest_descent_solver = SteepestDescentSolver()
    accepted_elbo_history = [state['elbo']]

    while iteration < max_iterations and gradient_norm > gradient_norm_tolerance:
        current_fom_sample_size = int(np.ceil(
            fom_sample_size * (line_search_sample_growth_factor ** step_failed_counter)
        ))
        current_fom_sample_size = max(current_fom_sample_size, 2)
        if rom_extra_sample_size > 0:
            current_rom_extra_sample_size = int(np.ceil(
                rom_extra_sample_size * (line_search_sample_growth_factor ** step_failed_counter)
            ))
            current_rom_extra_sample_size = max(current_rom_extra_sample_size, 1)
        else:
            current_rom_extra_sample_size = 0

        newton_metric_scale = None
        if optimization_method == 'newton':
            variational_std_for_metric, _ = _compute_variational_std(
                variational_log_std,
                min_variational_std,
                max_variational_std,
            )
            newton_metric_scale = _compute_newton_metric_scale(
                newton_metric,
                variational_std_for_metric,
            )

        line_search_predicted_slope = 0.0
        if optimization_method == 'gradient':
            gradient = np.concatenate([state['update_direction_mean'], state['update_direction_log_std']])
            step = steepest_descent_solver.step(gradient)
            dimensionality = state['update_direction_mean'].size
            direction_mean = step[:dimensionality]
            direction_log_std = step[dimensionality:]
            line_search_predicted_slope = float(
                np.dot(state['gradient_mean'], direction_mean)
                + np.dot(state['gradient_log_std'], direction_log_std)
            )
            test_variational_mean = variational_mean + step_size * direction_mean
            log_std_update = np.clip(
                step_size * direction_log_std,
                -max_log_std_update,
                max_log_std_update,
            )
            test_variational_log_std = variational_log_std + log_std_update
            test_variational_log_std = _clip_variational_log_std(
                test_variational_log_std,
                min_variational_std,
                max_variational_std,
            )
            test_variational_log_std = _enforce_variational_log_std_bounds(
                test_variational_mean,
                test_variational_log_std,
                min_variational_std,
                max_variational_std,
                bounded_parameter_handling,
                parameter_mins,
                parameter_maxes,
                transform_interior_margin,
                min_physical_variational_std_fraction,
            )
        else:
            direction_mean, direction_log_std = _compute_newton_step(
                state,
                newton_regularization,
                metric_scale=newton_metric_scale,
            )
            line_search_predicted_slope = float(
                np.dot(state['gradient_mean'], direction_mean)
                + np.dot(state['gradient_log_std'], direction_log_std)
            )
            test_variational_mean = variational_mean + step_size * direction_mean
            log_std_update = np.clip(
                step_size * direction_log_std,
                -max_log_std_update,
                max_log_std_update,
            )
            test_variational_log_std = variational_log_std + log_std_update
            test_variational_log_std = _clip_variational_log_std(
                test_variational_log_std,
                min_variational_std,
                max_variational_std,
            )
            test_variational_log_std = _enforce_variational_log_std_bounds(
                test_variational_mean,
                test_variational_log_std,
                min_variational_std,
                max_variational_std,
                bounded_parameter_handling,
                parameter_mins,
                parameter_maxes,
                transform_interior_margin,
                min_physical_variational_std_fraction,
            )

        test_state = _evaluate_mf_vi_state(
            model=model,
            rom_model=state['rom_model'],
            rom_model_builder=rom_model_builder,
            observations=observations,
            observations_covariance=observations_covariance,
            iteration_directory=f'{absolute_vi_directory}/iteration_{iteration}',
            parameter_names=parameter_names,
            variational_mean=test_variational_mean,
            variational_log_std=test_variational_log_std,
            fom_sample_size=current_fom_sample_size,
            rom_extra_sample_size=current_rom_extra_sample_size,
            fom_evaluation_concurrency=fom_evaluation_concurrency,
            rom_evaluation_concurrency=rom_evaluation_concurrency,
            covariance_regularization=covariance_regularization,
            baseline_method=baseline_method,
            use_mfmc_control_variate=use_mfmc_control_variate,
            variational_correlation_cholesky=variational_correlation_cholesky,
            elbo_scaling_factor=elbo_scaling_factor,
            gradient_method=gradient_method,
            bounded_parameter_handling=bounded_parameter_handling,
            min_variational_std=min_variational_std,
            max_variational_std=max_variational_std,
            rom_base_sampling_strategy=rom_base_sampling_strategy,
            parameter_mins=parameter_mins,
            parameter_maxes=parameter_maxes,
            transform_interior_margin=transform_interior_margin,
            rom_tolerance=rom_tolerance,
            max_rom_training_dirs=max_rom_training_dirs,
            training_dirs=state['training_dirs'],
            training_parameters=state['training_parameters'],
            training_qois=state['training_qois'],
            rom_training_dirs=state['rom_training_dirs'],
            rom_training_parameters=state['rom_training_parameters'],
            rom_training_qois=state['rom_training_qois'],
        )

        if line_search_method == 'legacy':
            allowable_elbo_drop = (relaxation_parameter - 1.0) * abs(state['elbo'])
            accept_step = test_state['elbo'] >= state['elbo'] - allowable_elbo_drop
        else:
            reference_elbo = max(accepted_elbo_history[-line_search_nonmonotone_window:])
            nonnegative_predicted_slope = max(line_search_predicted_slope, 0.0)
            armijo_target = (
                reference_elbo
                + line_search_armijo_coefficient * step_size * nonnegative_predicted_slope
            )
            state_standard_error = _compute_state_elbo_standard_error(state, elbo_scaling_factor)
            test_standard_error = _compute_state_elbo_standard_error(test_state, elbo_scaling_factor)
            delta_standard_error = np.sqrt(
                state_standard_error ** 2 + test_standard_error ** 2
            )
            armijo_target -= line_search_uncertainty_sigma * delta_standard_error
            accept_step = test_state['elbo'] >= armijo_target

        if accept_step:
            step_failed_counter = 0
            variational_mean = test_variational_mean*1.0
            variational_log_std = test_variational_log_std*1.0
            state = test_state
            accepted_elbo_history.append(state['elbo'])

            step_size = min(step_size * step_size_growth_factor, max_step_size)

            gradient_norm = _compute_gradient_norm(state, optimization_method)
            wall_time = time.time() - start_time
            alpha_mean_scalar = float(np.mean(state['mfmc_alpha_mean']))
            alpha_log_scalar = float(np.mean(state['mfmc_alpha_log_std']))
            print(
                f'Iteration: {iteration}, Relative MSE: {state["mean_relative_mse"]:.5f}, ELBO: {state["elbo"]:.5f}, '
                f'ROM err: {state["rom_error"]:.5f}, alpha_mean: {alpha_mean_scalar:.5f}, '
                f'alpha_logstd: {alpha_log_scalar:.5f}, Step size: {step_size:.5e}, '
                f'Gradient norm: {gradient_norm:.5f}, Wall time: {wall_time:.5f}'
            )
            _print_vi_parameters(
                variational_mean,
                variational_log_std,
                min_variational_std,
                max_variational_std,
                optimizer_samples=state['optimizer_samples'],
                parameter_samples=state['parameter_samples'],
            )
            _save_mf_vi_restart(
                f'{absolute_vi_directory}/iteration_{iteration}/restart.npz',
                state,
                variational_mean,
                variational_log_std,
                variational_distribution,
                variational_correlation_cholesky,
                elbo_scaling_factor,
                iteration,
                step_size,
                bounded_parameter_handling,
                optimization_method,
            )
            iteration += 1
        else:
            step_failed_counter += 1
            step_size /= step_size_decay_factor
            print(
                f'  Warning, lowering step size, Iteration: {iteration}, '
                f'Relative MSE: {state["mean_relative_mse"]:.5f}, '
                f'ELBO: {state["elbo"]:.5f}, Step size: {step_size:.5e}, Gradient norm: {gradient_norm:.5f}'
            )
            _print_vi_parameters(
                variational_mean,
                variational_log_std,
                min_variational_std,
                max_variational_std,
                optimizer_samples=state['optimizer_samples'],
                parameter_samples=state['parameter_samples'],
            )
            if step_failed_counter > max_step_size_decrease_trys:
                print(f'  Failed to advance after {max_step_size_decrease_trys}, exiting')
                break

    if iteration >= max_iterations:
        print('Max iterations reached, terminating')
    elif gradient_norm <= gradient_norm_tolerance:
        print('Gradient norm dropped below tolerance!')

    variational_std, _ = _compute_variational_std(
        variational_log_std,
        min_variational_std,
        max_variational_std,
    )
    return variational_mean, variational_std, state['parameter_samples_fom'], state['qois_fom']


def mf_vi_with_auto_rom(model: QoiModel,
                        parameter_space: ParameterSpace,
                        observations: np.ndarray,
                        observations_covariance: np.ndarray,
                        parameter_mins: np.ndarray = None,
                        parameter_maxes: np.ndarray = None,
                        restart_file: str = None,
                        optimizer_method: str = 'gradient',
                        optimizer_config=None,
                        line_search_method: str = 'stochastic_nonmonotone',
                        line_search_config=None,
                        absolute_vi_directory: str = os.getcwd() + "/work/",
                        fom_sample_size: int = 10,
                        rom_extra_sample_size: int = 30,
                        rom_tolerance: float = 0.005,
                        max_rom_training_history: int = 1,
                        random_seed: int = 1,
                        fom_evaluation_concurrency=1,
                        rom_evaluation_concurrency=1,
                        covariance_regularization: float = 1e-8,
                        elbo_scaling_factor='diag_mean',
                        baseline_method: str = None,
                        variational_distribution: str = 'diagonal',
                        use_mfmc_control_variate: bool = True,
                        rom_base_sampling_strategy: str = 'coupled',
                        bounded_parameter_handling: str = 'transform',
                        transform_interior_margin: float = 1e-8,
                        min_physical_variational_std_fraction: float = 1e-8,
                        rom_type: str = "gp",
                        rom_args: Optional[dict] = None,
                        **legacy_kwargs):
    """
    Wrapper around run_mf_vi that selects a default ROM surrogate by rom_type.
    Accepts the same rom_base_sampling_strategy options as run_mf_vi.
    """
    parameter_mins, parameter_maxes = _resolve_parameter_bounds(
        parameter_mins,
        parameter_maxes,
        legacy_kwargs,
    )
    restart_file = _resolve_restart_file(restart_file, legacy_kwargs)
    resolved_optimizer_method, resolved_optimizer_config = _resolve_optimizer_config(
        optimizer_method,
        optimizer_config,
        legacy_kwargs,
        VIGradientOptimizerConfig(),
        VINewtonOptimizerConfig(newton_regularization=1e-8),
    )
    resolved_line_search_method, resolved_line_search_config = _resolve_line_search_config(
        line_search_method,
        line_search_config,
        legacy_kwargs,
        VILegacyLineSearchConfig(
            step_size_growth_factor=1.05,
            max_step_size_decrease_trys=10,
            relaxation_parameter=1000.0,
        ),
        VIStochasticNonmonotoneLineSearchConfig(
            step_size_growth_factor=1.05,
            max_step_size_decrease_trys=10,
            relaxation_parameter=1000.0,
            line_search_nonmonotone_window=8,
            line_search_uncertainty_sigma=5.0,
        ),
    )
    _raise_for_unused_legacy_kwargs(legacy_kwargs, 'mf_vi_with_auto_rom')

    rom_args = {} if rom_args is None else dict(rom_args)
    rom_type_normalized = rom_type.strip().lower()
    if rom_type_normalized == "gp":
        rom_model_builder = GaussianProcessQoiModelBuilderWithTrainingData(
            parameter_names=parameter_space.get_names(),
            pod_energy_fraction=rom_args.get("pod_energy_fraction", 0.999999),
            max_pod_modes=rom_args.get("max_pod_modes"),
            kernel=rom_args.get("kernel"),
            noise_variance=rom_args.get("noise_variance"),
            auto_noise_variance=rom_args.get("auto_noise_variance", False),
            noise_variance_fraction=rom_args.get("noise_variance_fraction", 1e-6),
            tune_hyperparameters=rom_args.get("tune_hyperparameters", False),
            length_scale_grid=rom_args.get("length_scale_grid"),
            signal_variance_grid=rom_args.get("signal_variance_grid"),
            normalize_parameters=rom_args.get("normalize_parameters", False),
            normalize_targets=rom_args.get("normalize_targets", False),
        )
    else:
        raise ValueError(f"Unsupported rom_type '{rom_type}'.")

    return run_mf_vi(
        model=model,
        rom_model_builder=rom_model_builder,
        parameter_space=parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=parameter_mins,
        parameter_maxes=parameter_maxes,
        restart_file=restart_file,
        optimizer_method=resolved_optimizer_method,
        optimizer_config=resolved_optimizer_config,
        line_search_method=resolved_line_search_method,
        line_search_config=resolved_line_search_config,
        absolute_vi_directory=absolute_vi_directory,
        fom_sample_size=fom_sample_size,
        rom_extra_sample_size=rom_extra_sample_size,
        rom_tolerance=rom_tolerance,
        max_rom_training_history=max_rom_training_history,
        random_seed=random_seed,
        fom_evaluation_concurrency=fom_evaluation_concurrency,
        rom_evaluation_concurrency=rom_evaluation_concurrency,
        covariance_regularization=covariance_regularization,
        elbo_scaling_factor=elbo_scaling_factor,
        baseline_method=baseline_method,
        variational_distribution=variational_distribution,
        use_mfmc_control_variate=use_mfmc_control_variate,
        rom_base_sampling_strategy=rom_base_sampling_strategy,
        bounded_parameter_handling=bounded_parameter_handling,
        transform_interior_margin=transform_interior_margin,
        min_physical_variational_std_fraction=min_physical_variational_std_fraction,
    )
