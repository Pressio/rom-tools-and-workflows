import numpy as np
import os
import time
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows.inverse.eki_utils import run_eki_iteration
from romtools.workflows.inverse.eki_utils import bound_samples
from romtools.workflows.inverse.vi_optimization_methods import (
    NewtonSolver,
    SteepestDescentSolver,
    VIGradientOptimizerConfig,
    VINewtonOptimizerConfig,
    VILegacyLineSearchConfig,
    VIStochasticNonmonotoneLineSearchConfig,
    _normalize_line_search_method,
    _normalize_line_search_objective,
    _normalize_newton_metric,
    _normalize_optimization_method,
    _resolve_line_search_config,
    _resolve_optimizer_config,
    _raise_for_unused_legacy_kwargs,
)


def _resolve_parameter_bounds(parameter_mins: np.ndarray,
                              parameter_maxes: np.ndarray,
                              legacy_kwargs: dict) -> tuple[np.ndarray, np.ndarray]:
    _ = legacy_kwargs
    return parameter_mins, parameter_maxes


def _resolve_restart_file(restart_file: str,
                          legacy_kwargs: dict) -> str:
    _ = legacy_kwargs
    if restart_file is None:
        return None
    if not isinstance(restart_file, str):
        raise TypeError("restart_file must be a string or None.")
    return restart_file


def _compute_log_likelihood_precision_operator(observations_covariance: np.ndarray,
                                               covariance_regularization: float) -> np.ndarray:
    qoi_dimensionality = observations_covariance.shape[0]
    regularized_dtype = np.result_type(observations_covariance, covariance_regularization, np.float64)
    covariance_diagonal = np.diag(observations_covariance)
    if np.array_equal(observations_covariance, np.diag(covariance_diagonal)):
        regularized_diagonal = covariance_diagonal.astype(regularized_dtype, copy=True)
        regularized_diagonal += covariance_regularization
        absolute_regularized_diagonal = np.abs(regularized_diagonal)
        cutoff = 1e-15 * np.max(absolute_regularized_diagonal)
        inverse_diagonal = np.zeros_like(regularized_diagonal, dtype=regularized_dtype)
        nonzero_entries = absolute_regularized_diagonal > cutoff
        inverse_diagonal[nonzero_entries] = 1.0 / regularized_diagonal[nonzero_entries]
        return inverse_diagonal

    regularized_covariance = observations_covariance.astype(regularized_dtype, copy=True)
    regularized_covariance.flat[::qoi_dimensionality + 1] += covariance_regularization
    return np.linalg.pinv(regularized_covariance)


def _compute_log_likelihoods(errors: np.ndarray,
                             observations_covariance: np.ndarray,
                             covariance_regularization: float,
                             precision_operator: np.ndarray = None):
    if precision_operator is None:
        precision_operator = _compute_log_likelihood_precision_operator(
            observations_covariance,
            covariance_regularization,
        )

    if precision_operator.ndim == 1:
        quadratic_terms = np.sum((errors * errors) * precision_operator[:, None], axis=0)
    else:
        weighted_errors = precision_operator @ errors
        quadratic_terms = np.sum(errors * weighted_errors, axis=0)
    misfits = 0.5 * quadratic_terms
    log_likelihoods = -misfits
    return log_likelihoods, misfits


def _compute_relative_mse(errors: np.ndarray,
                          observations: np.ndarray,
                          epsilon: float = 1e-16):
    mse_per_sample = np.mean(errors ** 2, axis=0)
    observation_mse = np.mean(observations ** 2)
    relative_mse_per_sample = mse_per_sample / (observation_mse + epsilon)
    return relative_mse_per_sample


def _compute_standard_error(samples: np.ndarray) -> float:
    sample_count = samples.size
    if sample_count <= 1:
        return 0.0
    return float(np.std(samples, ddof=1) / np.sqrt(sample_count))


def _compute_state_elbo_standard_error(state, elbo_scaling_factor: float) -> float:
    return abs(elbo_scaling_factor) * _compute_standard_error(state['log_likelihoods'])


def _compute_optimal_baseline(values: np.ndarray, score_functions: np.ndarray) -> np.ndarray:
    numerator = np.mean(values[:, None] * (score_functions ** 2), axis=0)
    denominator = np.mean(score_functions ** 2, axis=0)
    baseline = np.zeros_like(numerator)
    nonzero_denominator = denominator > 0.0
    baseline[nonzero_denominator] = numerator[nonzero_denominator] / denominator[nonzero_denominator]
    return baseline


def _compute_leave_one_out_baseline(values: np.ndarray) -> np.ndarray:
    sample_count = values.size
    if sample_count <= 1:
        return np.zeros_like(values)
    return (np.sum(values) - values) / float(sample_count - 1)


def _clip_variational_log_std(variational_log_std: np.ndarray,
                              min_variational_std: float,
                              max_variational_std: float) -> np.ndarray:
    min_log_std = np.log(min_variational_std)
    max_log_std = np.log(max_variational_std)
    return np.clip(variational_log_std, min_log_std, max_log_std)


def _compute_variational_std(variational_log_std: np.ndarray,
                             min_variational_std: float,
                             max_variational_std: float):
    clipped_log_std = _clip_variational_log_std(
        variational_log_std,
        min_variational_std,
        max_variational_std,
    )
    variational_std = np.exp(clipped_log_std)
    return variational_std, clipped_log_std


def _normalize_bounded_parameter_handling(bounded_parameter_handling: str):
    handling = bounded_parameter_handling.strip().lower()
    if handling not in ('clip', 'transform'):
        raise ValueError(
            f"Unsupported bounded_parameter_handling '{bounded_parameter_handling}'. "
            "Supported options are 'clip' and 'transform'."
        )
    return handling


def _normalize_baseline_method(baseline_method: str):
    method = baseline_method.strip().lower()
    if method not in ('none', 'loo', 'optimal'):
        raise ValueError(
            f"Unsupported baseline_method '{baseline_method}'. "
            "Supported options are 'none', 'loo', and 'optimal'."
        )
    return method


def _normalize_variational_distribution(variational_distribution: str):
    distribution = variational_distribution.strip().lower()
    if distribution not in ('diagonal', 'multivariate'):
        raise ValueError(
            f"Unsupported variational_distribution '{variational_distribution}'. "
            "Supported options are 'diagonal' and 'multivariate'."
        )
    return distribution


def _resolve_elbo_scaling_factor(elbo_scaling_factor,
                                 observations_covariance: np.ndarray) -> float:
    if isinstance(elbo_scaling_factor, str):
        scaling_mode = elbo_scaling_factor.strip().lower()
        covariance_diagonal = np.diag(observations_covariance)
        if scaling_mode in ('diag_mean', 'mean_diag_cov', 'auto'):
            covariance_diagonal_mean = float(np.mean(covariance_diagonal))
            if covariance_diagonal_mean <= 0.0:
                raise ValueError(
                    "Cannot compute diagonal-mean ELBO scaling: "
                    "mean(diag(observations_covariance)) must be positive."
                )
            return covariance_diagonal_mean
        if scaling_mode in ('diag_trace', 'trace_diag_cov', 'trace'):
            covariance_trace = float(np.sum(covariance_diagonal))
            if covariance_trace <= 0.0:
                raise ValueError(
                    "Cannot compute diagonal-trace ELBO scaling: "
                    "sum(diag(observations_covariance)) must be positive."
                )
            return covariance_trace
        raise ValueError(
            f"Unsupported elbo_scaling_factor mode '{elbo_scaling_factor}'. "
            "Supported string modes are 'diag_mean', 'diag_trace', and 'auto'."
        )
    return float(elbo_scaling_factor)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped_values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped_values))


def _logit(values: np.ndarray) -> np.ndarray:
    return np.log(values / (1.0 - values))


def _transform_optimizer_to_parameter(optimizer_samples: np.ndarray,
                                      parameter_mins: np.ndarray,
                                      parameter_maxes: np.ndarray) -> np.ndarray:
    bounded_unit = _sigmoid(optimizer_samples)
    return parameter_mins[None, :] + (parameter_maxes - parameter_mins)[None, :] * bounded_unit


def _transform_parameter_to_optimizer(parameter_samples: np.ndarray,
                                      parameter_mins: np.ndarray,
                                      parameter_maxes: np.ndarray) -> np.ndarray:
    parameter_ranges = parameter_maxes - parameter_mins
    unit_values = (parameter_samples - parameter_mins[None, :]) / parameter_ranges[None, :]
    unit_values = np.clip(unit_values, 1e-12, 1.0 - 1e-12)
    return _logit(unit_values)


def _compute_correlation_cholesky_from_samples(samples: np.ndarray,
                                               initial_jitter: float = 1e-10,
                                               max_jitter: float = 1e-2):
    dimensionality = samples.shape[1]
    if samples.shape[0] <= 1:
        return np.eye(dimensionality)
    if dimensionality == 1:
        return np.ones((1, 1))

    sample_covariance = np.cov(samples, rowvar=False, ddof=1)
    sample_covariance = 0.5 * (sample_covariance + sample_covariance.T)
    stds = np.sqrt(np.maximum(np.diag(sample_covariance), 1e-16))
    inv_stds = 1.0 / stds
    correlation = sample_covariance * inv_stds[:, None] * inv_stds[None, :]
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)

    jitter = max(initial_jitter, 0.0)
    while True:
        try:
            return np.linalg.cholesky(correlation + jitter * np.eye(dimensionality))
        except np.linalg.LinAlgError:
            if jitter >= max_jitter:
                return np.eye(dimensionality)
            jitter = max(10.0 * jitter, 1e-12)


def _print_vi_parameters(variational_mean: np.ndarray,
                         variational_log_std: np.ndarray,
                         min_variational_std: float,
                         max_variational_std: float,
                         optimizer_samples: np.ndarray = None,
                         parameter_samples: np.ndarray = None):
    _ = (min_variational_std, max_variational_std, optimizer_samples)

    if parameter_samples is not None and parameter_samples.size > 0:
        std_ddof = 1 if parameter_samples.shape[0] > 1 else 0
        parameter_means = np.mean(parameter_samples, axis=0)
        parameter_stds = np.std(parameter_samples, axis=0, ddof=std_ddof)
    else:
        parameter_means = variational_mean
        parameter_stds = np.exp(variational_log_std)

    parameter_means_str = np.array2string(
        parameter_means,
        precision=6,
        suppress_small=True,
        max_line_width=200,
    )
    parameter_stds_str = np.array2string(
        parameter_stds,
        precision=6,
        suppress_small=True,
        max_line_width=200,
    )
    print(f'  Parameter means: {parameter_means_str}')
    print(f'  Parameter stds:  {parameter_stds_str}')


def _compute_reinforce_gradients(parameter_samples: np.ndarray,
                                 variational_mean: np.ndarray,
                                 variational_std: np.ndarray,
                                 log_likelihoods: np.ndarray,
                                 baseline_method: str,
                                 variational_correlation_cholesky: np.ndarray = None,
                                 elbo_scaling_factor: float = 1.0):
    centered_parameters = parameter_samples - variational_mean[None, :]
    normalized_parameters = centered_parameters / variational_std[None, :]
    if variational_correlation_cholesky is None:
        score_mean = centered_parameters / (variational_std[None, :] ** 2)
        score_log_std = normalized_parameters ** 2 - 1.0
    else:
        correlation_solve = np.linalg.solve(
            variational_correlation_cholesky,
            normalized_parameters.transpose(),
        )
        correlation_inverse_times_normalized = np.linalg.solve(
            variational_correlation_cholesky.transpose(),
            correlation_solve,
        ).transpose()
        score_mean = correlation_inverse_times_normalized / variational_std[None, :]
        score_log_std = normalized_parameters * correlation_inverse_times_normalized - 1.0

    baseline_method = _normalize_baseline_method(baseline_method)
    if baseline_method == 'optimal':
        baseline_mean = _compute_optimal_baseline(log_likelihoods, score_mean)
        baseline_log_std = _compute_optimal_baseline(log_likelihoods, score_log_std)
        centered_weights_mean = log_likelihoods[:, None] - baseline_mean[None, :]
        centered_weights_log_std = log_likelihoods[:, None] - baseline_log_std[None, :]
    elif baseline_method == 'loo':
        baseline_loo = _compute_leave_one_out_baseline(log_likelihoods)
        baseline_mean = np.zeros_like(variational_mean)
        baseline_log_std = np.zeros_like(variational_mean)
        centered_weights_mean = (log_likelihoods - baseline_loo)[:, None]
        centered_weights_log_std = (log_likelihoods - baseline_loo)[:, None]
    else:
        baseline_mean = np.zeros_like(variational_mean)
        baseline_log_std = np.zeros_like(variational_mean)
        centered_weights_mean = log_likelihoods[:, None]
        centered_weights_log_std = log_likelihoods[:, None]

    gradient_mean = np.mean(centered_weights_mean * score_mean, axis=0)
    gradient_log_std = np.mean(centered_weights_log_std * score_log_std, axis=0) + elbo_scaling_factor
    return gradient_mean, gradient_log_std, baseline_mean, baseline_log_std


def _compute_reinforce_hessian_diagonal(parameter_samples: np.ndarray,
                                        variational_mean: np.ndarray,
                                        variational_std: np.ndarray,
                                        log_likelihoods: np.ndarray,
                                        baseline_method: str,
                                        variational_correlation_cholesky: np.ndarray = None):
    hessian_full = _compute_reinforce_hessian_full(
        parameter_samples,
        variational_mean,
        variational_std,
        log_likelihoods,
        baseline_method,
        variational_correlation_cholesky,
    )
    dimensionality = variational_mean.size
    hessian_mean = hessian_full[:dimensionality, :dimensionality]
    hessian_log_std = hessian_full[dimensionality:, dimensionality:]
    hessian_diagonal_mean = np.diag(hessian_mean)
    hessian_diagonal_log_std = np.diag(hessian_log_std)
    return hessian_diagonal_mean, hessian_diagonal_log_std


def _compute_reinforce_hessian_full(parameter_samples: np.ndarray,
                                    variational_mean: np.ndarray,
                                    variational_std: np.ndarray,
                                    log_likelihoods: np.ndarray,
                                    baseline_method: str,
                                    variational_correlation_cholesky: np.ndarray = None) -> np.ndarray:
    centered_parameters = parameter_samples - variational_mean[None, :]
    normalized_parameters = centered_parameters / variational_std[None, :]
    dimensionality = variational_mean.size
    if variational_correlation_cholesky is None:
        score_mean = centered_parameters / (variational_std[None, :] ** 2)
        score_log_std = normalized_parameters ** 2 - 1.0

        score_mean_outer = score_mean[:, :, None] * score_mean[:, None, :]
        score_log_std_outer = score_log_std[:, :, None] * score_log_std[:, None, :]
        score_cross_outer = score_mean[:, :, None] * score_log_std[:, None, :]

        second_mean = np.zeros_like(score_mean_outer)
        second_log_std = np.zeros_like(score_log_std_outer)
        second_cross = np.zeros_like(score_cross_outer)
        index = np.arange(dimensionality)
        second_mean[:, index, index] = -1.0 / (variational_std ** 2)
        second_log_std[:, index, index] = -2.0 * (normalized_parameters ** 2)
        second_cross[:, index, index] = -2.0 * score_mean
    else:
        identity = np.eye(dimensionality)
        correlation_solve = np.linalg.solve(variational_correlation_cholesky, identity)
        correlation_inverse = np.linalg.solve(
            variational_correlation_cholesky.transpose(),
            correlation_solve,
        )

        correlation_solve_samples = np.linalg.solve(
            variational_correlation_cholesky,
            normalized_parameters.transpose(),
        )
        correlation_inverse_times_normalized = np.linalg.solve(
            variational_correlation_cholesky.transpose(),
            correlation_solve_samples,
        ).transpose()

        score_mean = correlation_inverse_times_normalized / variational_std[None, :]
        score_log_std = normalized_parameters * correlation_inverse_times_normalized - 1.0

        score_mean_outer = score_mean[:, :, None] * score_mean[:, None, :]
        score_log_std_outer = score_log_std[:, :, None] * score_log_std[:, None, :]
        score_cross_outer = score_mean[:, :, None] * score_log_std[:, None, :]

        sample_count = parameter_samples.shape[0]
        inverse_std_outer = 1.0 / (variational_std[:, None] * variational_std[None, :])
        second_mean = np.broadcast_to(
            -correlation_inverse * inverse_std_outer,
            (sample_count, dimensionality, dimensionality),
        ).copy()

        outer_normalized = (
            normalized_parameters[:, :, None] * normalized_parameters[:, None, :]
        )
        second_log_std = -(outer_normalized * correlation_inverse[None, :, :])
        index = np.arange(dimensionality)
        second_log_std[:, index, index] -= (
            normalized_parameters * correlation_inverse_times_normalized
        )

        second_cross = -(
            (normalized_parameters[:, None, :] * correlation_inverse[None, :, :])
            / variational_std[None, :, None]
        )
        second_cross[:, index, index] -= score_mean

    baseline_method = _normalize_baseline_method(baseline_method)
    if baseline_method == 'optimal':
        baseline_mean = _compute_optimal_baseline(
            log_likelihoods,
            (score_mean_outer + second_mean).reshape(log_likelihoods.size, -1),
        ).reshape(dimensionality, dimensionality)
        baseline_log_std = _compute_optimal_baseline(
            log_likelihoods,
            (score_log_std_outer + second_log_std).reshape(log_likelihoods.size, -1),
        ).reshape(dimensionality, dimensionality)
        baseline_cross = _compute_optimal_baseline(
            log_likelihoods,
            (score_cross_outer + second_cross).reshape(log_likelihoods.size, -1),
        ).reshape(dimensionality, dimensionality)
        centered_mean_weights = log_likelihoods[:, None, None] - baseline_mean[None, :, :]
        centered_log_std_weights = log_likelihoods[:, None, None] - baseline_log_std[None, :, :]
        centered_cross_weights = log_likelihoods[:, None, None] - baseline_cross[None, :, :]
    elif baseline_method == 'loo':
        baseline_loo = _compute_leave_one_out_baseline(log_likelihoods)
        centered_mean_weights = (log_likelihoods - baseline_loo)[:, None, None]
        centered_log_std_weights = centered_mean_weights
        centered_cross_weights = centered_mean_weights
    else:
        centered_mean_weights = log_likelihoods[:, None, None]
        centered_log_std_weights = centered_mean_weights
        centered_cross_weights = centered_mean_weights

    hessian_mean = np.mean(
        centered_mean_weights * (score_mean_outer + second_mean),
        axis=0,
    )
    hessian_log_std = np.mean(
        centered_log_std_weights * (score_log_std_outer + second_log_std),
        axis=0,
    )
    hessian_cross = np.mean(
        centered_cross_weights * (score_cross_outer + second_cross),
        axis=0,
    )
    hessian_full = np.block([
        [hessian_mean, hessian_cross],
        [hessian_cross.transpose(), hessian_log_std],
    ])
    hessian_full = 0.5 * (hessian_full + hessian_full.transpose())
    return np.nan_to_num(hessian_full, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_update_directions(gradient_mean: np.ndarray,
                               gradient_log_std: np.ndarray,
                               variational_std: np.ndarray,
                               gradient_method: str):
    method = gradient_method.strip().lower()
    if method == 'standard':
        update_direction_mean = gradient_mean
        update_direction_log_std = gradient_log_std
    elif method == 'natural':
        update_direction_mean = (variational_std ** 2) * gradient_mean
        update_direction_log_std = 0.5 * gradient_log_std
    else:
        raise ValueError(
            f"Unsupported gradient_method '{gradient_method}'. "
            "Supported options are 'standard' and 'natural'."
        )
    return update_direction_mean, update_direction_log_std, method


def _compute_gradient_norm(state, optimization_method: str):
    if optimization_method == 'newton':
        return np.sqrt(
            np.linalg.norm(state['gradient_mean']) ** 2
            + np.linalg.norm(state['gradient_log_std']) ** 2
        )
    return np.sqrt(
        np.linalg.norm(state['update_direction_mean']) ** 2
        + np.linalg.norm(state['update_direction_log_std']) ** 2
    )


def _compute_newton_step(state,
                         newton_regularization: float,
                         metric_scale: np.ndarray = None):
    gradient = np.concatenate([state['gradient_mean'], state['gradient_log_std']])
    hessian = state.get('hessian_full')
    if hessian is None:
        hessian = np.concatenate([state['hessian_diagonal_mean'], state['hessian_diagonal_log_std']])
    transformed_gradient = gradient
    transformed_hessian = hessian
    if metric_scale is not None:
        transformed_gradient = metric_scale * gradient
        if transformed_hessian.ndim == 1:
            transformed_hessian = (metric_scale ** 2) * transformed_hessian
        else:
            transformed_hessian = (
                metric_scale[:, None] * transformed_hessian * metric_scale[None, :]
            )
    solver = NewtonSolver(regularization=newton_regularization)
    step = solver.step(transformed_gradient, transformed_hessian)
    if metric_scale is not None:
        step = metric_scale * step
    dimensionality = state['gradient_mean'].size
    mean_step = step[:dimensionality]
    log_std_step = step[dimensionality:]
    return mean_step, log_std_step


def _compute_newton_metric_scale(newton_metric: str,
                                 variational_std: np.ndarray) -> np.ndarray:
    if newton_metric == 'natural':
        dimensionality = variational_std.size
        return np.concatenate([variational_std, np.full(dimensionality, np.sqrt(0.5))])
    return None


def _draw_parameter_samples(variational_mean: np.ndarray,
                            variational_log_std: np.ndarray,
                            sample_size: int,
                            min_variational_std: float,
                            max_variational_std: float,
                            bounded_parameter_handling: str,
                            parameter_mins: np.ndarray,
                            parameter_maxes: np.ndarray,
                            standard_normal_samples: np.ndarray = None,
                            variational_correlation_cholesky: np.ndarray = None):
    bounded_parameter_handling = _normalize_bounded_parameter_handling(bounded_parameter_handling)
    variational_std, _ = _compute_variational_std(
        variational_log_std,
        min_variational_std,
        max_variational_std,
    )
    if standard_normal_samples is None:
        standard_normal_samples = np.random.randn(sample_size, variational_mean.size)
    else:
        assert standard_normal_samples.ndim == 2, "standard_normal_samples must be 2D"
        assert standard_normal_samples.shape[0] == sample_size, (
            "standard_normal_samples row count must equal sample_size"
        )
        assert standard_normal_samples.shape[1] == variational_mean.size, (
            "standard_normal_samples column count must match variational mean size"
        )
    if variational_correlation_cholesky is not None:
        transformed_standard_normal_samples = (
            standard_normal_samples @ variational_correlation_cholesky.transpose()
        )
    else:
        transformed_standard_normal_samples = standard_normal_samples
    optimizer_samples = (
        variational_mean[None, :] + variational_std[None, :] * transformed_standard_normal_samples
    )
    if bounded_parameter_handling == 'transform':
        parameter_samples = _transform_optimizer_to_parameter(
            optimizer_samples,
            parameter_mins,
            parameter_maxes,
        )
    else:
        parameter_samples = bound_samples(optimizer_samples, parameter_mins, parameter_maxes)
    return optimizer_samples, parameter_samples


def _run_vi_iteration_samples(model: QoiModel,
                              observations: np.ndarray,
                              run_directory_base: str,
                              parameter_names,
                              parameter_samples: np.ndarray,
                              evaluation_concurrency: int):
    return run_eki_iteration(
        model,
        observations,
        run_directory_base,
        parameter_names,
        parameter_samples,
        evaluation_concurrency,
    )


def _build_vi_state_from_results(optimizer_samples: np.ndarray,
                                 parameter_samples: np.ndarray,
                                 iteration_results,
                                 observations: np.ndarray,
                                 observations_covariance: np.ndarray,
                                 variational_mean: np.ndarray,
                                 variational_log_std: np.ndarray,
                                 covariance_regularization: float,
                                 baseline_method: str,
                                 gradient_method: str,
                                 min_variational_std: float,
                                 max_variational_std: float,
                                 variational_correlation_cholesky: np.ndarray = None,
                                 elbo_scaling_factor: float = 1.0):
    qois = iteration_results['qois']
    mean_qoi = iteration_results['mean-qoi']
    errors = iteration_results['errors']

    log_likelihoods, misfits = _compute_log_likelihoods(
        errors,
        observations_covariance,
        covariance_regularization,
    )
    scaled_log_likelihoods = elbo_scaling_factor * log_likelihoods
    relative_mses = _compute_relative_mse(errors, observations)

    variational_std, variational_log_std = _compute_variational_std(
        variational_log_std,
        min_variational_std,
        max_variational_std,
    )
    gradient_mean, gradient_log_std, baseline_mean, baseline_log_std = _compute_reinforce_gradients(
        optimizer_samples,
        variational_mean,
        variational_std,
        scaled_log_likelihoods,
        baseline_method,
        variational_correlation_cholesky,
        elbo_scaling_factor,
    )
    hessian_diagonal_mean, hessian_diagonal_log_std = _compute_reinforce_hessian_diagonal(
        optimizer_samples,
        variational_mean,
        variational_std,
        scaled_log_likelihoods,
        baseline_method,
        variational_correlation_cholesky,
    )
    hessian_full = _compute_reinforce_hessian_full(
        optimizer_samples,
        variational_mean,
        variational_std,
        scaled_log_likelihoods,
        baseline_method,
        variational_correlation_cholesky,
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
    elbo = np.mean(scaled_log_likelihoods) + entropy

    state = {
        'optimizer_samples': optimizer_samples,
        'parameter_samples': parameter_samples,
        'qois': qois,
        'mean_qoi': mean_qoi,
        'errors': errors,
        'log_likelihoods': log_likelihoods,
        'mean_misfit': np.mean(misfits),
        'mean_relative_mse': np.mean(relative_mses),
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
    }
    return state


def _evaluate_vi_state(model: QoiModel,
                       observations: np.ndarray,
                       observations_covariance: np.ndarray,
                       run_directory_base: str,
                       parameter_names,
                       variational_mean: np.ndarray,
                       variational_log_std: np.ndarray,
                       sample_size: int,
                       evaluation_concurrency: int,
                       covariance_regularization: float,
                       baseline_method: str,
                       gradient_method: str,
                       bounded_parameter_handling: str,
                       min_variational_std: float,
                       max_variational_std: float,
                       parameter_mins: np.ndarray,
                       parameter_maxes: np.ndarray,
                       variational_correlation_cholesky: np.ndarray = None,
                       elbo_scaling_factor: float = 1.0):
    optimizer_samples, parameter_samples = _draw_parameter_samples(
        variational_mean,
        variational_log_std,
        sample_size,
        min_variational_std,
        max_variational_std,
        bounded_parameter_handling,
        parameter_mins,
        parameter_maxes,
        variational_correlation_cholesky=variational_correlation_cholesky,
    )

    iteration_results = _run_vi_iteration_samples(
        model,
        observations,
        run_directory_base,
        parameter_names,
        parameter_samples,
        evaluation_concurrency,
    )
    return _build_vi_state_from_results(
        optimizer_samples=optimizer_samples,
        parameter_samples=parameter_samples,
        iteration_results=iteration_results,
        observations=observations,
        observations_covariance=observations_covariance,
        variational_mean=variational_mean,
        variational_log_std=variational_log_std,
        covariance_regularization=covariance_regularization,
        baseline_method=baseline_method,
        gradient_method=gradient_method,
        min_variational_std=min_variational_std,
        max_variational_std=max_variational_std,
        variational_correlation_cholesky=variational_correlation_cholesky,
        elbo_scaling_factor=elbo_scaling_factor,
    )

def _evaluate_vi_candidate_for_line_search(model: QoiModel,
                                           observations: np.ndarray,
                                           observations_covariance: np.ndarray,
                                           run_directory_base: str,
                                           parameter_names,
                                           variational_mean: np.ndarray,
                                           variational_log_std: np.ndarray,
                                           sample_size: int,
                                           evaluation_concurrency: int,
                                           covariance_regularization: float,
                                           baseline_method: str,
                                           gradient_method: str,
                                           bounded_parameter_handling: str,
                                           min_variational_std: float,
                                           max_variational_std: float,
                                           parameter_mins: np.ndarray,
                                           parameter_maxes: np.ndarray,
                                           line_search_objective: str,
                                           standard_normal_samples: np.ndarray = None,
                                           variational_correlation_cholesky: np.ndarray = None,
                                           elbo_scaling_factor: float = 1.0):
    optimizer_samples, parameter_samples = _draw_parameter_samples(
        variational_mean,
        variational_log_std,
        sample_size,
        min_variational_std,
        max_variational_std,
        bounded_parameter_handling,
        parameter_mins,
        parameter_maxes,
        standard_normal_samples,
        variational_correlation_cholesky,
    )
    iteration_results = _run_vi_iteration_samples(
        model,
        observations,
        run_directory_base,
        parameter_names,
        parameter_samples,
        evaluation_concurrency,
    )
    mean_relative_mse = np.mean(_compute_relative_mse(iteration_results['errors'], observations))
    candidate = {
        'optimizer_samples': optimizer_samples,
        'parameter_samples': parameter_samples,
        'iteration_results': iteration_results,
        'mean_relative_mse': mean_relative_mse,
    }
    if line_search_objective == 'elbo':
        candidate['state'] = _build_vi_state_from_results(
            optimizer_samples=optimizer_samples,
            parameter_samples=parameter_samples,
            iteration_results=iteration_results,
            observations=observations,
            observations_covariance=observations_covariance,
            variational_mean=variational_mean,
            variational_log_std=variational_log_std,
            covariance_regularization=covariance_regularization,
            baseline_method=baseline_method,
            gradient_method=gradient_method,
            min_variational_std=min_variational_std,
            max_variational_std=max_variational_std,
            variational_correlation_cholesky=variational_correlation_cholesky,
            elbo_scaling_factor=elbo_scaling_factor,
        )
    return candidate


def _save_vi_restart(restart_path: str,
                     state,
                     variational_mean: np.ndarray,
                     variational_log_std: np.ndarray,
                     variational_distribution: str,
                     variational_correlation_cholesky: np.ndarray,
                     elbo_scaling_factor: float,
                     iteration: int,
                     step_size: float,
                     bounded_parameter_handling: str,
                     baseline_method: str,
                     optimization_method: str,
                     line_search_objective: str,
                     line_search_method: str,
                     line_search_nonmonotone_window: int = None,
                     line_search_armijo_coefficient: float = None,
                     line_search_uncertainty_sigma: float = None,
                     line_search_sample_growth_factor: float = None,
                     log_std_learning_rate_factor: float = None):
    save_data = dict(
        optimizer_samples=state['optimizer_samples'],
        parameter_samples=state['parameter_samples'],
        qois=state['qois'],
        mean_qoi=state['mean_qoi'],
        errors=state['errors'],
        log_likelihoods=state['log_likelihoods'],
        variational_mean=variational_mean,
        variational_log_std=variational_log_std,
        variational_distribution=variational_distribution,
        elbo_scaling_factor=elbo_scaling_factor,
        iteration=iteration,
        step_size=step_size,
        bounded_parameter_handling=bounded_parameter_handling,
        baseline_method=baseline_method,
        optimization_method=optimization_method,
        line_search_objective=line_search_objective,
        line_search_method=line_search_method,
        line_search_sample_growth_factor=line_search_sample_growth_factor,
        log_std_learning_rate_factor=log_std_learning_rate_factor,
    )
    if line_search_method == 'stochastic_nonmonotone':
        save_data['line_search_nonmonotone_window'] = line_search_nonmonotone_window
        save_data['line_search_armijo_coefficient'] = line_search_armijo_coefficient
        save_data['line_search_uncertainty_sigma'] = line_search_uncertainty_sigma
    if variational_correlation_cholesky is not None:
        save_data['variational_correlation_cholesky'] = variational_correlation_cholesky
    np.savez(restart_path, **save_data)


def _validate_run_vi_inputs(absolute_vi_directory: str,
                            sample_size: int,
                            max_step_size: float,
                            step_size_growth_factor: float,
                            step_size_decay_factor: float,
                            line_search_sample_growth_factor: float,
                            line_search_method: str,
                            line_search_nonmonotone_window: int,
                            line_search_armijo_coefficient: float,
                            line_search_uncertainty_sigma: float,
                            log_std_learning_rate_factor: float,
                            relaxation_parameter: float,
                            min_variational_std: float,
                            max_variational_std: float,
                            max_log_std_update: float,
                            newton_regularization: float,
                            covariance_regularization: float,
                            observations_covariance: np.ndarray,
                            observations: np.ndarray,
                            elbo_scaling_factor: float,
                            parameter_space: ParameterSpace,
                            parameter_mins: np.ndarray,
                            parameter_maxes: np.ndarray,
                            bounded_parameter_handling: str) -> None:
    assert os.path.isabs(absolute_vi_directory), (
        f"absolute_vi_directory is not an absolute path ({absolute_vi_directory})"
    )
    assert sample_size > 1, "sample_size must be greater than 1"
    assert max_step_size > 0.0, "max_step_size must be positive"
    assert step_size_growth_factor >= 1.0, "step_size_growth_factor must be greater than 1.0"
    assert step_size_decay_factor >= 1.0, "step_size_decay_factor must be greater than 1.0"
    assert line_search_sample_growth_factor >= 1.0, (
        "line_search_sample_growth_factor must be greater than or equal to 1.0"
    )
    if line_search_method == 'stochastic_nonmonotone':
        assert line_search_nonmonotone_window >= 1, "line_search_nonmonotone_window must be >= 1"
        assert 0.0 <= line_search_armijo_coefficient <= 1.0, (
            "line_search_armijo_coefficient must be in [0,1]"
        )
        assert line_search_uncertainty_sigma >= 0.0, "line_search_uncertainty_sigma must be non-negative"
    assert log_std_learning_rate_factor > 0.0, (
        "log_std_learning_rate_factor must be positive"
    )
    assert relaxation_parameter >= 1.0, "relaxation_parameter must be >= 1.0"
    assert min_variational_std > 0.0, "min_variational_std must be positive"
    assert max_variational_std > min_variational_std, (
        "max_variational_std must be greater than min_variational_std"
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


def run_vi(model: QoiModel,
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
           sample_size: int = 30,
           random_seed: int = 1,
           evaluation_concurrency=1,
           covariance_regularization: float = 1e-8,
           elbo_scaling_factor='diag_mean',
           baseline_method: str = None,
           variational_distribution: str = 'diagonal',
           bounded_parameter_handling: str = 'transform',
           **legacy_kwargs):
    '''
    Run Gaussian variational inference with score-function gradients.

    This routine approximates the posterior with a Gaussian variational family
    (diagonal or fixed-correlation multivariate) and updates its mean and
    log-standard deviation using REINFORCE. An optional baseline can be
    enabled to reduce gradient estimator variance.

    Args:
        model: QoiModel to evaluate at sampled parameters.
        parameter_space: ParameterSpace used for initial variational moments.
        observations: Observed QoI vector.
        observations_covariance: Observation covariance matrix.
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
        absolute_vi_directory: Absolute path to the working directory for runs.
        sample_size: Number of MC samples per iteration.
        random_seed: RNG seed for reproducibility.
        evaluation_concurrency: Concurrent model evaluations per iteration.
        covariance_regularization: Diagonal regularization for covariance inversion.
        elbo_scaling_factor: Positive scalar that multiplies the ELBO objective,
            or a string mode. Supported string modes are:
            'diag_mean' (or 'auto'): mean(diag(observations_covariance)),
            'diag_trace': sum(diag(observations_covariance)).
        baseline_method: Baseline used in REINFORCE gradient estimation.
            Supported options are 'none', 'loo', and 'optimal'.
        variational_distribution: Variational family used for sampling and
            score computation. Supported options are 'diagonal' and
            'multivariate'. The multivariate option uses a full
            correlated Gaussian with fixed correlation structure and learned
            means/scales.
        bounded_parameter_handling: Parameter bounds handling. Supported options
            are 'clip' and 'transform'.
        legacy_kwargs: Optional scalar overrides for fields in
            `optimizer_config` and `line_search_config`.

    Returns:
        Tuple of (variational_mean, variational_std, parameter_samples, qois).
    '''

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
        VINewtonOptimizerConfig(),
    )
    line_search_method, resolved_line_search_config = _resolve_line_search_config(
        line_search_method,
        line_search_config,
        legacy_kwargs,
        VILegacyLineSearchConfig(),
        VIStochasticNonmonotoneLineSearchConfig(),
    )
    _raise_for_unused_legacy_kwargs(legacy_kwargs, 'run_vi')

    gradient_method = 'standard'
    if optimization_method == 'gradient':
        gradient_method = resolved_optimizer_config.gradient_method
    gradient_norm_tolerance = resolved_optimizer_config.gradient_norm_tolerance
    max_iterations = resolved_optimizer_config.max_iterations
    max_log_std_update = resolved_optimizer_config.max_log_std_update
    min_variational_std = resolved_optimizer_config.min_variational_std
    max_variational_std = resolved_optimizer_config.max_variational_std

    newton_defaults = VINewtonOptimizerConfig()
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
    line_search_objective = resolved_line_search_config.line_search_objective
    line_search_sample_growth_factor = resolved_line_search_config.line_search_sample_growth_factor
    log_std_learning_rate_factor = resolved_line_search_config.log_std_learning_rate_factor
    line_search_nonmonotone_window = 1
    line_search_armijo_coefficient = 0.0
    line_search_uncertainty_sigma = 0.0
    if line_search_method == 'stochastic_nonmonotone':
        line_search_nonmonotone_window = resolved_line_search_config.line_search_nonmonotone_window
        line_search_armijo_coefficient = resolved_line_search_config.line_search_armijo_coefficient
        line_search_uncertainty_sigma = resolved_line_search_config.line_search_uncertainty_sigma

    elbo_scaling_factor = _resolve_elbo_scaling_factor(
        elbo_scaling_factor,
        observations_covariance,
    )
    bounded_parameter_handling = _normalize_bounded_parameter_handling(bounded_parameter_handling)
    if baseline_method is None:
        baseline_method = 'loo'
    baseline_method = _normalize_baseline_method(baseline_method)
    variational_distribution = _normalize_variational_distribution(variational_distribution)
    line_search_objective = _normalize_line_search_objective(line_search_objective)
    if bounded_parameter_handling == 'transform':
        if parameter_mins is None and hasattr(parameter_space, 'parameter_mins'):
            parameter_mins = np.asarray(parameter_space.parameter_mins)
        if parameter_maxes is None and hasattr(parameter_space, 'parameter_maxes'):
            parameter_maxes = np.asarray(parameter_space.parameter_maxes)
    _validate_run_vi_inputs(
        absolute_vi_directory=absolute_vi_directory,
        sample_size=sample_size,
        max_step_size=max_step_size,
        step_size_growth_factor=step_size_growth_factor,
        step_size_decay_factor=step_size_decay_factor,
        line_search_sample_growth_factor=line_search_sample_growth_factor,
        line_search_method=line_search_method,
        line_search_nonmonotone_window=line_search_nonmonotone_window,
        line_search_armijo_coefficient=line_search_armijo_coefficient,
        line_search_uncertainty_sigma=line_search_uncertainty_sigma,
        log_std_learning_rate_factor=log_std_learning_rate_factor,
        relaxation_parameter=relaxation_parameter,
        min_variational_std=min_variational_std,
        max_variational_std=max_variational_std,
        max_log_std_update=max_log_std_update,
        newton_regularization=newton_regularization,
        covariance_regularization=covariance_regularization,
        observations_covariance=observations_covariance,
        observations=observations,
        elbo_scaling_factor=elbo_scaling_factor,
        parameter_space=parameter_space,
        parameter_mins=parameter_mins,
        parameter_maxes=parameter_maxes,
        bounded_parameter_handling=bounded_parameter_handling,
    )
    np.random.seed(random_seed)
    parameter_names = parameter_space.get_names()
    variational_correlation_cholesky = None

    if restart_file is None:
        iteration = 0
        step_size = min(initial_step_size, max_step_size)

        initial_samples = parameter_space.generate_samples(sample_size)
        initial_samples = bound_samples(initial_samples, parameter_mins, parameter_maxes)
        if bounded_parameter_handling == 'transform':
            optimizer_initial_samples = _transform_parameter_to_optimizer(
                initial_samples,
                parameter_mins,
                parameter_maxes,
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
        if variational_distribution == 'multivariate':
            variational_correlation_cholesky = _compute_correlation_cholesky_from_samples(
                optimizer_initial_samples
            )

        run_directory_base = f'{absolute_vi_directory}/iteration_{iteration}/run_'
        state = _evaluate_vi_state(
            model,
            observations,
            observations_covariance,
            run_directory_base,
            parameter_names,
            variational_mean,
            variational_log_std,
            sample_size,
            evaluation_concurrency,
            covariance_regularization,
            baseline_method,
            gradient_method,
            bounded_parameter_handling,
            min_variational_std,
            max_variational_std,
            parameter_mins,
            parameter_maxes,
            variational_correlation_cholesky,
            elbo_scaling_factor,
        )
    else:
        restart_data = np.load(restart_file)
        iteration = int(restart_data['iteration'])
        step_size = min(float(restart_data['step_size']), max_step_size)
        variational_mean = restart_data['variational_mean']
        variational_log_std = _clip_variational_log_std(
            restart_data['variational_log_std'],
            min_variational_std,
            max_variational_std,
        )
        if 'bounded_parameter_handling' in restart_data:
            restart_bounded_parameter_handling = str(restart_data['bounded_parameter_handling'].item())
            if restart_bounded_parameter_handling != bounded_parameter_handling:
                raise ValueError(
                    "Restart file bounded_parameter_handling does not match current run."
                )
        if 'baseline_method' in restart_data:
            restart_baseline_method = str(restart_data['baseline_method'].item())
            if restart_baseline_method != baseline_method:
                raise ValueError("Restart file baseline_method does not match current run.")
        if 'variational_distribution' in restart_data:
            restart_variational_distribution = str(restart_data['variational_distribution'].item())
            if restart_variational_distribution != variational_distribution:
                raise ValueError("Restart file variational_distribution does not match current run.")
        if variational_distribution == 'multivariate':
            if 'variational_correlation_cholesky' in restart_data:
                variational_correlation_cholesky = restart_data['variational_correlation_cholesky']
            else:
                variational_correlation_cholesky = np.eye(variational_mean.size)
        if 'optimization_method' in restart_data:
            restart_optimization_method = _normalize_optimization_method(
                str(restart_data['optimization_method'].item())
            )
            if restart_optimization_method != optimization_method:
                raise ValueError("Restart file optimization_method does not match current run.")
        if 'line_search_objective' in restart_data:
            restart_line_search_objective = str(restart_data['line_search_objective'].item())
            if restart_line_search_objective != line_search_objective:
                raise ValueError(
                    "Restart file line_search_objective does not match current run."
                )
        if 'line_search_method' in restart_data:
            restart_line_search_method = _normalize_line_search_method(
                str(restart_data['line_search_method'].item())
            )
            if restart_line_search_method != line_search_method:
                raise ValueError(
                    "Restart file line_search_method does not match current run."
                )
        if line_search_method == 'stochastic_nonmonotone':
            if 'line_search_nonmonotone_window' in restart_data:
                restart_line_search_nonmonotone_window = int(restart_data['line_search_nonmonotone_window'])
                if restart_line_search_nonmonotone_window != line_search_nonmonotone_window:
                    raise ValueError(
                        "Restart file line_search_nonmonotone_window does not match current run."
                    )
            if 'line_search_armijo_coefficient' in restart_data:
                restart_line_search_armijo_coefficient = float(restart_data['line_search_armijo_coefficient'])
                if not np.isclose(restart_line_search_armijo_coefficient, line_search_armijo_coefficient):
                    raise ValueError(
                        "Restart file line_search_armijo_coefficient does not match current run."
                    )
            if 'line_search_uncertainty_sigma' in restart_data:
                restart_line_search_uncertainty_sigma = float(restart_data['line_search_uncertainty_sigma'])
                if not np.isclose(restart_line_search_uncertainty_sigma, line_search_uncertainty_sigma):
                    raise ValueError(
                        "Restart file line_search_uncertainty_sigma does not match current run."
                    )
        if 'line_search_sample_growth_factor' in restart_data:
            restart_line_search_sample_growth_factor = float(restart_data['line_search_sample_growth_factor'])
            if not np.isclose(restart_line_search_sample_growth_factor, line_search_sample_growth_factor):
                raise ValueError(
                    "Restart file line_search_sample_growth_factor does not match current run."
                )
        if 'log_std_learning_rate_factor' in restart_data:
            restart_log_std_learning_rate_factor = float(restart_data['log_std_learning_rate_factor'])
            if not np.isclose(restart_log_std_learning_rate_factor, log_std_learning_rate_factor):
                raise ValueError(
                    "Restart file log_std_learning_rate_factor does not match current run."
                )
        if 'elbo_scaling_factor' in restart_data:
            restart_elbo_scaling_factor = float(restart_data['elbo_scaling_factor'])
            if not np.isclose(restart_elbo_scaling_factor, elbo_scaling_factor):
                raise ValueError(
                    "Restart file elbo_scaling_factor does not match current run."
                )

        state = {
            'optimizer_samples': restart_data['optimizer_samples']
            if 'optimizer_samples' in restart_data else restart_data['parameter_samples'],
            'parameter_samples': restart_data['parameter_samples'],
            'qois': restart_data['qois'],
            'mean_qoi': restart_data['mean_qoi'],
            'errors': restart_data['errors'],
        }
        if bounded_parameter_handling == 'transform' and 'optimizer_samples' not in restart_data:
            raise ValueError(
                "Restart file missing optimizer_samples required for bounded_parameter_handling='transform'."
            )

        if 'log_likelihoods' in restart_data:
            state['log_likelihoods'] = restart_data['log_likelihoods']
            state['mean_misfit'] = -np.mean(state['log_likelihoods'])
            state['mean_relative_mse'] = np.mean(_compute_relative_mse(state['errors'], observations))
        else:
            log_likelihoods, misfits = _compute_log_likelihoods(
                state['errors'],
                observations_covariance,
                covariance_regularization,
            )
            state['log_likelihoods'] = log_likelihoods
            state['mean_misfit'] = np.mean(misfits)
            state['mean_relative_mse'] = np.mean(_compute_relative_mse(state['errors'], observations))

        variational_std, variational_log_std = _compute_variational_std(
            variational_log_std,
            min_variational_std,
            max_variational_std,
        )
        gradient_mean, gradient_log_std, baseline_mean, baseline_log_std = _compute_reinforce_gradients(
            state['optimizer_samples'],
            variational_mean,
            variational_std,
            elbo_scaling_factor * state['log_likelihoods'],
            baseline_method,
            variational_correlation_cholesky,
            elbo_scaling_factor,
        )
        hessian_diagonal_mean, hessian_diagonal_log_std = _compute_reinforce_hessian_diagonal(
            state['optimizer_samples'],
            variational_mean,
            variational_std,
            elbo_scaling_factor * state['log_likelihoods'],
            baseline_method,
            variational_correlation_cholesky,
        )
        hessian_full = _compute_reinforce_hessian_full(
            state['optimizer_samples'],
            variational_mean,
            variational_std,
            elbo_scaling_factor * state['log_likelihoods'],
            baseline_method,
            variational_correlation_cholesky,
        )
        update_direction_mean, update_direction_log_std, normalized_gradient_method = _compute_update_directions(
            gradient_mean,
            gradient_log_std,
            variational_std,
            gradient_method,
        )
        state['gradient_mean'] = gradient_mean
        state['gradient_log_std'] = gradient_log_std
        state['hessian_diagonal_mean'] = hessian_diagonal_mean
        state['hessian_diagonal_log_std'] = hessian_diagonal_log_std
        state['hessian_full'] = hessian_full
        state['update_direction_mean'] = update_direction_mean
        state['update_direction_log_std'] = update_direction_log_std
        state['gradient_method'] = normalized_gradient_method
        state['baseline_mean'] = baseline_mean
        state['baseline_log_std'] = baseline_log_std
        state['entropy'] = np.sum(variational_log_std) + 0.5 * variational_mean.size * (1.0 + np.log(2.0 * np.pi))
        if variational_correlation_cholesky is not None:
            state['entropy'] += np.sum(np.log(np.diag(variational_correlation_cholesky)))
        state['entropy'] *= elbo_scaling_factor
        state['elbo'] = np.mean(elbo_scaling_factor * state['log_likelihoods']) + state['entropy']
    if restart_file is None:
        _save_vi_restart(
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
            baseline_method,
            optimization_method,
            line_search_objective,
            line_search_method,
            line_search_nonmonotone_window,
            line_search_armijo_coefficient,
            line_search_uncertainty_sigma,
            line_search_sample_growth_factor,
            log_std_learning_rate_factor,
        )

    gradient_norm = _compute_gradient_norm(state, optimization_method)
    wall_time = time.time() - start_time
    print(
        f'Iteration: {iteration}, Relative MSE: {state["mean_relative_mse"]:.5f}, '
        f'ELBO: {state["elbo"]:.5f}, Step size: {step_size:.5f}, '
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

    iteration += 1
    step_failed_counter = 0
    line_search_standard_normal_cache = None
    steepest_descent_solver = SteepestDescentSolver()
    accepted_elbo_history = [state['elbo']]

    while iteration < max_iterations and gradient_norm > gradient_norm_tolerance:
        line_search_sample_size = int(np.ceil(
            sample_size * (line_search_sample_growth_factor ** step_failed_counter)
        ))
        line_search_sample_size = max(line_search_sample_size, 2)
        if line_search_standard_normal_cache is None:
            line_search_standard_normal_cache = np.random.randn(
                line_search_sample_size,
                variational_mean.size,
            )
        elif line_search_sample_size > line_search_standard_normal_cache.shape[0]:
            extra_sample_count = line_search_sample_size - line_search_standard_normal_cache.shape[0]
            extra_standard_normal_samples = np.random.randn(
                extra_sample_count,
                variational_mean.size,
            )
            line_search_standard_normal_cache = np.vstack(
                [line_search_standard_normal_cache, extra_standard_normal_samples]
            )
        line_search_standard_normal_samples = line_search_standard_normal_cache[:line_search_sample_size]
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

        if optimization_method == 'gradient':
            gradient = np.concatenate([state['update_direction_mean'], state['update_direction_log_std']])
            step = steepest_descent_solver.step(gradient)
            dimensionality = state['update_direction_mean'].size
            direction_mean = step[:dimensionality]
            direction_log_std = step[dimensionality:]

            line_search_predicted_slope = float(
                np.dot(state['gradient_mean'], direction_mean)
                + log_std_learning_rate_factor * np.dot(state['gradient_log_std'], direction_log_std)
            )
            test_variational_mean = variational_mean + step_size * direction_mean
            log_std_update = np.clip(
                step_size * log_std_learning_rate_factor * direction_log_std,
                -max_log_std_update,
                max_log_std_update,
            )
            test_variational_log_std = variational_log_std + log_std_update
            test_variational_log_std = _clip_variational_log_std(
                test_variational_log_std,
                min_variational_std,
                max_variational_std,
            )

            run_directory_base = f'{absolute_vi_directory}/iteration_{iteration}/run_'
            test_candidate = _evaluate_vi_candidate_for_line_search(
                model,
                observations,
                observations_covariance,
                run_directory_base,
                parameter_names,
                test_variational_mean,
                test_variational_log_std,
                line_search_sample_size,
                evaluation_concurrency,
                covariance_regularization,
                baseline_method,
                gradient_method,
                bounded_parameter_handling,
                min_variational_std,
                max_variational_std,
                parameter_mins,
                parameter_maxes,
                line_search_objective,
                line_search_standard_normal_samples,
                variational_correlation_cholesky,
                elbo_scaling_factor,
            )

            if line_search_objective == 'elbo':
                test_state = test_candidate['state']
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
            else:
                test_state = None
                accept_step = test_candidate['mean_relative_mse'] <= (
                    relaxation_parameter * state['mean_relative_mse']
                )

            if accept_step:
                step_failed_counter = 0
                variational_mean = test_variational_mean*1.0
                variational_log_std = test_variational_log_std*1.0
                if test_state is None:
                    test_state = _build_vi_state_from_results(
                        optimizer_samples=test_candidate['optimizer_samples'],
                        parameter_samples=test_candidate['parameter_samples'],
                        iteration_results=test_candidate['iteration_results'],
                        observations=observations,
                        observations_covariance=observations_covariance,
                        variational_mean=test_variational_mean,
                        variational_log_std=test_variational_log_std,
                        covariance_regularization=covariance_regularization,
                        baseline_method=baseline_method,
                        gradient_method=gradient_method,
                        min_variational_std=min_variational_std,
                        max_variational_std=max_variational_std,
                        variational_correlation_cholesky=variational_correlation_cholesky,
                        elbo_scaling_factor=elbo_scaling_factor,
                    )
                state = test_state
                accepted_elbo_history.append(state['elbo'])
                step_size = min(step_size * step_size_growth_factor, max_step_size)
                line_search_standard_normal_cache = None

                gradient_norm = _compute_gradient_norm(state, optimization_method)
                wall_time = time.time() - start_time
                print(
                    f'Iteration: {iteration}, Relative MSE: {state["mean_relative_mse"]:.5f}, '
                    f'ELBO: {state["elbo"]:.5f}, Step size: {step_size:.5e}, '
                    f'Gradient norm: {gradient_norm:.5f}, Samples: {line_search_sample_size}, '
                    f'Wall time: {wall_time:.5f}'
                )
                _print_vi_parameters(
                    variational_mean,
                    variational_log_std,
                    min_variational_std,
                    max_variational_std,
                    optimizer_samples=state['optimizer_samples'],
                    parameter_samples=state['parameter_samples'],
                )

                _save_vi_restart(
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
                    baseline_method,
                    optimization_method,
                    line_search_objective,
                    line_search_method,
                    line_search_nonmonotone_window,
                    line_search_armijo_coefficient,
                    line_search_uncertainty_sigma,
                    line_search_sample_growth_factor,
                    log_std_learning_rate_factor,
                )
                iteration += 1
            else:
                step_failed_counter += 1
                step_size /= step_size_decay_factor
                if line_search_objective == 'elbo':
                    print(
                        f'  Warning, lowering step size, Iteration: {iteration}, '
                        f'Relative MSE: {state["mean_relative_mse"]:.5f}, '
                        f'ELBO: {state["elbo"]:.5f}, Step size: {step_size:.5e}, '
                        f'Gradient norm: {gradient_norm:.5f}, Samples: {line_search_sample_size}'
                    )
                else:
                    print(
                        f'  Warning, lowering step size, Iteration: {iteration}, '
                        f'Relative MSE: {state["mean_relative_mse"]:.5f}, '
                        f'Test Relative MSE: {test_candidate["mean_relative_mse"]:.5f}, '
                        f'ELBO: {state["elbo"]:.5f}, Step size: {step_size:.5e}, '
                        f'Gradient norm: {gradient_norm:.5f}, Samples: {line_search_sample_size}'
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
        else:
            direction_mean, direction_log_std = _compute_newton_step(
                state,
                newton_regularization,
                metric_scale=newton_metric_scale,
            )
            line_search_predicted_slope = float(
                np.dot(state['gradient_mean'], direction_mean)
                + log_std_learning_rate_factor * np.dot(state['gradient_log_std'], direction_log_std)
            )
            test_variational_mean = variational_mean + step_size * direction_mean
            log_std_update = np.clip(
                step_size * log_std_learning_rate_factor * direction_log_std,
                -max_log_std_update,
                max_log_std_update,
            )
            test_variational_log_std = variational_log_std + log_std_update
            test_variational_log_std = _clip_variational_log_std(
                test_variational_log_std,
                min_variational_std,
                max_variational_std,
            )

            run_directory_base = f'{absolute_vi_directory}/iteration_{iteration}/run_'
            test_candidate = _evaluate_vi_candidate_for_line_search(
                model,
                observations,
                observations_covariance,
                run_directory_base,
                parameter_names,
                test_variational_mean,
                test_variational_log_std,
                line_search_sample_size,
                evaluation_concurrency,
                covariance_regularization,
                baseline_method,
                gradient_method,
                bounded_parameter_handling,
                min_variational_std,
                max_variational_std,
                parameter_mins,
                parameter_maxes,
                line_search_objective,
                line_search_standard_normal_samples,
                variational_correlation_cholesky,
                elbo_scaling_factor,
            )

            if line_search_objective == 'elbo':
                test_state = test_candidate['state']
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
            else:
                test_state = None
                accept_step = test_candidate['mean_relative_mse'] <= (
                    relaxation_parameter * state['mean_relative_mse']
                )

            if accept_step:
                step_failed_counter = 0
                variational_mean = test_variational_mean*1.0
                variational_log_std = test_variational_log_std*1.0
                if test_state is None:
                    test_state = _build_vi_state_from_results(
                        optimizer_samples=test_candidate['optimizer_samples'],
                        parameter_samples=test_candidate['parameter_samples'],
                        iteration_results=test_candidate['iteration_results'],
                        observations=observations,
                        observations_covariance=observations_covariance,
                        variational_mean=test_variational_mean,
                        variational_log_std=test_variational_log_std,
                        covariance_regularization=covariance_regularization,
                        baseline_method=baseline_method,
                        gradient_method=gradient_method,
                        min_variational_std=min_variational_std,
                        max_variational_std=max_variational_std,
                        variational_correlation_cholesky=variational_correlation_cholesky,
                        elbo_scaling_factor=elbo_scaling_factor,
                )
                state = test_state
                step_size = min(step_size * step_size_growth_factor, max_step_size)
                line_search_standard_normal_cache = None

                accepted_elbo_history.append(state['elbo'])
                gradient_norm = _compute_gradient_norm(state, optimization_method)
                wall_time = time.time() - start_time
                print(
                    f'Iteration: {iteration}, Relative MSE: {state["mean_relative_mse"]:.5f}, '
                    f'ELBO: {state["elbo"]:.5f}, Step size: {step_size:.5e}, '
                    f'Gradient norm: {gradient_norm:.5f}, Samples: {line_search_sample_size}, '
                    f'Wall time: {wall_time:.5f}'
                )
                _print_vi_parameters(
                    variational_mean,
                    variational_log_std,
                    min_variational_std,
                    max_variational_std,
                    optimizer_samples=state['optimizer_samples'],
                    parameter_samples=state['parameter_samples'],
                )

                _save_vi_restart(
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
                    baseline_method,
                    optimization_method,
                    line_search_objective,
                    line_search_method,
                    line_search_nonmonotone_window,
                    line_search_armijo_coefficient,
                    line_search_uncertainty_sigma,
                    line_search_sample_growth_factor,
                    log_std_learning_rate_factor,
                )
                iteration += 1
            else:
                step_failed_counter += 1
                step_size /= step_size_decay_factor
                if line_search_objective == 'elbo':
                    print(
                        f'  Warning, lowering step size, Iteration: {iteration}, '
                        f'Relative MSE: {state["mean_relative_mse"]:.5f}, '
                        f'ELBO: {state["elbo"]:.5f}, Step size: {step_size:.5e}, '
                        f'Gradient norm: {gradient_norm:.5f}, '
                        f'Samples: {line_search_sample_size}'
                    )
                else:
                    print(
                        f'  Warning, lowering step size, Iteration: {iteration}, '
                        f'Relative MSE: {state["mean_relative_mse"]:.5f}, '
                        f'Test Relative MSE: {test_candidate["mean_relative_mse"]:.5f}, '
                        f'ELBO: {state["elbo"]:.5f}, Step size: {step_size:.5e}, '
                        f'Gradient norm: {gradient_norm:.5f}, '
                        f'Samples: {line_search_sample_size}'
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
    return variational_mean, variational_std, state['parameter_samples'], state['qois']
