r"""
Variational inference drivers for Gaussian posterior approximations.

This module provides derivative-free variational inference (VI) routines for
black-box forward models. An approximate posterior is represented by a
Gaussian variational family, and the optimizer updates the variational
parameters to maximize the evidence lower bound (ELBO).

.. rubric:: Theory

``run_vi`` solves a Gaussian variational inference problem by maximizing

.. math::

   \max_{\zeta}\;
   \mathbb{E}_{\theta \sim q(\theta;\zeta)}
   \left[
   \log p(y,\theta) - \log q(\theta;\zeta)
   \right],

where :math:`q(\theta;\zeta)` is the variational search distribution and
:math:`\zeta` denotes its parameters. In this implementation, the variational
family is Gaussian in either diagonal form or fixed-correlation multivariate
form, with the optimizer state stored as the variational mean and log standard
deviation.

For black-box forward models, the code uses the score-function
(REINFORCE/log-likelihood-trick) estimator instead of reparameterization
gradients through the PDE model:

.. math::

   \nabla_{\zeta}
   \mathbb{E}_{\theta \sim q(\theta;\zeta)}
   \left[\mathcal{L}(\theta;\zeta)\right]
   =
   \mathbb{E}_{\theta \sim q(\theta;\zeta)}
   \left[
   \mathcal{L}(\theta;\zeta)\,
   \nabla_{\zeta}\log q(\theta;\zeta)
   \right],

with

.. math::

   \mathcal{L}(\theta;\zeta)
   =
   \log p(y,\theta) - \log q(\theta;\zeta).

The expectation is approximated with Monte Carlo or randomized quasi-Monte
Carlo samples from the current variational distribution. Because only
:math:`\nabla_{\zeta}\log q` is needed, the forward model itself is treated as
derivative-free.

When ``optimizer_method="newton"``, the routine also forms a second-order
score-function estimator for curvature:

.. math::

   \nabla_{\zeta}^2
   \mathbb{E}_{\theta \sim q(\theta;\zeta)}
   \left[\mathcal{L}(\theta;\zeta)\right]
   =
   \mathbb{E}_{\theta \sim q(\theta;\zeta)}
   \left[
   \mathcal{L}(\theta;\zeta)
   \left(
   \nabla_{\zeta}\log q\,\nabla_{\zeta}\log q^{\top}
   + \nabla_{\zeta}^2 \log q
   \right)
   \right].

This is the curvature model used by the Newton update path. The implementation
supports a metric rescaling via ``newton_metric`` and either diagonal or full
projected Hessian solves via ``newton_hessian_type``.

.. rubric:: Variance Reduction

The gradient estimator optionally supports a baseline through
``baseline_method``. In particular, the leave-one-out option subtracts a
sample mean baseline that is independent of the current score term,
preserving unbiasedness while reducing Monte Carlo variance:

.. math::

   \widehat g_{\mathrm{LOO}}
   =
   \frac{1}{N}\sum_{i=1}^{N}
   \left(
   \mathcal{L}(\theta^{(i)};\zeta) - b^{(-i)}
   \right)
   \nabla_{\zeta}\log q(\theta^{(i)};\zeta),

where :math:`b^{(-i)}` is the mean ELBO over all samples except
:math:`\theta^{(i)}`.

.. rubric:: Gaussian Structure

For Gaussian priors and Gaussian variational families, the ELBO integrand
decomposes into

.. math::

   \log p(y,\theta)
   =
   \log p(y\mid\theta) + \log p_0(\theta),

so the forward-model dependence enters through the log-likelihood term, while
the prior and variational-density terms are handled analytically. This is why
the routine can remain derivative-free with respect to the forward model while
still using gradient- and Hessian-based updates in the variational
parameters.
"""

import numpy as np
import os
import re
import time
from scipy.stats import norm, qmc
from typing import Optional, Tuple
from romtools.hpc.dispatchers import BaseDispatcher, resolve_dispatcher
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import (
    GaussianParameterSpace,
    MultivariateGaussianParameterSpace,
)
from romtools.workflows.inverse._inverse_utils import run_vi_iteration
from romtools.workflows.inverse._inverse_utils import bound_samples
from romtools.workflows.inverse.vi_optimization_methods import (
    NewtonSolver,
    SteepestDescentSolver,
    VIGradientOptimizerConfig,
    VINewtonOptimizerConfig,
    VILegacyLineSearchConfig,
    VIStochasticNonmonotoneLineSearchConfig,
    _normalize_line_search_method,
    _normalize_line_search_objective,
    _normalize_newton_hessian_type,
    _normalize_newton_metric,
    _normalize_optimization_method,
    _resolve_line_search_config,
    _resolve_optimizer_config,
)


def _resolve_parameter_bounds(parameter_mins: np.ndarray,
                              parameter_maxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return parameter_mins, parameter_maxes


def _resolve_restart_file(restart_file: str) -> str:
    if restart_file is None:
        return None
    if not isinstance(restart_file, str):
        raise TypeError("restart_file must be a string or None.")
    return restart_file


def _prune_old_restart_files(absolute_vi_directory: str,
                             restart_files_to_keep: int,
                             dispatcher: Optional[BaseDispatcher] = None) -> None:
    if restart_files_to_keep is None:
        return
    if restart_files_to_keep < 1:
        return

    dispatcher = resolve_dispatcher(dispatcher)
    restart_entries = []
    for entry_name in dispatcher.list_dir(absolute_vi_directory):
        match = re.match(r"iteration_(\d+)$", entry_name)
        if match is None:
            continue
        restart_path = f"{absolute_vi_directory}/{entry_name}/restart.npz"
        if dispatcher.path_exists(restart_path):
            restart_entries.append((int(match.group(1)), restart_path))

    restart_entries.sort(key=lambda pair: pair[0])
    if len(restart_entries) <= restart_files_to_keep:
        return
    for _, restart_path in restart_entries[:-restart_files_to_keep]:
        dispatcher.remove(restart_path)


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


def _extract_gaussian_parameter_space(parameter_space,
                                      argument_name: str = 'parameter_space'):
    if isinstance(parameter_space, GaussianParameterSpace):
        parameter_names = list(parameter_space.get_names())
        gaussian_parameters = parameter_space._get_parameter_list()
        means = np.asarray([parameter._mean for parameter in gaussian_parameters], dtype=float)
        stds = np.asarray([parameter._std for parameter in gaussian_parameters], dtype=float)
        covariance = np.diag(stds ** 2)
        return parameter_names, means, covariance, 'diagonal'
    if isinstance(parameter_space, MultivariateGaussianParameterSpace):
        parameter_names = list(parameter_space.get_names())
        means = np.asarray(parameter_space._means, dtype=float)
        covariance = np.asarray(parameter_space._covariance, dtype=float)
        return parameter_names, means, covariance, 'multivariate'
    raise TypeError(
        f"{argument_name} must be GaussianParameterSpace or "
        "MultivariateGaussianParameterSpace."
    )


def _resolve_variational_initialization(prior_parameter_space,
                                        initial_variational_parameter_space=None):
    if initial_variational_parameter_space is None:
        initial_variational_parameter_space = prior_parameter_space
    return _extract_gaussian_parameter_space(
        initial_variational_parameter_space,
        argument_name='initial_variational_parameter_space',
    )


def _validate_gaussian_parameter_spaces(prior_parameter_space,
                                        initial_variational_parameter_space=None):
    prior_names, prior_mean, prior_covariance, prior_distribution = _extract_gaussian_parameter_space(
        prior_parameter_space,
        argument_name='prior_parameter_space',
    )
    (
        initial_names,
        initial_mean,
        initial_covariance,
        variational_distribution,
    ) = _resolve_variational_initialization(
        prior_parameter_space,
        initial_variational_parameter_space,
    )
    if prior_names != initial_names:
        raise ValueError(
            "prior_parameter_space and initial_variational_parameter_space must define the same "
            "parameter names in the same order."
        )
    if prior_distribution != variational_distribution:
        raise ValueError(
            "prior_parameter_space and initial_variational_parameter_space must use the same "
            "Gaussian family."
        )
    return (
        prior_names,
        prior_mean,
        prior_covariance,
        prior_distribution,
        initial_mean,
        initial_covariance,
        variational_distribution,
    )


def _compute_gaussian_log_density_data(covariance: np.ndarray):
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    covariance_diagonal = np.diag(covariance)
    if np.array_equal(covariance, np.diag(covariance_diagonal)):
        absolute_diagonal = np.abs(covariance_diagonal)
        cutoff = 1e-15 * np.max(absolute_diagonal)
        precision_diagonal = np.zeros_like(covariance_diagonal, dtype=float)
        nonzero_entries = absolute_diagonal > cutoff
        precision_diagonal[nonzero_entries] = 1.0 / covariance_diagonal[nonzero_entries]
        log_det = float(np.sum(np.log(covariance_diagonal)))
        return precision_diagonal, log_det

    sign, log_det = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        raise ValueError("Gaussian covariance must be positive definite.")
    return np.linalg.pinv(covariance), float(log_det)


def _compute_gaussian_log_densities(samples: np.ndarray,
                                    mean: np.ndarray,
                                    precision_operator: np.ndarray,
                                    covariance_log_det: float) -> np.ndarray:
    centered_samples = samples - mean[None, :]
    if precision_operator.ndim == 1:
        quadratic_terms = np.sum((centered_samples ** 2) * precision_operator[None, :], axis=1)
    else:
        weighted_samples = centered_samples @ precision_operator
        quadratic_terms = np.sum(centered_samples * weighted_samples, axis=1)
    dimensionality = mean.size
    normalizer = dimensionality * np.log(2.0 * np.pi) + covariance_log_det
    return -0.5 * (quadratic_terms + normalizer)


def _compute_log_transform_jacobian(optimizer_samples: np.ndarray,
                                    bounded_parameter_handling: str,
                                    parameter_mins: np.ndarray,
                                    parameter_maxes: np.ndarray,
                                    transform_interior_margin: float,
                                    transform_map: str) -> np.ndarray:
    if bounded_parameter_handling != 'transform':
        return np.zeros(optimizer_samples.shape[0], dtype=float)
    jacobian_diagonal = _compute_transform_jacobian_diagonal(
        optimizer_samples,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin,
        transform_map,
    )
    safe_diagonal = np.maximum(np.abs(jacobian_diagonal), 1e-300)
    return np.sum(np.log(safe_diagonal), axis=1)


def _compute_log_prior_and_joint_terms(log_likelihoods: np.ndarray,
                                       parameter_samples: np.ndarray,
                                       optimizer_samples: np.ndarray,
                                       prior_mean: np.ndarray,
                                       prior_precision_operator: np.ndarray,
                                       prior_covariance_log_det: float,
                                       bounded_parameter_handling: str,
                                       parameter_mins: np.ndarray,
                                       parameter_maxes: np.ndarray,
                                       transform_interior_margin: float,
                                       transform_map: str):
    log_priors = _compute_gaussian_log_densities(
        parameter_samples,
        prior_mean,
        prior_precision_operator,
        prior_covariance_log_det,
    )
    log_transform_jacobian = _compute_log_transform_jacobian(
        optimizer_samples,
        bounded_parameter_handling,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin,
        transform_map,
    )
    log_joint_terms = log_likelihoods + log_priors + log_transform_jacobian
    return log_priors, log_transform_jacobian, log_joint_terms


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


def _compute_state_elbo_standard_error(state, elbo_scaling_factor: float) -> float:
    return abs(elbo_scaling_factor) * _compute_standard_error(state['log_joint_terms'])


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


def _compute_variational_covariance(variational_std: np.ndarray,
                                    variational_correlation_cholesky: np.ndarray = None) -> np.ndarray:
    if variational_correlation_cholesky is None:
        return np.diag(variational_std ** 2)
    correlation = variational_correlation_cholesky @ variational_correlation_cholesky.transpose()
    return (variational_std[:, None] * correlation) * variational_std[None, :]


def _extract_variational_initialization(initial_variational_parameter_space):
    return _extract_gaussian_parameter_space(
        initial_variational_parameter_space,
        argument_name='initial_variational_parameter_space',
    )


def _initialize_variational_from_mean_cov(initial_variational_mean: np.ndarray,
                                          initial_variational_covariance: np.ndarray,
                                          variational_distribution: str,
                                          min_variational_std: float,
                                          max_variational_std: float):
    variational_mean = np.asarray(initial_variational_mean, dtype=float)
    variational_covariance = np.asarray(initial_variational_covariance, dtype=float)

    if variational_mean.ndim != 1:
        raise ValueError("initial_variational_mean must be a 1D array.")
    if variational_covariance.ndim != 2:
        raise ValueError("initial_variational_covariance must be a 2D array.")
    if variational_covariance.shape[0] != variational_covariance.shape[1]:
        raise ValueError("initial_variational_covariance must be square.")
    if variational_covariance.shape[0] != variational_mean.size:
        raise ValueError(
            "initial_variational_covariance shape must match initial_variational_mean size."
        )

    variational_covariance = 0.5 * (variational_covariance + variational_covariance.T)
    covariance_diagonal = np.diag(variational_covariance)
    if np.any(covariance_diagonal < 0.0):
        raise ValueError("initial_variational_covariance must have non-negative diagonal entries.")
    variational_std = np.sqrt(np.maximum(covariance_diagonal, 0.0))
    variational_std = np.maximum(variational_std, min_variational_std)
    variational_std = np.minimum(variational_std, max_variational_std)
    variational_log_std = np.log(variational_std)
    variational_log_std = _clip_variational_log_std(
        variational_log_std,
        min_variational_std,
        max_variational_std,
    )

    variational_correlation_cholesky = None
    if variational_distribution == 'multivariate':
        if variational_mean.size == 1:
            variational_correlation_cholesky = np.ones((1, 1))
        else:
            std_outer = variational_std[:, None] * variational_std[None, :]
            correlation = np.eye(variational_mean.size)
            nonzero_entries = std_outer > 0.0
            correlation[nonzero_entries] = variational_covariance[nonzero_entries] / std_outer[nonzero_entries]
            correlation = 0.5 * (correlation + correlation.T)
            np.fill_diagonal(correlation, 1.0)
            jitter = 1e-12
            while True:
                try:
                    variational_correlation_cholesky = np.linalg.cholesky(
                        correlation + jitter * np.eye(variational_mean.size)
                    )
                    break
                except np.linalg.LinAlgError:
                    if jitter >= 1e-2:
                        variational_correlation_cholesky = np.eye(variational_mean.size)
                        break
                    jitter *= 10.0

    return variational_mean, variational_log_std, variational_correlation_cholesky


def _write_iteration_stats_file(iteration_directory: str,
                                variational_mean: np.ndarray,
                                variational_log_std: np.ndarray,
                                min_variational_std: float,
                                max_variational_std: float,
                                variational_correlation_cholesky: np.ndarray,
                                elbo: float,
                                log_likelihoods: np.ndarray,
                                log_priors: np.ndarray,
                                mean_relative_mse: float,
                                wall_time: float,
                                cpu_time: float,
                                bounded_parameter_handling: str = 'clip',
                                parameter_mins: np.ndarray = None,
                                parameter_maxes: np.ndarray = None,
                                transform_interior_margin: float = 0.0,
                                transform_map: str = 'sigmoid',
                                dispatcher: Optional[BaseDispatcher] = None) -> None:
    dispatcher = resolve_dispatcher(dispatcher)
    dispatcher.create_empty_dir(iteration_directory)
    variational_std, _ = _compute_variational_std(
        variational_log_std,
        min_variational_std,
        max_variational_std,
    )
    variational_covariance = _compute_variational_covariance(
        variational_std,
        variational_correlation_cholesky,
    )
    mean_log_likelihood = float(np.mean(log_likelihoods))
    mean_log_prior = float(np.mean(log_priors))
    persisted_variational_mean = _get_persisted_variational_mean(
        variational_mean,
        bounded_parameter_handling,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin,
        transform_map,
    )
    stats_lines = [
        f"elbo: {float(elbo):.16e}",
        f"mean_relative_mse: {float(mean_relative_mse):.16e}",
        f"mean_log_likelihood: {mean_log_likelihood:.16e}",
        f"mean_log_prior: {mean_log_prior:.16e}",
        f"wall_time_seconds: {float(wall_time):.16e}",
        f"cpu_time_seconds: {float(cpu_time):.16e}",
        "variational_mean: "
        f"{np.array2string(persisted_variational_mean, precision=16, separator=', ')}",
        "variational_covariance: "
        f"{np.array2string(variational_covariance, precision=16, separator=', ')}",
    ]
    dispatcher.write_text(f"{iteration_directory}/stats.txt", "\n".join(stats_lines) + "\n")


def _get_persisted_variational_mean(variational_mean: np.ndarray,
                                    bounded_parameter_handling: str,
                                    parameter_mins: np.ndarray = None,
                                    parameter_maxes: np.ndarray = None,
                                    transform_interior_margin: float = 0.0,
                                    transform_map: str = 'sigmoid') -> np.ndarray:
    persisted_mean = variational_mean.copy()
    if bounded_parameter_handling == 'transform':
        persisted_mean = _transform_optimizer_to_parameter(
            variational_mean[None, :],
            parameter_mins,
            parameter_maxes,
            transform_interior_margin,
            transform_map,
        )[0]
    return persisted_mean


def _restore_variational_mean_from_restart(restart_data,
                                           bounded_parameter_handling: str,
                                           parameter_mins: np.ndarray = None,
                                           parameter_maxes: np.ndarray = None,
                                           transform_interior_margin: float = 0.0,
                                           transform_map: str = 'sigmoid') -> np.ndarray:
    persisted_variational_mean = restart_data['variational_mean']
    coordinate_system = None
    if 'variational_mean_coordinates' in restart_data:
        coordinate_system = str(restart_data['variational_mean_coordinates'].item()).strip().lower()
    if bounded_parameter_handling != 'transform':
        return persisted_variational_mean
    if coordinate_system != 'physical':
        raise ValueError(
            "Restart file variational_mean_coordinates must be 'physical'; "
            "older optimizer-coordinate restart files are not supported."
        )
    return _transform_parameter_to_optimizer(
        persisted_variational_mean[None, :],
        parameter_mins,
        parameter_maxes,
        transform_interior_margin,
        transform_map,
    )[0]


def _initialize_vi_history():
    return {
        'variational_mean': [],
        'variational_covariance': [],
        'relative_mse': [],
        'loglikelihood': [],
        'cpu_time_seconds': [],
    }


def _load_vi_history_from_restart(restart_data):
    vi_history = _initialize_vi_history()
    required_keys = (
        'vi_history_variational_mean',
        'vi_history_variational_covariance',
        'vi_history_relative_mse',
        'vi_history_loglikelihood',
        'vi_history_cpu_time_seconds',
    )
    if not all(key in restart_data for key in required_keys):
        return vi_history
    vi_history['variational_mean'] = [entry.copy() for entry in restart_data['vi_history_variational_mean']]
    vi_history['variational_covariance'] = [
        entry.copy() for entry in restart_data['vi_history_variational_covariance']
    ]
    vi_history['relative_mse'] = [float(entry) for entry in restart_data['vi_history_relative_mse']]
    vi_history['loglikelihood'] = [float(entry) for entry in restart_data['vi_history_loglikelihood']]
    vi_history['cpu_time_seconds'] = [float(entry) for entry in restart_data['vi_history_cpu_time_seconds']]
    return vi_history


def _append_vi_history(vi_history,
                       variational_mean: np.ndarray,
                       variational_log_std: np.ndarray,
                       min_variational_std: float,
                       max_variational_std: float,
                       variational_correlation_cholesky: np.ndarray,
                       mean_relative_mse: float,
                       log_likelihoods: np.ndarray,
                       cpu_time_seconds: float,
                       bounded_parameter_handling: str = 'clip',
                       parameter_mins: np.ndarray = None,
                       parameter_maxes: np.ndarray = None,
                       transform_interior_margin: float = 0.0,
                       transform_map: str = 'sigmoid') -> None:
    variational_std, _ = _compute_variational_std(
        variational_log_std,
        min_variational_std,
        max_variational_std,
    )
    variational_covariance = _compute_variational_covariance(
        variational_std,
        variational_correlation_cholesky,
    )
    history_variational_mean = _get_persisted_variational_mean(
        variational_mean,
        bounded_parameter_handling,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin,
        transform_map,
    )
    vi_history['variational_mean'].append(history_variational_mean)
    vi_history['variational_covariance'].append(variational_covariance.copy())
    vi_history['relative_mse'].append(float(mean_relative_mse))
    vi_history['loglikelihood'].append(float(np.mean(log_likelihoods)))
    vi_history['cpu_time_seconds'].append(float(cpu_time_seconds))


def _pack_vi_history(vi_history):
    mean_shape = None
    covariance_shape = None
    if vi_history['variational_mean']:
        mean_shape = vi_history['variational_mean'][0].shape
    if vi_history['variational_covariance']:
        covariance_shape = vi_history['variational_covariance'][0].shape
    return {
        'vi_history_variational_mean': (
            np.asarray(vi_history['variational_mean'])
            if vi_history['variational_mean']
            else np.empty((0,) + (() if mean_shape is None else mean_shape))
        ),
        'vi_history_variational_covariance': (
            np.asarray(vi_history['variational_covariance'])
            if vi_history['variational_covariance']
            else np.empty((0,) + (() if covariance_shape is None else covariance_shape))
        ),
        'vi_history_relative_mse': np.asarray(vi_history['relative_mse'], dtype=float),
        'vi_history_loglikelihood': np.asarray(vi_history['loglikelihood'], dtype=float),
        'vi_history_cpu_time_seconds': np.asarray(vi_history['cpu_time_seconds'], dtype=float),
    }


def _save_vi_history(absolute_vi_directory: str,
                     vi_history,
                     dispatcher: Optional[BaseDispatcher] = None) -> None:
    dispatcher = resolve_dispatcher(dispatcher)
    dispatcher.create_empty_dir(absolute_vi_directory)
    dispatcher.np_savez(
        f'{absolute_vi_directory}/history.npz',
        **_pack_vi_history(vi_history),
    )


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


def _normalize_sampling_method(sampling_method: str) -> str:
    method = sampling_method.strip().lower()
    if method in ('mc', 'montecarlo', 'monte_carlo'):
        return 'mc'
    if method in ('rqmc', 'randomized_qmc', 'randomized_quasi_monte_carlo'):
        return 'rqmc'
    raise ValueError(
        f"Unsupported sampling_method '{sampling_method}'. "
        "Supported options are 'mc' and 'rqmc'."
    )


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


def _normalize_transform_map(transform_map: str) -> str:
    normalized = transform_map.strip().lower()
    if normalized in ('sigmoid', 'logistic'):
        return 'sigmoid'
    if normalized in ('arctan', 'atan'):
        return 'arctan'
    raise ValueError(
        f"Unsupported transform_map '{transform_map}'. "
        "Supported options are 'sigmoid' and 'arctan'."
    )


def _transform_unit_from_optimizer(optimizer_values: np.ndarray,
                                   transform_map: str) -> np.ndarray:
    transform_map = _normalize_transform_map(transform_map)
    if transform_map == 'sigmoid':
        return _sigmoid(optimizer_values)
    return 0.5 + np.arctan(optimizer_values) / np.pi


def _inverse_transform_unit_to_optimizer(unit_values: np.ndarray,
                                         transform_map: str) -> np.ndarray:
    transform_map = _normalize_transform_map(transform_map)
    if transform_map == 'sigmoid':
        return _logit(unit_values)
    return np.tan(np.pi * (unit_values - 0.5))


def _compute_transform_jacobian_diagonal(optimizer_values: np.ndarray,
                                         parameter_mins: np.ndarray,
                                         parameter_maxes: np.ndarray,
                                         transform_interior_margin: float,
                                         transform_map: str = 'sigmoid') -> np.ndarray:
    margin_scale = 1.0 - 2.0 * transform_interior_margin
    if margin_scale <= 0.0:
        raise ValueError("transform_interior_margin must be in [0.0, 0.5).")
    transform_map = _normalize_transform_map(transform_map)
    bounded_unit = _transform_unit_from_optimizer(optimizer_values, transform_map)
    if transform_map == 'sigmoid':
        base_jacobian = bounded_unit * (1.0 - bounded_unit)
    else:
        base_jacobian = 1.0 / (np.pi * (1.0 + optimizer_values ** 2))
    return (
        (parameter_maxes - parameter_mins)
        * margin_scale
        * base_jacobian
    )


def _enforce_variational_log_std_bounds(variational_mean: np.ndarray,
                                        variational_log_std: np.ndarray,
                                        min_variational_std: float,
                                        max_variational_std: float,
                                        bounded_parameter_handling: str,
                                        parameter_mins: np.ndarray,
                                        parameter_maxes: np.ndarray,
                                        transform_interior_margin: float,
                                        min_physical_variational_std_fraction: float,
                                        transform_map: str = 'sigmoid') -> np.ndarray:
    clipped_log_std = _clip_variational_log_std(
        variational_log_std,
        min_variational_std,
        max_variational_std,
    )
    if bounded_parameter_handling != 'transform' or min_physical_variational_std_fraction <= 0.0:
        return clipped_log_std

    if parameter_mins is None or parameter_maxes is None:
        raise ValueError("parameter bounds are required for bounded_parameter_handling='transform'.")

    parameter_ranges = parameter_maxes - parameter_mins
    physical_std_floor = min_physical_variational_std_fraction * parameter_ranges
    jacobian_diagonal = _compute_transform_jacobian_diagonal(
        variational_mean,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin,
        transform_map,
    )
    jacobian_diagonal = np.maximum(np.abs(jacobian_diagonal), 1e-16)
    variational_std = np.exp(clipped_log_std)
    required_optimizer_std = physical_std_floor / jacobian_diagonal
    adjusted_std = np.maximum(variational_std, required_optimizer_std)
    adjusted_std = np.clip(adjusted_std, min_variational_std, max_variational_std)
    return np.log(adjusted_std)


def _transform_optimizer_to_parameter(optimizer_samples: np.ndarray,
                                      parameter_mins: np.ndarray,
                                      parameter_maxes: np.ndarray,
                                      transform_interior_margin: float = 0.0,
                                      transform_map: str = 'sigmoid') -> np.ndarray:
    margin_scale = 1.0 - 2.0 * transform_interior_margin
    if margin_scale <= 0.0:
        raise ValueError("transform_interior_margin must be in [0.0, 0.5).")
    bounded_unit = transform_interior_margin + margin_scale * _transform_unit_from_optimizer(
        optimizer_samples,
        transform_map,
    )
    return parameter_mins[None, :] + (parameter_maxes - parameter_mins)[None, :] * bounded_unit


def _transform_parameter_to_optimizer(parameter_samples: np.ndarray,
                                      parameter_mins: np.ndarray,
                                      parameter_maxes: np.ndarray,
                                      transform_interior_margin: float = 0.0,
                                      transform_map: str = 'sigmoid') -> np.ndarray:
    parameter_ranges = parameter_maxes - parameter_mins
    unit_values = (parameter_samples - parameter_mins[None, :]) / parameter_ranges[None, :]
    margin_scale = 1.0 - 2.0 * transform_interior_margin
    if margin_scale <= 0.0:
        raise ValueError("transform_interior_margin must be in [0.0, 0.5).")
    unit_values = (unit_values - transform_interior_margin) / margin_scale
    unit_values = np.clip(unit_values, 1e-12, 1.0 - 1e-12)
    return _inverse_transform_unit_to_optimizer(unit_values, transform_map)


def _convert_physical_moments_to_optimizer_moments(physical_mean: np.ndarray,
                                                   physical_covariance: np.ndarray,
                                                   bounded_parameter_handling: str,
                                                   parameter_mins: np.ndarray,
                                                   parameter_maxes: np.ndarray,
                                                   transform_interior_margin: float,
                                                   transform_map: str = 'sigmoid'):
    if bounded_parameter_handling != 'transform':
        return physical_mean.copy(), physical_covariance.copy()

    optimizer_mean = _transform_parameter_to_optimizer(
        physical_mean[None, :],
        parameter_mins,
        parameter_maxes,
        transform_interior_margin,
        transform_map,
    )[0]
    jacobian_diagonal = _compute_transform_jacobian_diagonal(
        optimizer_mean,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin,
        transform_map,
    )
    inverse_jacobian_diagonal = np.zeros_like(jacobian_diagonal)
    nonzero_jacobian = np.abs(jacobian_diagonal) > 1e-15
    inverse_jacobian_diagonal[nonzero_jacobian] = 1.0 / jacobian_diagonal[nonzero_jacobian]
    optimizer_covariance = (
        inverse_jacobian_diagonal[:, None]
        * physical_covariance
        * inverse_jacobian_diagonal[None, :]
    )
    optimizer_covariance = 0.5 * (optimizer_covariance + optimizer_covariance.T)
    return optimizer_mean, optimizer_covariance


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


def _print_gradient_signal_to_noise_ratio(state) -> None:
    gradient_signal_to_noise_ratio = state.get('gradient_signal_to_noise_ratio', np.nan)
    print(f'  Gradient SNR:    {gradient_signal_to_noise_ratio:.5f}')


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

    gradient_terms_mean = centered_weights_mean * score_mean
    gradient_terms_log_std = centered_weights_log_std * score_log_std
    gradient_mean = np.mean(gradient_terms_mean, axis=0)
    gradient_log_std = np.mean(gradient_terms_log_std, axis=0) + elbo_scaling_factor
    gradient_standard_error = np.concatenate([
        _compute_component_standard_error(gradient_terms_mean),
        _compute_component_standard_error(gradient_terms_log_std),
    ])
    gradient_signal_to_noise_ratio = _compute_gradient_signal_to_noise_ratio(
        np.concatenate([gradient_mean, gradient_log_std]),
        gradient_standard_error,
    )
    return gradient_mean, gradient_log_std, baseline_mean, baseline_log_std, gradient_signal_to_noise_ratio


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
                         newton_hessian_type: str = 'diagonal',
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
    solver = NewtonSolver(
        regularization=newton_regularization,
        hessian_type=newton_hessian_type,
    )
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


def _draw_standard_normal_samples(sample_size: int,
                                  dimensionality: int,
                                  sampling_method: str) -> np.ndarray:
    sampling_method = _normalize_sampling_method(sampling_method)
    if sampling_method == 'mc':
        return np.random.randn(sample_size, dimensionality)
    sobol_seed = int(np.random.randint(0, np.iinfo(np.uint32).max))
    uniform_samples = qmc.Sobol(
        d=dimensionality,
        scramble=True,
        seed=sobol_seed,
    ).random(n=sample_size)
    uniform_samples = np.clip(uniform_samples, 1e-12, 1.0 - 1e-12)
    return norm.ppf(uniform_samples)


def _draw_parameter_samples(variational_mean: np.ndarray,
                            variational_log_std: np.ndarray,
                            sample_size: int,
                            min_variational_std: float,
                            max_variational_std: float,
                            bounded_parameter_handling: str,
                            parameter_mins: np.ndarray,
                            parameter_maxes: np.ndarray,
                            transform_interior_margin: float = 0.0,
                            transform_map: str = 'sigmoid',
                            standard_normal_samples: np.ndarray = None,
                            variational_correlation_cholesky: np.ndarray = None,
                            sampling_method: str = 'mc'):
    bounded_parameter_handling = _normalize_bounded_parameter_handling(bounded_parameter_handling)
    variational_std, _ = _compute_variational_std(
        variational_log_std,
        min_variational_std,
        max_variational_std,
    )
    sampling_method = _normalize_sampling_method(sampling_method)
    if standard_normal_samples is None:
        standard_normal_samples = _draw_standard_normal_samples(
            sample_size,
            variational_mean.size,
            sampling_method,
        )
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
            transform_interior_margin,
            transform_map,
        )
    else:
        parameter_samples = bound_samples(optimizer_samples, parameter_mins, parameter_maxes)
    return optimizer_samples, parameter_samples


def _run_vi_iteration_samples(model: QoiModel,
                              observations: np.ndarray,
                              run_directory_base: str,
                              parameter_names,
                              parameter_samples: np.ndarray,
                              evaluation_concurrency: int,
                              dispatcher: Optional[BaseDispatcher] = None):
    return run_vi_iteration(
        model,
        observations,
        run_directory_base,
        parameter_names,
        parameter_samples,
        evaluation_concurrency,
        dispatcher,
    )


def _build_vi_state_from_results(optimizer_samples: np.ndarray,
                                 parameter_samples: np.ndarray,
                                 iteration_results,
                                 observations: np.ndarray,
                                 observations_covariance: np.ndarray,
                                 variational_mean: np.ndarray,
                                 variational_log_std: np.ndarray,
                                 prior_mean: np.ndarray,
                                 prior_precision_operator: np.ndarray,
                                 prior_covariance_log_det: float,
                                 covariance_regularization: float,
                                 baseline_method: str,
                                 gradient_method: str,
                                 bounded_parameter_handling: str,
                                 min_variational_std: float,
                                 max_variational_std: float,
                                 parameter_mins: np.ndarray,
                                 parameter_maxes: np.ndarray,
                                 transform_interior_margin: float = 0.0,
                                 transform_map: str = 'sigmoid',
                                 variational_correlation_cholesky: np.ndarray = None,
                                 elbo_scaling_factor: float = 1.0,
                                 log_likelihood_precision_operator: np.ndarray = None):
    qois = iteration_results['qois']
    mean_qoi = iteration_results['mean-qoi']
    errors = iteration_results['errors']

    log_likelihoods, misfits = _compute_log_likelihoods(
        errors,
        observations_covariance,
        covariance_regularization,
        precision_operator=log_likelihood_precision_operator,
    )
    log_priors, log_transform_jacobian, log_joint_terms = _compute_log_prior_and_joint_terms(
        log_likelihoods,
        parameter_samples,
        optimizer_samples,
        prior_mean,
        prior_precision_operator,
        prior_covariance_log_det,
        bounded_parameter_handling,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin,
        transform_map,
    )
    scaled_log_joint_terms = elbo_scaling_factor * log_joint_terms
    relative_mses = _compute_relative_mse(errors, observations)

    variational_std, variational_log_std = _compute_variational_std(
        variational_log_std,
        min_variational_std,
        max_variational_std,
    )
    (
        gradient_mean,
        gradient_log_std,
        baseline_mean,
        baseline_log_std,
        gradient_signal_to_noise_ratio,
    ) = _compute_reinforce_gradients(
        optimizer_samples,
        variational_mean,
        variational_std,
        scaled_log_joint_terms,
        baseline_method,
        variational_correlation_cholesky,
        elbo_scaling_factor,
    )
    hessian_diagonal_mean, hessian_diagonal_log_std = _compute_reinforce_hessian_diagonal(
        optimizer_samples,
        variational_mean,
        variational_std,
        scaled_log_joint_terms,
        baseline_method,
        variational_correlation_cholesky,
    )
    hessian_full = _compute_reinforce_hessian_full(
        optimizer_samples,
        variational_mean,
        variational_std,
        scaled_log_joint_terms,
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
    elbo = np.mean(scaled_log_joint_terms) + entropy

    state = {
        'optimizer_samples': optimizer_samples,
        'parameter_samples': parameter_samples,
        'qois': qois,
        'mean_qoi': mean_qoi,
        'errors': errors,
        'log_likelihoods': log_likelihoods,
        'log_priors': log_priors,
        'log_joint_terms': log_joint_terms,
        'log_transform_jacobian': log_transform_jacobian,
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
        'gradient_signal_to_noise_ratio': gradient_signal_to_noise_ratio,
    }
    return state


def _evaluate_vi_state(model: QoiModel,
                       observations: np.ndarray,
                       observations_covariance: np.ndarray,
                       run_directory_base: str,
                       parameter_names,
                       variational_mean: np.ndarray,
                       variational_log_std: np.ndarray,
                       prior_mean: np.ndarray,
                       prior_precision_operator: np.ndarray,
                       prior_covariance_log_det: float,
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
                       transform_interior_margin: float = 0.0,
                       transform_map: str = 'sigmoid',
                       variational_correlation_cholesky: np.ndarray = None,
                       elbo_scaling_factor: float = 1.0,
                       log_likelihood_precision_operator: np.ndarray = None,
                       sampling_method: str = 'mc',
                       dispatcher: Optional[BaseDispatcher] = None):
    optimizer_samples, parameter_samples = _draw_parameter_samples(
        variational_mean,
        variational_log_std,
        sample_size,
        min_variational_std,
        max_variational_std,
        bounded_parameter_handling,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin=transform_interior_margin,
        transform_map=transform_map,
        variational_correlation_cholesky=variational_correlation_cholesky,
        sampling_method=sampling_method,
    )
    iteration_results = _run_vi_iteration_samples(
        model,
        observations,
        run_directory_base,
        parameter_names,
        parameter_samples,
        evaluation_concurrency,
        dispatcher,
    )
    return _build_vi_state_from_results(
        optimizer_samples=optimizer_samples,
        parameter_samples=parameter_samples,
        iteration_results=iteration_results,
        observations=observations,
        observations_covariance=observations_covariance,
        variational_mean=variational_mean,
        variational_log_std=variational_log_std,
        prior_mean=prior_mean,
        prior_precision_operator=prior_precision_operator,
        prior_covariance_log_det=prior_covariance_log_det,
        covariance_regularization=covariance_regularization,
        baseline_method=baseline_method,
        gradient_method=gradient_method,
        bounded_parameter_handling=bounded_parameter_handling,
        min_variational_std=min_variational_std,
        max_variational_std=max_variational_std,
        parameter_mins=parameter_mins,
        parameter_maxes=parameter_maxes,
        transform_interior_margin=transform_interior_margin,
        transform_map=transform_map,
        variational_correlation_cholesky=variational_correlation_cholesky,
        elbo_scaling_factor=elbo_scaling_factor,
        log_likelihood_precision_operator=log_likelihood_precision_operator,
    )

def _evaluate_vi_candidate_for_line_search(model: QoiModel,
                                           observations: np.ndarray,
                                           observations_covariance: np.ndarray,
                                           run_directory_base: str,
                                           parameter_names,
                                           variational_mean: np.ndarray,
                                           variational_log_std: np.ndarray,
                                           prior_mean: np.ndarray,
                                           prior_precision_operator: np.ndarray,
                                           prior_covariance_log_det: float,
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
                                           transform_interior_margin: float,
                                           transform_map: str,
                                           line_search_objective: str,
                                           standard_normal_samples: np.ndarray = None,
                                           variational_correlation_cholesky: np.ndarray = None,
                                           elbo_scaling_factor: float = 1.0,
                                           log_likelihood_precision_operator: np.ndarray = None,
                                           dispatcher: Optional[BaseDispatcher] = None):
    optimizer_samples, parameter_samples = _draw_parameter_samples(
        variational_mean,
        variational_log_std,
        sample_size,
        min_variational_std,
        max_variational_std,
        bounded_parameter_handling,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin,
        transform_map,
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
        dispatcher,
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
            prior_mean=prior_mean,
            prior_precision_operator=prior_precision_operator,
            prior_covariance_log_det=prior_covariance_log_det,
            covariance_regularization=covariance_regularization,
            baseline_method=baseline_method,
            gradient_method=gradient_method,
            bounded_parameter_handling=bounded_parameter_handling,
            min_variational_std=min_variational_std,
            max_variational_std=max_variational_std,
            parameter_mins=parameter_mins,
            parameter_maxes=parameter_maxes,
            transform_interior_margin=transform_interior_margin,
            transform_map=transform_map,
            variational_correlation_cholesky=variational_correlation_cholesky,
            elbo_scaling_factor=elbo_scaling_factor,
            log_likelihood_precision_operator=log_likelihood_precision_operator,
        )
    return candidate


def _save_vi_restart(restart_path: str,
                     state,
                     variational_mean: np.ndarray,
                     variational_log_std: np.ndarray,
                     variational_distribution: str,
                     prior_mean: np.ndarray,
                     prior_covariance: np.ndarray,
                     variational_correlation_cholesky: np.ndarray,
                     elbo_scaling_factor: float,
                     elbo_relative_tolerance: float,
                     initial_elbo_reference: float,
                     iteration: int,
                     step_size: float,
                     bounded_parameter_handling: str,
                     transform_map: str,
                     baseline_method: str,
                     optimization_method: str,
                     line_search_objective: str,
                     line_search_method: str,
                     line_search_nonmonotone_window: int = None,
                     line_search_armijo_coefficient: float = None,
                     line_search_uncertainty_sigma: float = None,
                     line_search_sample_growth_factor: float = None,
                     log_std_learning_rate_factor: float = None,
                     parameter_mins: np.ndarray = None,
                     parameter_maxes: np.ndarray = None,
                     transform_interior_margin: float = 0.0,
                     vi_history=None,
                     sampling_method: str = None,
                     dispatcher: Optional[BaseDispatcher] = None):
    persisted_variational_mean = _get_persisted_variational_mean(
        variational_mean,
        bounded_parameter_handling,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin,
        transform_map,
    )
    save_data = dict(
        log_likelihoods=state['log_likelihoods'],
        log_priors=state['log_priors'],
        mean_relative_mse=float(state['mean_relative_mse']),
        variational_mean=persisted_variational_mean,
        variational_mean_coordinates='physical',
        variational_log_std=variational_log_std,
        variational_distribution=variational_distribution,
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
        elbo_scaling_factor=elbo_scaling_factor,
        elbo_relative_tolerance=(
            np.nan if elbo_relative_tolerance is None else float(elbo_relative_tolerance)
        ),
        initial_elbo_reference=float(initial_elbo_reference),
        iteration=iteration,
        step_size=step_size,
        bounded_parameter_handling=bounded_parameter_handling,
        transform_map=transform_map,
        baseline_method=baseline_method,
        optimization_method=optimization_method,
        line_search_objective=line_search_objective,
        line_search_method=line_search_method,
        line_search_sample_growth_factor=line_search_sample_growth_factor,
        log_std_learning_rate_factor=log_std_learning_rate_factor,
        sampling_method=sampling_method,
        rng_state=np.array(np.random.get_state(), dtype=object),
    )
    if line_search_method == 'stochastic_nonmonotone':
        save_data['line_search_nonmonotone_window'] = line_search_nonmonotone_window
        save_data['line_search_armijo_coefficient'] = line_search_armijo_coefficient
        save_data['line_search_uncertainty_sigma'] = line_search_uncertainty_sigma
    if vi_history is not None:
        save_data.update(_pack_vi_history(vi_history))
    if variational_correlation_cholesky is not None:
        save_data['variational_correlation_cholesky'] = variational_correlation_cholesky
    resolve_dispatcher(dispatcher).np_savez(restart_path, **save_data)


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
                            newton_hessian_type: str,
                            covariance_regularization: float,
                            restart_files_to_keep: int,
                            observations_covariance: np.ndarray,
                            observations: np.ndarray,
                            elbo_scaling_factor: float,
                            elbo_relative_tolerance: float,
                            sampling_method: str,
                            transform_interior_margin: float,
                            transform_map: str,
                            min_physical_variational_std_fraction: float,
                            prior_parameter_space,
                            initial_variational_parameter_space,
                            restart_file: str,
                            parameter_mins: np.ndarray,
                            parameter_maxes: np.ndarray,
                            bounded_parameter_handling: str,
                            dispatcher: Optional[BaseDispatcher] = None) -> None:
    dispatcher = resolve_dispatcher(dispatcher)
    dispatcher.require_absolute_path(absolute_vi_directory)
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
    assert min_physical_variational_std_fraction >= 0.0, (
        "min_physical_variational_std_fraction must be non-negative"
    )
    assert max_log_std_update > 0.0, "max_log_std_update must be positive"
    assert newton_regularization > 0.0, "newton_regularization must be positive"
    _normalize_newton_hessian_type(newton_hessian_type)
    assert covariance_regularization >= 0.0, "covariance_regularization must be non-negative"
    assert restart_files_to_keep >= 1, "restart_files_to_keep must be >= 1"
    assert observations_covariance.shape[0] == observations_covariance.shape[1], (
        "observations_covariance must be square"
    )
    assert observations_covariance.shape[0] == observations.size, (
        "observations_covariance shape must match observations size"
    )
    assert elbo_scaling_factor > 0.0, "elbo_scaling_factor must be positive"
    if elbo_relative_tolerance is not None:
        assert elbo_relative_tolerance >= 0.0, "elbo_relative_tolerance must be non-negative"
    _normalize_sampling_method(sampling_method)
    _normalize_transform_map(transform_map)

    parameter_names, _, _, _, initial_variational_mean, initial_variational_covariance, _ = (
        _validate_gaussian_parameter_spaces(
            prior_parameter_space,
            initial_variational_parameter_space,
        )
    )
    parameter_dimensionality = np.asarray(initial_variational_mean).size
    assert len(parameter_names) > 0, "prior_parameter_space must define at least one parameter"
    covariance = np.asarray(initial_variational_covariance)
    assert covariance.ndim == 2, "initial variational covariance must be a 2D array"
    assert covariance.shape[0] == covariance.shape[1], (
        "initial variational covariance must be square"
    )
    assert covariance.shape[0] == parameter_dimensionality, (
        "initial variational covariance shape must match initial variational mean size"
    )
    if parameter_mins is not None:
        assert np.size(parameter_mins) == parameter_dimensionality, (
            f"parameter_mins of size {np.size(parameter_mins)} is inconsistent with "
            f"the variational dimensionality of size {parameter_dimensionality}"
        )
    if parameter_maxes is not None:
        assert np.size(parameter_maxes) == parameter_dimensionality, (
            f"parameter_maxes of size {np.size(parameter_maxes)} is inconsistent with "
            f"the variational dimensionality of size {parameter_dimensionality}"
        )
    if bounded_parameter_handling == 'transform':
        assert 0.0 <= transform_interior_margin < 0.5, (
            "transform_interior_margin must be in [0.0, 0.5)"
        )
        assert parameter_mins is not None, "parameter_mins must be provided for bounded_parameter_handling='transform'"
        assert parameter_maxes is not None, "parameter_maxes must be provided for bounded_parameter_handling='transform'"
        assert np.size(parameter_mins) == parameter_dimensionality, (
            f"parameter_mins of size {np.size(parameter_mins)} is inconsistent with "
            f"the variational dimensionality of size {parameter_dimensionality}"
        )
        assert np.size(parameter_maxes) == parameter_dimensionality, (
            f"parameter_maxes of size {np.size(parameter_maxes)} is inconsistent with "
            f"the variational dimensionality of size {parameter_dimensionality}"
        )
        assert np.all(parameter_maxes > parameter_mins), (
            "All parameter_maxes entries must be greater than parameter_mins entries "
            "for bounded_parameter_handling='transform'"
        )


def run_vi(model: QoiModel,
           prior_parameter_space,
           observations: np.ndarray,
           observations_covariance: np.ndarray,
           parameter_mins: np.ndarray = None,
           parameter_maxes: np.ndarray = None,
           initial_variational_parameter_space=None,
           restart_file: str = None,
           optimizer_method: str = 'gradient',
           optimizer_config=None,
           line_search_method: str = 'stochastic_nonmonotone',
           line_search_config=None,
           absolute_vi_directory: str = os.getcwd() + "/work/",
           sample_size: int = 30,
           random_seed: int = 1,
           sampling_method: str = 'mc',
           evaluation_concurrency=1,
           covariance_regularization: float = 1e-8,
           restart_files_to_keep: int = 10,
           elbo_scaling_factor='diag_mean',
           elbo_relative_tolerance: float = None,
           baseline_method: str = None,
           bounded_parameter_handling: str = 'transform',
           transform_interior_margin: float = 1e-6,
           transform_map: str = 'sigmoid',
           min_physical_variational_std_fraction: float = 1e-6,
           dispatcher: Optional[BaseDispatcher] = None):
    '''
    Run Gaussian variational inference with score-function gradients.

    This routine approximates the posterior with a Gaussian variational family
    (diagonal or fixed-correlation multivariate) and updates its mean and
    log-standard deviation using REINFORCE. An optional baseline can be
    enabled to reduce gradient estimator variance.

    Args:
        model: QoiModel to evaluate at sampled parameters.
        prior_parameter_space: Either GaussianParameterSpace (diagonal VI)
            or MultivariateGaussianParameterSpace (multivariate VI). Defines
            the Bayesian prior in physical parameter space.
        observations: Observed QoI vector.
        observations_covariance: Observation covariance matrix.
        parameter_mins: Optional lower bounds on parameters.
        parameter_maxes: Optional upper bounds on parameters.
        initial_variational_parameter_space: Optional Gaussian initializer for
            the variational state in physical parameter space. Defaults to the
            prior moments.
        restart_file: Optional restart file path. Restart files written by this
            routine store `variational_mean` in physical coordinates.
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
        sampling_method: Sampling method for variational draws. Supported
            options are 'mc' and 'rqmc'.
        evaluation_concurrency: Concurrent model evaluations per iteration.
        covariance_regularization: Diagonal regularization for covariance inversion.
        restart_files_to_keep: Number of most-recent restart files to retain
            under `absolute_vi_directory`. Older restart files are removed.
        elbo_scaling_factor: Positive scalar that multiplies the ELBO objective,
            or a string mode. Supported string modes are:
            'diag_mean' (or 'auto'): mean(diag(observations_covariance)),
            'diag_trace': sum(diag(observations_covariance)).
        elbo_relative_tolerance: Optional non-negative relative tolerance for
            ELBO improvement relative to the initial ELBO at the initial
            variational guess. When set, VI stops when
            `elbo_current / (elbo_initial + 1e-16)` is less than or equal to
            this tolerance.
        baseline_method: Baseline used in REINFORCE gradient estimation.
            Supported options are 'none', 'loo', and 'optimal'.
        bounded_parameter_handling: Parameter bounds handling. Supported options
            are 'clip' and 'transform'.
        transform_interior_margin: Margin used by bounded_parameter_handling='transform'
            to keep mapped samples away from exact bounds.
        transform_map: Transform used by bounded_parameter_handling='transform'.
            Supported options are 'sigmoid' and 'arctan'.
        min_physical_variational_std_fraction: Minimum physical-space
            variational standard deviation as a fraction of each parameter range
            when bounded_parameter_handling='transform'.
        dispatcher: Optional (defaults to None, which instantiates a
            LocalDispatcher). Pass a RemoteDispatcher to send model evaluations
            to the configured remote host (e.g. an HPC cluster).

    Returns:
        Tuple of (variational_mean, variational_std, parameter_samples, qois).
    '''
    dispatcher = resolve_dispatcher(dispatcher)
    dispatcher.require_supported_concurrency(evaluation_concurrency)

    start_time = time.time()
    start_cpu_time = time.process_time()
    parameter_mins, parameter_maxes = _resolve_parameter_bounds(
        parameter_mins,
        parameter_maxes,
    )
    restart_file = _resolve_restart_file(restart_file)
    optimization_method, resolved_optimizer_config = _resolve_optimizer_config(
        optimizer_method,
        optimizer_config,
        VIGradientOptimizerConfig(),
        VINewtonOptimizerConfig(),
    )
    line_search_method, resolved_line_search_config = _resolve_line_search_config(
        line_search_method,
        line_search_config,
        VILegacyLineSearchConfig(),
        VIStochasticNonmonotoneLineSearchConfig(),
    )

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
    newton_hessian_type = _normalize_newton_hessian_type(newton_defaults.newton_hessian_type)
    if optimization_method == 'newton':
        newton_metric = _normalize_newton_metric(resolved_optimizer_config.newton_metric)
        newton_regularization = resolved_optimizer_config.newton_regularization
        newton_hessian_type = _normalize_newton_hessian_type(
            resolved_optimizer_config.newton_hessian_type
        )

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
    transform_map = _normalize_transform_map(transform_map)
    if baseline_method is None:
        baseline_method = 'loo'
    baseline_method = _normalize_baseline_method(baseline_method)
    sampling_method = _normalize_sampling_method(sampling_method)
    (
        parameter_names,
        prior_mean,
        prior_covariance,
        _,
        initial_variational_mean,
        initial_variational_covariance,
        variational_distribution,
    ) = _validate_gaussian_parameter_spaces(
        prior_parameter_space,
        initial_variational_parameter_space,
    )
    variational_distribution = _normalize_variational_distribution(variational_distribution)
    line_search_objective = _normalize_line_search_objective(line_search_objective)
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
        newton_hessian_type=newton_hessian_type,
        covariance_regularization=covariance_regularization,
        restart_files_to_keep=restart_files_to_keep,
        observations_covariance=observations_covariance,
        observations=observations,
        elbo_scaling_factor=elbo_scaling_factor,
        elbo_relative_tolerance=elbo_relative_tolerance,
        sampling_method=sampling_method,
        transform_interior_margin=transform_interior_margin,
        transform_map=transform_map,
        min_physical_variational_std_fraction=min_physical_variational_std_fraction,
        prior_parameter_space=prior_parameter_space,
        initial_variational_parameter_space=initial_variational_parameter_space,
        restart_file=restart_file,
        parameter_mins=parameter_mins,
        parameter_maxes=parameter_maxes,
        bounded_parameter_handling=bounded_parameter_handling,
        dispatcher=dispatcher,
    )
    log_likelihood_precision_operator = _compute_log_likelihood_precision_operator(
        observations_covariance,
        covariance_regularization,
    )
    prior_precision_operator, prior_covariance_log_det = _compute_gaussian_log_density_data(
        prior_covariance,
    )
    variational_correlation_cholesky = None
    vi_history = _initialize_vi_history()

    if restart_file is None:
        np.random.seed(random_seed)
        iteration = 0
        step_size = min(initial_step_size, max_step_size)
        initial_optimizer_mean, initial_optimizer_covariance = (
            _convert_physical_moments_to_optimizer_moments(
                initial_variational_mean,
                initial_variational_covariance,
                bounded_parameter_handling,
                parameter_mins,
                parameter_maxes,
                transform_interior_margin,
                transform_map,
            )
        )

        (
            variational_mean,
            variational_log_std,
            variational_correlation_cholesky,
        ) = _initialize_variational_from_mean_cov(
            initial_optimizer_mean,
            initial_optimizer_covariance,
            variational_distribution,
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
            transform_map,
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
            prior_mean,
            prior_precision_operator,
            prior_covariance_log_det,
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
            transform_interior_margin,
            transform_map,
            variational_correlation_cholesky,
            elbo_scaling_factor,
            log_likelihood_precision_operator,
            sampling_method,
            dispatcher,
        )
        initial_elbo_reference = float(state['elbo'])
    else:
        restart_data = np.load(restart_file, allow_pickle=True)
        vi_history = _load_vi_history_from_restart(restart_data)
        if 'rng_state' in restart_data:
            np.random.set_state(tuple(restart_data['rng_state'].tolist()))
        else:
            np.random.seed(random_seed)
        iteration = int(restart_data['iteration'])
        step_size = min(float(restart_data['step_size']), max_step_size)
        variational_mean = _restore_variational_mean_from_restart(
            restart_data,
            bounded_parameter_handling,
            parameter_mins,
            parameter_maxes,
            transform_interior_margin,
            transform_map,
        )
        variational_log_std = _clip_variational_log_std(
            restart_data['variational_log_std'],
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
            transform_map,
        )
        if 'bounded_parameter_handling' in restart_data:
            restart_bounded_parameter_handling = str(restart_data['bounded_parameter_handling'].item())
            if restart_bounded_parameter_handling != bounded_parameter_handling:
                raise ValueError(
                    "Restart file bounded_parameter_handling does not match current run."
                )
        if 'transform_map' in restart_data:
            restart_transform_map = _normalize_transform_map(
                str(restart_data['transform_map'].item())
            )
            if restart_transform_map != transform_map:
                raise ValueError("Restart file transform_map does not match current run.")
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
        if 'elbo_relative_tolerance' in restart_data:
            restart_elbo_relative_tolerance = float(restart_data['elbo_relative_tolerance'])
            restart_tolerance_is_none = np.isnan(restart_elbo_relative_tolerance)
            current_tolerance_is_none = elbo_relative_tolerance is None
            if restart_tolerance_is_none != current_tolerance_is_none:
                raise ValueError(
                    "Restart file elbo_relative_tolerance does not match current run."
                )
            if (
                not current_tolerance_is_none
                and not np.isclose(restart_elbo_relative_tolerance, elbo_relative_tolerance)
            ):
                raise ValueError(
                    "Restart file elbo_relative_tolerance does not match current run."
                )
        if 'sampling_method' in restart_data:
            restart_sampling_method = _normalize_sampling_method(
                str(restart_data['sampling_method'].item())
            )
            if restart_sampling_method != sampling_method:
                raise ValueError("Restart file sampling_method does not match current run.")
        if 'prior_mean' in restart_data and not np.allclose(restart_data['prior_mean'], prior_mean):
            raise ValueError("Restart file prior_mean does not match current run.")
        if 'prior_covariance' in restart_data and not np.allclose(
            restart_data['prior_covariance'],
            prior_covariance,
        ):
            raise ValueError("Restart file prior_covariance does not match current run.")

        run_directory_base = f'{absolute_vi_directory}/iteration_{iteration}/run_'
        state = _evaluate_vi_state(
            model,
            observations,
            observations_covariance,
            run_directory_base,
            parameter_names,
            variational_mean,
            variational_log_std,
            prior_mean,
            prior_precision_operator,
            prior_covariance_log_det,
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
            transform_interior_margin,
            transform_map,
            variational_correlation_cholesky,
            elbo_scaling_factor,
            log_likelihood_precision_operator,
            sampling_method,
            dispatcher,
        )
        if 'initial_elbo_reference' in restart_data:
            initial_elbo_reference = float(restart_data['initial_elbo_reference'])
        else:
            initial_elbo_reference = float(state['elbo'])
    if restart_file is None:
        _save_vi_restart(
            f'{absolute_vi_directory}/iteration_{iteration}/restart.npz',
            state,
            variational_mean,
            variational_log_std,
            variational_distribution,
            prior_mean,
            prior_covariance,
            variational_correlation_cholesky,
            elbo_scaling_factor,
            elbo_relative_tolerance,
            initial_elbo_reference,
            iteration,
            step_size,
            bounded_parameter_handling,
            transform_map,
            baseline_method,
            optimization_method,
            line_search_objective,
            line_search_method,
            line_search_nonmonotone_window,
            line_search_armijo_coefficient,
            line_search_uncertainty_sigma,
            line_search_sample_growth_factor,
            log_std_learning_rate_factor,
            parameter_mins=parameter_mins,
            parameter_maxes=parameter_maxes,
            transform_interior_margin=transform_interior_margin,
            vi_history=vi_history,
            sampling_method=sampling_method,
            dispatcher=dispatcher,
        )
        _prune_old_restart_files(absolute_vi_directory, restart_files_to_keep, dispatcher)

    gradient_norm = _compute_gradient_norm(state, optimization_method)
    wall_time = time.time() - start_time
    cpu_time = time.process_time() - start_cpu_time
    if len(vi_history['cpu_time_seconds']) <= iteration:
        _append_vi_history(
            vi_history,
            variational_mean,
            variational_log_std,
            min_variational_std,
            max_variational_std,
            variational_correlation_cholesky,
            state['mean_relative_mse'],
            state['log_likelihoods'],
            cpu_time,
            bounded_parameter_handling,
            parameter_mins,
            parameter_maxes,
            transform_interior_margin,
            transform_map,
        )
    print(
        f'Iteration: {iteration}, Relative MSE: {state["mean_relative_mse"]:.5f}, '
        f'ELBO: {state["elbo"]:.5f}, Step size: {step_size:.5f}, '
        f'Gradient norm: {gradient_norm:.5f}, Wall time: {wall_time:.5f}'
    )
    _print_gradient_signal_to_noise_ratio(state)
    _print_vi_parameters(
        variational_mean,
        variational_log_std,
        min_variational_std,
        max_variational_std,
        optimizer_samples=state['optimizer_samples'],
        parameter_samples=state['parameter_samples'],
    )
    _write_iteration_stats_file(
        f'{absolute_vi_directory}/iteration_{iteration}',
        variational_mean,
        variational_log_std,
        min_variational_std,
        max_variational_std,
        variational_correlation_cholesky,
        state['elbo'],
        state['log_likelihoods'],
        state['log_priors'],
        state['mean_relative_mse'],
        wall_time,
        cpu_time,
        dispatcher=dispatcher,
    )

    iteration += 1
    step_failed_counter = 0
    line_search_standard_normal_cache = None
    steepest_descent_solver = SteepestDescentSolver()
    accepted_elbo_history = [state['elbo']]
    elbo_converged = False

    while iteration < max_iterations and gradient_norm > gradient_norm_tolerance:
        line_search_sample_size = int(np.ceil(
            sample_size * (line_search_sample_growth_factor ** step_failed_counter)
        ))
        line_search_sample_size = max(line_search_sample_size, 2)
        if line_search_standard_normal_cache is None:
            line_search_standard_normal_cache = _draw_standard_normal_samples(
                line_search_sample_size,
                variational_mean.size,
                sampling_method,
            )
        elif line_search_sample_size > line_search_standard_normal_cache.shape[0]:
            extra_sample_count = line_search_sample_size - line_search_standard_normal_cache.shape[0]
            extra_standard_normal_samples = _draw_standard_normal_samples(
                extra_sample_count,
                variational_mean.size,
                sampling_method,
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
                transform_map,
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
                prior_mean,
                prior_precision_operator,
                prior_covariance_log_det,
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
                transform_interior_margin,
                transform_map,
                line_search_objective,
                line_search_standard_normal_samples,
                variational_correlation_cholesky,
                elbo_scaling_factor,
                log_likelihood_precision_operator,
                dispatcher,
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
                    prior_mean=prior_mean,
                    prior_precision_operator=prior_precision_operator,
                    prior_covariance_log_det=prior_covariance_log_det,
                    covariance_regularization=covariance_regularization,
                    baseline_method=baseline_method,
                    gradient_method=gradient_method,
                    bounded_parameter_handling=bounded_parameter_handling,
                    min_variational_std=min_variational_std,
                    max_variational_std=max_variational_std,
                    parameter_mins=parameter_mins,
                    parameter_maxes=parameter_maxes,
                    transform_interior_margin=transform_interior_margin,
                    transform_map=transform_map,
                    variational_correlation_cholesky=variational_correlation_cholesky,
                    elbo_scaling_factor=elbo_scaling_factor,
                    log_likelihood_precision_operator=log_likelihood_precision_operator,
                    )
                state = test_state
                accepted_elbo_history.append(state['elbo'])
                step_size = min(step_size * step_size_growth_factor, max_step_size)
                line_search_standard_normal_cache = None
                relative_elbo_improvement = (
                    float(state['elbo']) / (initial_elbo_reference + 1e-16)
                )
                if (
                    elbo_relative_tolerance is not None
                    and relative_elbo_improvement <= elbo_relative_tolerance
                ):
                    elbo_converged = True

                gradient_norm = _compute_gradient_norm(state, optimization_method)
                wall_time = time.time() - start_time
                cpu_time = time.process_time() - start_cpu_time
                _append_vi_history(
                    vi_history,
                    variational_mean,
                    variational_log_std,
                    min_variational_std,
                    max_variational_std,
                    variational_correlation_cholesky,
                    state['mean_relative_mse'],
                    state['log_likelihoods'],
                    cpu_time,
                    bounded_parameter_handling,
                    parameter_mins,
                    parameter_maxes,
                    transform_interior_margin,
                    transform_map,
                )
                print(
                    f'Iteration: {iteration}, Relative MSE: {state["mean_relative_mse"]:.5f}, '
                    f'ELBO: {state["elbo"]:.5f}, Relative ELBO (initial ref): {relative_elbo_improvement:.5e}, '
                    f'Step size: {step_size:.5e}, '
                    f'Gradient norm: {gradient_norm:.5f}, Samples: {line_search_sample_size}, '
                    f'Wall time: {wall_time:.5f}'
                )
                _print_gradient_signal_to_noise_ratio(state)
                _print_vi_parameters(
                    variational_mean,
                    variational_log_std,
                    min_variational_std,
                    max_variational_std,
                    optimizer_samples=state['optimizer_samples'],
                    parameter_samples=state['parameter_samples'],
                )
                _write_iteration_stats_file(
                    f'{absolute_vi_directory}/iteration_{iteration}',
                    variational_mean,
                    variational_log_std,
                    min_variational_std,
                    max_variational_std,
                    variational_correlation_cholesky,
                    state['elbo'],
                    state['log_likelihoods'],
                    state['log_priors'],
                    state['mean_relative_mse'],
                    wall_time,
                    cpu_time,
                    bounded_parameter_handling,
                    parameter_mins,
                    parameter_maxes,
                    transform_interior_margin,
                    transform_map,
                    dispatcher=dispatcher,
                )

                _save_vi_restart(
                    f'{absolute_vi_directory}/iteration_{iteration}/restart.npz',
                    state,
                    variational_mean,
                    variational_log_std,
                    variational_distribution,
                    prior_mean,
                    prior_covariance,
                    variational_correlation_cholesky,
                    elbo_scaling_factor,
                    elbo_relative_tolerance,
                    initial_elbo_reference,
                    iteration,
                    step_size,
                    bounded_parameter_handling,
                    transform_map,
                    baseline_method,
                    optimization_method,
                    line_search_objective,
                    line_search_method,
                    line_search_nonmonotone_window,
                    line_search_armijo_coefficient,
                    line_search_uncertainty_sigma,
                    line_search_sample_growth_factor,
                    log_std_learning_rate_factor,
                    parameter_mins=parameter_mins,
                    parameter_maxes=parameter_maxes,
                    transform_interior_margin=transform_interior_margin,
                    vi_history=vi_history,
                    sampling_method=sampling_method,
                    dispatcher=dispatcher,
                )
                _prune_old_restart_files(absolute_vi_directory, restart_files_to_keep, dispatcher)
                iteration += 1
                if elbo_converged:
                    print(
                        "ELBO relative-improvement tolerance reached "
                        f"({relative_elbo_improvement:.5e} <= {elbo_relative_tolerance:.5e}), terminating"
                    )
                    break
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
                newton_hessian_type=newton_hessian_type,
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
                transform_map,
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
                prior_mean,
                prior_precision_operator,
                prior_covariance_log_det,
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
                transform_interior_margin,
                transform_map,
                line_search_objective,
                line_search_standard_normal_samples,
                variational_correlation_cholesky,
                elbo_scaling_factor,
                log_likelihood_precision_operator,
                dispatcher,
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
                    prior_mean=prior_mean,
                    prior_precision_operator=prior_precision_operator,
                    prior_covariance_log_det=prior_covariance_log_det,
                    covariance_regularization=covariance_regularization,
                    baseline_method=baseline_method,
                    gradient_method=gradient_method,
                    bounded_parameter_handling=bounded_parameter_handling,
                    min_variational_std=min_variational_std,
                    max_variational_std=max_variational_std,
                    parameter_mins=parameter_mins,
                    parameter_maxes=parameter_maxes,
                    transform_interior_margin=transform_interior_margin,
                    transform_map=transform_map,
                    variational_correlation_cholesky=variational_correlation_cholesky,
                    elbo_scaling_factor=elbo_scaling_factor,
                    log_likelihood_precision_operator=log_likelihood_precision_operator,
                    )
                state = test_state
                step_size = min(step_size * step_size_growth_factor, max_step_size)
                line_search_standard_normal_cache = None

                accepted_elbo_history.append(state['elbo'])
                relative_elbo_improvement = (
                    float(state['elbo']) / (initial_elbo_reference + 1e-16)
                )
                if (
                    elbo_relative_tolerance is not None
                    and relative_elbo_improvement <= elbo_relative_tolerance
                ):
                    elbo_converged = True
                gradient_norm = _compute_gradient_norm(state, optimization_method)
                wall_time = time.time() - start_time
                cpu_time = time.process_time() - start_cpu_time
                _append_vi_history(
                    vi_history,
                    variational_mean,
                    variational_log_std,
                    min_variational_std,
                    max_variational_std,
                    variational_correlation_cholesky,
                    state['mean_relative_mse'],
                    state['log_likelihoods'],
                    cpu_time,
                    bounded_parameter_handling,
                    parameter_mins,
                    parameter_maxes,
                    transform_interior_margin,
                    transform_map,
                )
                print(
                    f'Iteration: {iteration}, Relative MSE: {state["mean_relative_mse"]:.5f}, '
                    f'ELBO: {state["elbo"]:.5f}, Relative ELBO (initial ref): {relative_elbo_improvement:.5e}, '
                    f'Step size: {step_size:.5e}, '
                    f'Gradient norm: {gradient_norm:.5f}, Samples: {line_search_sample_size}, '
                    f'Wall time: {wall_time:.5f}'
                )
                _print_gradient_signal_to_noise_ratio(state)
                _print_vi_parameters(
                    variational_mean,
                    variational_log_std,
                    min_variational_std,
                    max_variational_std,
                    optimizer_samples=state['optimizer_samples'],
                    parameter_samples=state['parameter_samples'],
                )
                _write_iteration_stats_file(
                    f'{absolute_vi_directory}/iteration_{iteration}',
                    variational_mean,
                    variational_log_std,
                    min_variational_std,
                    max_variational_std,
                    variational_correlation_cholesky,
                    state['elbo'],
                    state['log_likelihoods'],
                    state['log_priors'],
                    state['mean_relative_mse'],
                    wall_time,
                    cpu_time,
                    bounded_parameter_handling,
                    parameter_mins,
                    parameter_maxes,
                    transform_interior_margin,
                    transform_map,
                    dispatcher=dispatcher,
                )

                _save_vi_restart(
                    f'{absolute_vi_directory}/iteration_{iteration}/restart.npz',
                    state,
                    variational_mean,
                    variational_log_std,
                    variational_distribution,
                    prior_mean,
                    prior_covariance,
                    variational_correlation_cholesky,
                    elbo_scaling_factor,
                    elbo_relative_tolerance,
                    initial_elbo_reference,
                    iteration,
                    step_size,
                    bounded_parameter_handling,
                    transform_map,
                    baseline_method,
                    optimization_method,
                    line_search_objective,
                    line_search_method,
                    line_search_nonmonotone_window,
                    line_search_armijo_coefficient,
                    line_search_uncertainty_sigma,
                    line_search_sample_growth_factor,
                    log_std_learning_rate_factor,
                    parameter_mins=parameter_mins,
                    parameter_maxes=parameter_maxes,
                    transform_interior_margin=transform_interior_margin,
                    vi_history=vi_history,
                    sampling_method=sampling_method,
                    dispatcher=dispatcher,
                )
                _prune_old_restart_files(absolute_vi_directory, restart_files_to_keep, dispatcher)
                iteration += 1
                if elbo_converged:
                    print(
                        "ELBO relative-improvement tolerance reached "
                        f"({relative_elbo_improvement:.5e} <= {elbo_relative_tolerance:.5e}), terminating"
                    )
                    break
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
    elif elbo_converged:
        print('ELBO relative-improvement tolerance reached!')
    elif gradient_norm <= gradient_norm_tolerance:
        print('Gradient norm dropped below tolerance!')

    variational_std, _ = _compute_variational_std(
        variational_log_std,
        min_variational_std,
        max_variational_std,
    )
    _save_vi_history(absolute_vi_directory, vi_history, dispatcher)
    return variational_mean, variational_std, state['parameter_samples'], state['qois']
