"""Full-covariance Gaussian VI wrappers.

This module adds a true full-covariance Gaussian variational family without
changing the established diagonal VI and MF-VI implementations.  The Gaussian
lives in optimizer coordinates, ``q(x)=N(mu, Sigma)``, with ``Sigma=L L.T``.
The Cholesky factor is the persisted numerical state while updates use the
Gaussian Fisher geometry in mean/covariance coordinates.
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Optional

import numpy as np

from romtools.hpc.dispatchers import resolve_dispatcher
from romtools.workflows.inverse import vi_drivers as _vi
from romtools.workflows.inverse import mf_vi_drivers as _mf
from romtools.workflows.inverse import vi_run_directory_policy as _directory_policy
from romtools.workflows.inverse.full_covariance_vi import (
    correlation_cholesky_from_covariance,
    covariance_from_cholesky,
    entropy as full_covariance_entropy,
    estimate_natural_gradient,
    log_density as full_covariance_log_density,
    marginal_std_from_cholesky,
    natural_score_terms,
    packed_direction_to_covariance,
    rescale_cholesky_marginals,
    retract_covariance,
    robust_cholesky,
    svec,
    svec_inverse,
)
from romtools.workflows.inverse.vi_optimization_methods import (
    AdamSolver,
    SteepestDescentSolver,
    VIAdamOptimizerConfig,
    VIGradientOptimizerConfig,
    VINewtonOptimizerConfig,
    VILegacyLineSearchConfig,
    VIStochasticNonmonotoneLineSearchConfig,
    _normalize_line_search_objective,
    _resolve_line_search_config,
    _resolve_optimizer_config,
)
from romtools.workflows.parameter_spaces import MultivariateGaussianParameterSpace


_FALLBACK_RUN_VI = _directory_policy.run_vi
_FALLBACK_RUN_MF_VI = _directory_policy.run_mf_vi
_FALLBACK_AUTO_MF_VI = _directory_policy.mf_vi_with_auto_rom


def _resolve_variational_distribution(
    requested,
    prior_parameter_space,
    initial_variational_parameter_space,
) -> str:
    """Resolve public variational-family semantics.

    For backward compatibility, a multivariate Gaussian initializer/prior now
    selects the corrected full-covariance implementation when the family is not
    specified explicitly.  ``multivariate`` is retained as a deprecated alias
    for ``full_covariance``.
    """
    if requested is None:
        source = (
            initial_variational_parameter_space
            if initial_variational_parameter_space is not None
            else prior_parameter_space
        )
        return (
            "full_covariance"
            if isinstance(source, MultivariateGaussianParameterSpace)
            else "diagonal"
        )
    normalized = str(requested).strip().lower()
    if normalized in ("full_covariance", "full-covariance", "full", "multivariate"):
        if normalized == "multivariate":
            warnings.warn(
                "variational_distribution='multivariate' is deprecated; use "
                "'full_covariance'. The multivariate family now has freely "
                "evolving covariance rather than fixed correlation.",
                DeprecationWarning,
                stacklevel=3,
            )
        return "full_covariance"
    if normalized in ("diagonal", "mean_field", "mean-field"):
        return "diagonal"
    raise ValueError(
        f"Unsupported variational_distribution '{requested}'. Supported "
        "options are 'diagonal' and 'full_covariance'."
    )


def _extract_prior_and_initial_moments(
    prior_parameter_space,
    initial_variational_parameter_space,
):
    prior_names, prior_mean, prior_covariance, _ = _vi._extract_gaussian_parameter_space(
        prior_parameter_space,
        argument_name="prior_parameter_space",
    )
    if initial_variational_parameter_space is None:
        initial_names = prior_names
        initial_mean = prior_mean.copy()
        initial_covariance = prior_covariance.copy()
    else:
        initial_names, initial_mean, initial_covariance, _ = _vi._extract_gaussian_parameter_space(
            initial_variational_parameter_space,
            argument_name="initial_variational_parameter_space",
        )
    if list(initial_names) != list(prior_names):
        raise ValueError(
            "prior_parameter_space and initial_variational_parameter_space must "
            "define the same parameter names in the same order."
        )
    return (
        list(prior_names),
        np.asarray(prior_mean, dtype=float),
        np.asarray(prior_covariance, dtype=float),
        np.asarray(initial_mean, dtype=float),
        np.asarray(initial_covariance, dtype=float),
    )


def _representation_from_cholesky(cholesky: np.ndarray):
    covariance = covariance_from_cholesky(cholesky)
    variational_std, correlation_cholesky = correlation_cholesky_from_covariance(covariance)
    return covariance, variational_std, np.log(variational_std), correlation_cholesky


def _enforce_full_covariance_scale_bounds(
    variational_mean: np.ndarray,
    cholesky: np.ndarray,
    min_variational_std: float,
    max_variational_std: float,
    bounded_parameter_handling: str,
    parameter_mins: np.ndarray,
    parameter_maxes: np.ndarray,
    transform_interior_margin: float,
    min_physical_variational_std_fraction: float,
    transform_map: str,
) -> np.ndarray:
    dimensionality = variational_mean.size
    minimum = np.full(dimensionality, min_variational_std, dtype=float)
    maximum = np.full(dimensionality, max_variational_std, dtype=float)
    if (
        bounded_parameter_handling == "transform"
        and min_physical_variational_std_fraction > 0.0
    ):
        parameter_ranges = np.asarray(parameter_maxes) - np.asarray(parameter_mins)
        physical_floor = min_physical_variational_std_fraction * parameter_ranges
        jacobian = _vi._compute_transform_jacobian_diagonal(
            variational_mean,
            parameter_mins,
            parameter_maxes,
            transform_interior_margin,
            transform_map,
        )
        required_optimizer_std = physical_floor / np.maximum(np.abs(jacobian), 1e-16)
        minimum = np.minimum(np.maximum(minimum, required_optimizer_std), maximum)
    return rescale_cholesky_marginals(cholesky, minimum, maximum)


def _ordinary_from_natural(
    natural_mean: np.ndarray,
    natural_covariance_svec: np.ndarray,
    cholesky: np.ndarray,
):
    covariance = covariance_from_cholesky(cholesky)
    ordinary_mean = np.linalg.solve(covariance, natural_mean)
    natural_covariance = svec_inverse(natural_covariance_svec, natural_mean.size)
    left = np.linalg.solve(covariance, natural_covariance)
    ordinary_covariance = 0.5 * np.linalg.solve(covariance, left.T).T
    ordinary_covariance = 0.5 * (ordinary_covariance + ordinary_covariance.T)
    return ordinary_mean, svec(ordinary_covariance)


def _upgrade_single_fidelity_state(
    state,
    variational_mean: np.ndarray,
    cholesky: np.ndarray,
    baseline_method: str,
    gradient_method: str,
    elbo_scaling_factor: float,
    score_function_entropy_strategy: str,
):
    estimate = estimate_natural_gradient(
        state["optimizer_samples"],
        variational_mean,
        cholesky,
        elbo_scaling_factor * np.asarray(state["log_joint_terms"]),
        baseline_method=baseline_method,
        elbo_scaling_factor=elbo_scaling_factor,
        entropy_strategy=score_function_entropy_strategy,
    )
    if gradient_method == "natural":
        update_mean = estimate["natural_mean"]
        update_covariance = estimate["natural_covariance_svec"]
    elif gradient_method == "standard":
        update_mean = estimate["ordinary_mean"]
        update_covariance = estimate["ordinary_covariance_svec"]
    else:
        raise ValueError("Full-covariance VI supports gradient_method='standard' or 'natural'.")

    state = dict(state)
    state.update(
        gradient_mean=estimate["ordinary_mean"],
        gradient_covariance_svec=estimate["ordinary_covariance_svec"],
        natural_gradient_mean=estimate["natural_mean"],
        natural_gradient_covariance_svec=estimate["natural_covariance_svec"],
        update_direction_mean=update_mean,
        update_direction_covariance_svec=update_covariance,
        baseline_mean=estimate["baseline_mean"],
        baseline_covariance_svec=estimate["baseline_covariance_svec"],
        gradient_standard_error=estimate["gradient_standard_error"],
        gradient_signal_to_noise_ratio=estimate["gradient_signal_to_noise_ratio"],
        gradient_method=gradient_method,
    )
    state["entropy"] = full_covariance_entropy(cholesky, elbo_scaling_factor)
    state["elbo"] = (
        elbo_scaling_factor * float(np.mean(state["log_joint_terms"]))
        + state["entropy"]
    )
    return state


def _evaluate_single_fidelity_state(
    *,
    model,
    observations,
    observations_covariance,
    run_directory_base,
    parameter_names,
    variational_mean,
    cholesky,
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
    elbo_scaling_factor,
    log_likelihood_precision_operator,
    sampling_method,
    dispatcher,
    score_function_entropy_strategy,
    standard_normal_samples=None,
):
    _, variational_std, variational_log_std, correlation_cholesky = (
        _representation_from_cholesky(cholesky)
    )
    if standard_normal_samples is None:
        state = _vi._evaluate_vi_state(
            model=model,
            observations=observations,
            observations_covariance=observations_covariance,
            run_directory_base=run_directory_base,
            parameter_names=parameter_names,
            variational_mean=variational_mean,
            variational_log_std=variational_log_std,
            prior_mean=prior_mean,
            prior_precision_operator=prior_precision_operator,
            prior_covariance_log_det=prior_covariance_log_det,
            sample_size=sample_size,
            evaluation_concurrency=evaluation_concurrency,
            covariance_regularization=covariance_regularization,
            baseline_method=baseline_method,
            gradient_method="standard",
            bounded_parameter_handling=bounded_parameter_handling,
            min_variational_std=min_variational_std,
            max_variational_std=max_variational_std,
            parameter_mins=parameter_mins,
            parameter_maxes=parameter_maxes,
            transform_interior_margin=transform_interior_margin,
            transform_map=transform_map,
            variational_correlation_cholesky=correlation_cholesky,
            elbo_scaling_factor=elbo_scaling_factor,
            log_likelihood_precision_operator=log_likelihood_precision_operator,
            sampling_method=sampling_method,
            dispatcher=dispatcher,
            score_function_entropy_strategy=score_function_entropy_strategy,
        )
    else:
        candidate = _vi._evaluate_vi_candidate_for_line_search(
            model=model,
            observations=observations,
            observations_covariance=observations_covariance,
            run_directory_base=run_directory_base,
            parameter_names=parameter_names,
            variational_mean=variational_mean,
            variational_log_std=variational_log_std,
            prior_mean=prior_mean,
            prior_precision_operator=prior_precision_operator,
            prior_covariance_log_det=prior_covariance_log_det,
            sample_size=sample_size,
            evaluation_concurrency=evaluation_concurrency,
            covariance_regularization=covariance_regularization,
            baseline_method=baseline_method,
            gradient_method="standard",
            bounded_parameter_handling=bounded_parameter_handling,
            min_variational_std=min_variational_std,
            max_variational_std=max_variational_std,
            parameter_mins=parameter_mins,
            parameter_maxes=parameter_maxes,
            transform_interior_margin=transform_interior_margin,
            transform_map=transform_map,
            line_search_objective="elbo",
            standard_normal_samples=standard_normal_samples,
            variational_correlation_cholesky=correlation_cholesky,
            elbo_scaling_factor=elbo_scaling_factor,
            log_likelihood_precision_operator=log_likelihood_precision_operator,
            dispatcher=dispatcher,
            score_function_entropy_strategy=score_function_entropy_strategy,
        )
        state = candidate["state"]
    return _upgrade_single_fidelity_state(
        state,
        variational_mean,
        cholesky,
        baseline_method,
        gradient_method,
        elbo_scaling_factor,
        score_function_entropy_strategy,
    )


def _gradient_vector(state, gradient_method: str) -> np.ndarray:
    if gradient_method == "natural":
        return np.concatenate([
            state["natural_gradient_mean"],
            state["natural_gradient_covariance_svec"],
        ])
    return np.concatenate([
        state["gradient_mean"], state["gradient_covariance_svec"]
    ])


def _update_vector(state) -> np.ndarray:
    return np.concatenate([
        state["update_direction_mean"], state["update_direction_covariance_svec"]
    ])


def _gradient_norm(state) -> float:
    return float(np.linalg.norm(_update_vector(state)))


def _append_history(
    history,
    variational_mean,
    cholesky,
    state,
    cpu_time,
    accepted_step_size,
    bounded_parameter_handling,
    parameter_mins,
    parameter_maxes,
    transform_interior_margin,
    transform_map,
):
    history["variational_mean"].append(
        _vi._get_persisted_variational_mean(
            variational_mean,
            bounded_parameter_handling,
            parameter_mins,
            parameter_maxes,
            transform_interior_margin,
            transform_map,
        )
    )
    history["variational_covariance"].append(covariance_from_cholesky(cholesky))
    history["relative_mse"].append(float(state["mean_relative_mse"]))
    history["loglikelihood"].append(float(np.mean(state["log_likelihoods"])))
    history["elbo"].append(float(state["elbo"]))
    history["cpu_time_seconds"].append(float(cpu_time))
    history["accepted_step_size"].append(float(accepted_step_size))
    history["gradient"].append(_gradient_vector(state, state["gradient_method"]).copy())
    history["gradient_standard_error"].append(
        np.asarray(state["gradient_standard_error"]).copy()
    )


def _new_history():
    return {
        "variational_mean": [],
        "variational_covariance": [],
        "relative_mse": [],
        "loglikelihood": [],
        "elbo": [],
        "cpu_time_seconds": [],
        "accepted_step_size": [],
        "gradient": [],
        "gradient_standard_error": [],
    }


def _save_history(absolute_work_dir, history, dispatcher):
    dispatcher.np_savez(
        f"{absolute_work_dir}/history.npz",
        vi_history_variational_mean=np.asarray(history["variational_mean"]),
        vi_history_variational_covariance=np.asarray(history["variational_covariance"]),
        vi_history_relative_mse=np.asarray(history["relative_mse"]),
        vi_history_loglikelihood=np.asarray(history["loglikelihood"]),
        vi_history_elbo=np.asarray(history["elbo"]),
        vi_history_cpu_time_seconds=np.asarray(history["cpu_time_seconds"]),
        vi_history_accepted_step_size=np.asarray(history["accepted_step_size"]),
        vi_history_gradient=np.asarray(history["gradient"]),
        vi_history_gradient_standard_error=np.asarray(history["gradient_standard_error"]),
        variational_family="full_covariance",
    )


def _save_full_covariance_restart(
    restart_path,
    *,
    variational_mean,
    cholesky,
    state,
    prior_mean,
    prior_covariance,
    iteration,
    step_size,
    bounded_parameter_handling,
    transform_map,
    parameter_mins,
    parameter_maxes,
    transform_interior_margin,
    baseline_method,
    score_function_entropy_strategy,
    optimization_method,
    gradient_method,
    sampling_method,
    elbo_scaling_factor,
    accepted_elbo_history,
    max_covariance_log_step,
    adam_solver,
    dispatcher,
):
    save_data = {
        "variational_mean": _vi._get_persisted_variational_mean(
            variational_mean,
            bounded_parameter_handling,
            parameter_mins,
            parameter_maxes,
            transform_interior_margin,
            transform_map,
        ),
        "variational_mean_coordinates": "physical",
        "variational_cholesky_optimizer": np.asarray(cholesky),
        "variational_distribution": "full_covariance",
        "prior_mean": prior_mean,
        "prior_covariance": prior_covariance,
        "iteration": int(iteration),
        "step_size": float(step_size),
        "bounded_parameter_handling": bounded_parameter_handling,
        "transform_map": transform_map,
        "baseline_method": baseline_method,
        "score_function_entropy_strategy": score_function_entropy_strategy,
        "optimization_method": optimization_method,
        "gradient_method": gradient_method,
        "sampling_method": sampling_method,
        "elbo_scaling_factor": float(elbo_scaling_factor),
        "accepted_elbo_history": np.asarray(accepted_elbo_history, dtype=float),
        "max_covariance_log_step": float(max_covariance_log_step),
        "rng_state": np.array(np.random.get_state(), dtype=object),
    }
    for key in (
        "optimizer_samples",
        "parameter_samples",
        "qois",
        "mean_qoi",
        "errors",
        "log_likelihoods",
        "log_priors",
        "log_joint_terms",
        "mean_misfit",
        "mean_relative_mse",
        "entropy",
        "elbo",
        "gradient_mean",
        "gradient_covariance_svec",
        "natural_gradient_mean",
        "natural_gradient_covariance_svec",
        "update_direction_mean",
        "update_direction_covariance_svec",
        "gradient_standard_error",
        "gradient_signal_to_noise_ratio",
    ):
        if key in state and state[key] is not None:
            save_data[key] = state[key]
    if adam_solver is not None:
        save_data.update(adam_solver.restart_state_dict())
        save_data["adam_gradient_method"] = gradient_method
    dispatcher.np_savez(restart_path, **save_data)


def _restore_single_state(restart_data, gradient_method):
    required = (
        "optimizer_samples",
        "parameter_samples",
        "qois",
        "mean_qoi",
        "errors",
        "log_likelihoods",
        "log_priors",
        "log_joint_terms",
        "mean_misfit",
        "mean_relative_mse",
        "entropy",
        "elbo",
        "gradient_mean",
        "gradient_covariance_svec",
        "natural_gradient_mean",
        "natural_gradient_covariance_svec",
        "update_direction_mean",
        "update_direction_covariance_svec",
        "gradient_standard_error",
        "gradient_signal_to_noise_ratio",
    )
    if not all(key in restart_data for key in required):
        return None
    state = {key: restart_data[key] for key in required}
    state["mean_misfit"] = float(state["mean_misfit"])
    state["mean_relative_mse"] = float(state["mean_relative_mse"])
    state["entropy"] = float(state["entropy"])
    state["elbo"] = float(state["elbo"])
    state["gradient_signal_to_noise_ratio"] = float(
        state["gradient_signal_to_noise_ratio"]
    )
    state["gradient_method"] = gradient_method
    return state


def _write_iteration_outputs(
    absolute_work_dir,
    iteration,
    variational_mean,
    cholesky,
    state,
    wall_time,
    cpu_time,
    bounded_parameter_handling,
    parameter_mins,
    parameter_maxes,
    transform_interior_margin,
    transform_map,
    dispatcher,
):
    _, variational_std, variational_log_std, correlation_cholesky = (
        _representation_from_cholesky(cholesky)
    )
    _vi._write_iteration_stats_file(
        f"{absolute_work_dir}/iteration_{iteration}",
        variational_mean,
        variational_log_std,
        float(np.min(variational_std)) * 0.5,
        float(np.max(variational_std)) * 2.0 + 1e-16,
        correlation_cholesky,
        state["elbo"],
        state["log_likelihoods"],
        state["log_priors"],
        state["mean_relative_mse"],
        wall_time,
        cpu_time,
        bounded_parameter_handling,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin,
        transform_map,
        dispatcher=dispatcher,
    )


def _run_full_covariance_vi(
    *,
    model,
    prior_parameter_space,
    observations,
    observations_covariance,
    parameter_mins,
    parameter_maxes,
    initial_variational_parameter_space,
    restart_file,
    optimizer_method,
    optimizer_config,
    line_search_method,
    line_search_config,
    absolute_work_dir,
    sample_size,
    random_seed,
    sampling_method,
    evaluation_concurrency,
    covariance_regularization,
    restart_files_to_keep,
    elbo_scaling_factor,
    elbo_relative_tolerance,
    baseline_method,
    bounded_parameter_handling,
    transform_interior_margin,
    transform_map,
    min_physical_variational_std_fraction,
    dispatcher,
    score_function_entropy_strategy,
    max_covariance_log_step,
):
    dispatcher = resolve_dispatcher(dispatcher)
    dispatcher.require_supported_concurrency(evaluation_concurrency)
    if absolute_work_dir is None:
        absolute_work_dir = os.getcwd() + "/work/"
    dispatcher.require_absolute_path(absolute_work_dir)
    if sample_size <= 1:
        raise ValueError("sample_size must be greater than 1")
    if max_covariance_log_step is None or max_covariance_log_step <= 0.0:
        raise ValueError("max_covariance_log_step must be positive")

    optimization_method, resolved_optimizer_config = _resolve_optimizer_config(
        optimizer_method,
        optimizer_config,
        VIGradientOptimizerConfig(),
        VINewtonOptimizerConfig(),
        VIAdamOptimizerConfig(),
    )
    if optimization_method == "newton":
        raise NotImplementedError(
            "Full-covariance Newton/Hessian support is intentionally out of scope; "
            "use optimizer_method='gradient' or 'adam'."
        )
    gradient_method = resolved_optimizer_config.gradient_method.strip().lower()
    if gradient_method not in ("standard", "natural"):
        raise ValueError("gradient_method must be 'standard' or 'natural'")
    gradient_norm_tolerance = resolved_optimizer_config.gradient_norm_tolerance
    max_iterations = resolved_optimizer_config.max_iterations
    min_variational_std = resolved_optimizer_config.min_variational_std
    max_variational_std = resolved_optimizer_config.max_variational_std

    if optimization_method == "adam":
        if line_search_config is not None:
            raise ValueError("line_search_config is not supported with Adam")
        line_search_method = "legacy"
        resolved_line_search_config = VILegacyLineSearchConfig(
            initial_step_size=1.0,
            max_step_size=1.0,
            step_size_growth_factor=1.0,
            step_size_decay_factor=1.0,
            max_step_size_decrease_trys=0,
            relaxation_parameter=1.0,
            line_search_objective="elbo",
            line_search_sample_growth_factor=1.0,
            log_std_learning_rate_factor=1.0,
        )
    else:
        line_search_method, resolved_line_search_config = _resolve_line_search_config(
            line_search_method,
            line_search_config,
            VILegacyLineSearchConfig(),
            VIStochasticNonmonotoneLineSearchConfig(),
        )
    line_search_objective = _normalize_line_search_objective(
        resolved_line_search_config.line_search_objective
    )
    initial_step_size = resolved_line_search_config.initial_step_size
    max_step_size = resolved_line_search_config.max_step_size
    step_size_growth_factor = resolved_line_search_config.step_size_growth_factor
    step_size_decay_factor = resolved_line_search_config.step_size_decay_factor
    max_step_size_decrease_trys = resolved_line_search_config.max_step_size_decrease_trys
    relaxation_parameter = resolved_line_search_config.relaxation_parameter
    line_search_sample_growth_factor = resolved_line_search_config.line_search_sample_growth_factor
    nonmonotone_window = getattr(resolved_line_search_config, "line_search_nonmonotone_window", 1)
    armijo_coefficient = getattr(resolved_line_search_config, "line_search_armijo_coefficient", 0.0)
    uncertainty_sigma = getattr(resolved_line_search_config, "line_search_uncertainty_sigma", 0.0)

    parameter_mins, parameter_maxes = _vi._resolve_parameter_bounds(
        parameter_mins, parameter_maxes
    )
    bounded_parameter_handling = _vi._normalize_bounded_parameter_handling(
        bounded_parameter_handling
    )
    transform_map = _vi._normalize_transform_map(transform_map)
    sampling_method = _vi._normalize_sampling_method(sampling_method)
    score_function_entropy_strategy = _vi._normalize_score_function_entropy_strategy(
        score_function_entropy_strategy
    )
    baseline_method = _vi._normalize_baseline_method(
        "loo" if baseline_method is None else baseline_method
    )
    elbo_scaling_factor = _vi._resolve_elbo_scaling_factor(
        elbo_scaling_factor, observations_covariance
    )

    (
        parameter_names,
        prior_mean,
        prior_covariance,
        initial_mean,
        initial_covariance,
    ) = _extract_prior_and_initial_moments(
        prior_parameter_space, initial_variational_parameter_space
    )
    dimensionality = initial_mean.size
    if bounded_parameter_handling == "transform":
        if parameter_mins is None or parameter_maxes is None:
            raise ValueError(
                "parameter_mins and parameter_maxes are required for transformed bounds"
            )
    log_likelihood_precision_operator = _vi._compute_log_likelihood_precision_operator(
        observations_covariance, covariance_regularization
    )
    prior_precision_operator, prior_covariance_log_det = _vi._compute_gaussian_log_density_data(
        prior_covariance
    )

    start_time = time.time()
    start_cpu_time = time.process_time()
    history = _new_history()
    adam_solver = None
    step_size = min(initial_step_size, max_step_size)
    iteration = 0

    if restart_file is None:
        np.random.seed(random_seed)
        optimizer_mean, optimizer_covariance = _vi._convert_physical_moments_to_optimizer_moments(
            initial_mean,
            initial_covariance,
            bounded_parameter_handling,
            parameter_mins,
            parameter_maxes,
            transform_interior_margin,
            transform_map,
        )
        variational_mean = optimizer_mean
        cholesky = robust_cholesky(optimizer_covariance)
        cholesky = _enforce_full_covariance_scale_bounds(
            variational_mean,
            cholesky,
            min_variational_std,
            max_variational_std,
            bounded_parameter_handling,
            parameter_mins,
            parameter_maxes,
            transform_interior_margin,
            min_physical_variational_std_fraction,
            transform_map,
        )
        state = _evaluate_single_fidelity_state(
            model=model,
            observations=observations,
            observations_covariance=observations_covariance,
            run_directory_base=f"{absolute_work_dir}/iteration_0/run_",
            parameter_names=parameter_names,
            variational_mean=variational_mean,
            cholesky=cholesky,
            prior_mean=prior_mean,
            prior_precision_operator=prior_precision_operator,
            prior_covariance_log_det=prior_covariance_log_det,
            sample_size=sample_size,
            evaluation_concurrency=evaluation_concurrency,
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
            elbo_scaling_factor=elbo_scaling_factor,
            log_likelihood_precision_operator=log_likelihood_precision_operator,
            sampling_method=sampling_method,
            dispatcher=dispatcher,
            score_function_entropy_strategy=score_function_entropy_strategy,
        )
        accepted_elbo_history = [float(state["elbo"])]
        initial_elbo_reference = float(state["elbo"])
    else:
        restart_data = np.load(restart_file, allow_pickle=True)
        if str(restart_data["variational_distribution"].item()) != "full_covariance":
            raise ValueError("restart_file is not a full-covariance VI restart")
        if "rng_state" in restart_data:
            np.random.set_state(tuple(restart_data["rng_state"].tolist()))
        else:
            np.random.seed(random_seed)
        iteration = int(restart_data["iteration"])
        step_size = min(float(restart_data["step_size"]), max_step_size)
        variational_mean = _vi._restore_variational_mean_from_restart(
            restart_data,
            bounded_parameter_handling,
            parameter_mins,
            parameter_maxes,
            transform_interior_margin,
            transform_map,
        )
        cholesky = np.asarray(restart_data["variational_cholesky_optimizer"], dtype=float)
        state = _restore_single_state(restart_data, gradient_method)
        if state is None:
            state = _evaluate_single_fidelity_state(
                model=model,
                observations=observations,
                observations_covariance=observations_covariance,
                run_directory_base=f"{absolute_work_dir}/iteration_{iteration}/run_restart_",
                parameter_names=parameter_names,
                variational_mean=variational_mean,
                cholesky=cholesky,
                prior_mean=prior_mean,
                prior_precision_operator=prior_precision_operator,
                prior_covariance_log_det=prior_covariance_log_det,
                sample_size=sample_size,
                evaluation_concurrency=evaluation_concurrency,
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
                elbo_scaling_factor=elbo_scaling_factor,
                log_likelihood_precision_operator=log_likelihood_precision_operator,
                sampling_method=sampling_method,
                dispatcher=dispatcher,
                score_function_entropy_strategy=score_function_entropy_strategy,
            )
        accepted_elbo_history = (
            restart_data["accepted_elbo_history"].tolist()
            if "accepted_elbo_history" in restart_data
            else [float(state["elbo"])]
        )
        initial_elbo_reference = float(accepted_elbo_history[0])

    solver = (
        AdamSolver.from_config(resolved_optimizer_config)
        if optimization_method == "adam"
        else SteepestDescentSolver()
    )
    if optimization_method == "adam":
        solver.parameter_dimension = dimensionality
        adam_solver = solver
        if restart_file is not None and "adam_iteration" in restart_data:
            solver.load_restart_state_dict(restart_data)

    gradient_norm = _gradient_norm(state)
    wall_time = time.time() - start_time
    cpu_time = time.process_time() - start_cpu_time
    _append_history(
        history,
        variational_mean,
        cholesky,
        state,
        cpu_time,
        np.nan,
        bounded_parameter_handling,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin,
        transform_map,
    )
    _write_iteration_outputs(
        absolute_work_dir,
        iteration,
        variational_mean,
        cholesky,
        state,
        wall_time,
        cpu_time,
        bounded_parameter_handling,
        parameter_mins,
        parameter_maxes,
        transform_interior_margin,
        transform_map,
        dispatcher,
    )
    _save_full_covariance_restart(
        f"{absolute_work_dir}/iteration_{iteration}/restart.npz",
        variational_mean=variational_mean,
        cholesky=cholesky,
        state=state,
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
        iteration=iteration,
        step_size=step_size,
        bounded_parameter_handling=bounded_parameter_handling,
        transform_map=transform_map,
        parameter_mins=parameter_mins,
        parameter_maxes=parameter_maxes,
        transform_interior_margin=transform_interior_margin,
        baseline_method=baseline_method,
        score_function_entropy_strategy=score_function_entropy_strategy,
        optimization_method=optimization_method,
        gradient_method=gradient_method,
        sampling_method=sampling_method,
        elbo_scaling_factor=elbo_scaling_factor,
        accepted_elbo_history=accepted_elbo_history,
        max_covariance_log_step=max_covariance_log_step,
        adam_solver=adam_solver,
        dispatcher=dispatcher,
    )
    _vi._prune_old_restart_files(absolute_work_dir, restart_files_to_keep, dispatcher)

    print(
        f"Iteration: {iteration}, Relative MSE: {state['mean_relative_mse']:.5f}, "
        f"ELBO: {state['elbo']:.5f}, Gradient norm: {gradient_norm:.5f}, "
        f"Wall time: {wall_time:.5f}"
    )
    iteration += 1
    failed_steps = 0
    standard_normal_cache = None
    elbo_converged = False

    while iteration < max_iterations and gradient_norm > gradient_norm_tolerance:
        current_sample_size = max(
            2,
            int(np.ceil(sample_size * line_search_sample_growth_factor**failed_steps)),
        )
        if standard_normal_cache is None or standard_normal_cache.shape[0] < current_sample_size:
            standard_normal_cache = _vi._draw_standard_normal_samples(
                current_sample_size, dimensionality, sampling_method
            )
        standard_normals = standard_normal_cache[:current_sample_size]

        direction_vector = _update_vector(state)
        if optimization_method == "adam":
            direction_vector = solver.step(direction_vector, fisher_diagonal=None)
        else:
            direction_vector = solver.step(direction_vector)
        direction_mean = direction_vector[:dimensionality]
        direction_covariance_svec = direction_vector[dimensionality:]
        covariance_direction = packed_direction_to_covariance(
            direction_covariance_svec, dimensionality
        )

        candidate_mean = variational_mean + step_size * direction_mean
        candidate_cholesky, covariance_step_scale = retract_covariance(
            cholesky,
            covariance_direction,
            step_size=step_size,
            max_covariance_log_step=max_covariance_log_step,
        )
        candidate_cholesky = _enforce_full_covariance_scale_bounds(
            candidate_mean,
            candidate_cholesky,
            min_variational_std,
            max_variational_std,
            bounded_parameter_handling,
            parameter_mins,
            parameter_maxes,
            transform_interior_margin,
            min_physical_variational_std_fraction,
            transform_map,
        )

        candidate_state = _evaluate_single_fidelity_state(
            model=model,
            observations=observations,
            observations_covariance=observations_covariance,
            run_directory_base=f"{absolute_work_dir}/iteration_{iteration}/run_",
            parameter_names=parameter_names,
            variational_mean=candidate_mean,
            cholesky=candidate_cholesky,
            prior_mean=prior_mean,
            prior_precision_operator=prior_precision_operator,
            prior_covariance_log_det=prior_covariance_log_det,
            sample_size=current_sample_size,
            evaluation_concurrency=evaluation_concurrency,
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
            elbo_scaling_factor=elbo_scaling_factor,
            log_likelihood_precision_operator=log_likelihood_precision_operator,
            sampling_method=sampling_method,
            dispatcher=dispatcher,
            score_function_entropy_strategy=score_function_entropy_strategy,
            standard_normal_samples=standard_normals,
        )

        ordinary_gradient = np.concatenate([
            state["gradient_mean"], state["gradient_covariance_svec"]
        ])
        effective_direction = np.concatenate([
            direction_mean,
            covariance_step_scale * direction_covariance_svec,
        ])
        predicted_slope = float(np.dot(ordinary_gradient, effective_direction))

        if optimization_method == "adam":
            accept_step = True
        elif line_search_objective == "mse":
            accept_step = candidate_state["mean_relative_mse"] <= (
                relaxation_parameter * state["mean_relative_mse"]
            )
        elif line_search_method == "legacy":
            allowable_drop = (relaxation_parameter - 1.0) * abs(state["elbo"])
            accept_step = candidate_state["elbo"] >= state["elbo"] - allowable_drop
        else:
            reference_elbo = max(accepted_elbo_history[-nonmonotone_window:])
            armijo_target = (
                reference_elbo
                + armijo_coefficient * step_size * max(predicted_slope, 0.0)
            )
            current_se = _vi._compute_state_elbo_standard_error(state, elbo_scaling_factor)
            candidate_se = _vi._compute_state_elbo_standard_error(
                candidate_state, elbo_scaling_factor
            )
            armijo_target -= uncertainty_sigma * np.sqrt(
                current_se**2 + candidate_se**2
            )
            accept_step = candidate_state["elbo"] >= armijo_target

        if accept_step:
            variational_mean = candidate_mean
            cholesky = candidate_cholesky
            state = candidate_state
            accepted_step_size = step_size
            accepted_elbo_history.append(float(state["elbo"]))
            failed_steps = 0
            standard_normal_cache = None
            if optimization_method != "adam":
                step_size = min(step_size * step_size_growth_factor, max_step_size)
            gradient_norm = _gradient_norm(state)
            wall_time = time.time() - start_time
            cpu_time = time.process_time() - start_cpu_time
            _append_history(
                history,
                variational_mean,
                cholesky,
                state,
                cpu_time,
                accepted_step_size,
                bounded_parameter_handling,
                parameter_mins,
                parameter_maxes,
                transform_interior_margin,
                transform_map,
            )
            _write_iteration_outputs(
                absolute_work_dir,
                iteration,
                variational_mean,
                cholesky,
                state,
                wall_time,
                cpu_time,
                bounded_parameter_handling,
                parameter_mins,
                parameter_maxes,
                transform_interior_margin,
                transform_map,
                dispatcher,
            )
            _save_full_covariance_restart(
                f"{absolute_work_dir}/iteration_{iteration}/restart.npz",
                variational_mean=variational_mean,
                cholesky=cholesky,
                state=state,
                prior_mean=prior_mean,
                prior_covariance=prior_covariance,
                iteration=iteration,
                step_size=step_size,
                bounded_parameter_handling=bounded_parameter_handling,
                transform_map=transform_map,
                parameter_mins=parameter_mins,
                parameter_maxes=parameter_maxes,
                transform_interior_margin=transform_interior_margin,
                baseline_method=baseline_method,
                score_function_entropy_strategy=score_function_entropy_strategy,
                optimization_method=optimization_method,
                gradient_method=gradient_method,
                sampling_method=sampling_method,
                elbo_scaling_factor=elbo_scaling_factor,
                accepted_elbo_history=accepted_elbo_history,
                max_covariance_log_step=max_covariance_log_step,
                adam_solver=adam_solver,
                dispatcher=dispatcher,
            )
            _vi._prune_old_restart_files(
                absolute_work_dir, restart_files_to_keep, dispatcher
            )
            relative_elbo = float(state["elbo"]) / (initial_elbo_reference + 1e-16)
            print(
                f"Iteration: {iteration}, Relative MSE: {state['mean_relative_mse']:.5f}, "
                f"ELBO: {state['elbo']:.5f}, Gradient norm: {gradient_norm:.5f}, "
                f"Covariance log-step scale: {covariance_step_scale:.3f}, "
                f"Wall time: {wall_time:.5f}"
            )
            iteration += 1
            if (
                elbo_relative_tolerance is not None
                and relative_elbo <= elbo_relative_tolerance
            ):
                elbo_converged = True
                break
        else:
            failed_steps += 1
            step_size /= step_size_decay_factor
            if failed_steps > max_step_size_decrease_trys:
                print(
                    f"Failed to advance after {max_step_size_decrease_trys} line-search reductions"
                )
                break

    _save_history(absolute_work_dir, history, dispatcher)
    final_std = marginal_std_from_cholesky(cholesky)
    if iteration >= max_iterations:
        print("Max iterations reached, terminating")
    elif elbo_converged:
        print("ELBO relative-improvement tolerance reached!")
    elif gradient_norm <= gradient_norm_tolerance:
        print("Gradient norm dropped below tolerance!")
    return variational_mean, final_std, state["parameter_samples"], state["qois"]


def run_vi(
    model,
    prior_parameter_space,
    observations: np.ndarray,
    observations_covariance: np.ndarray,
    parameter_mins: np.ndarray = None,
    parameter_maxes: np.ndarray = None,
    initial_variational_parameter_space=None,
    restart_file: str = None,
    optimizer_method: str = "gradient",
    optimizer_config=None,
    line_search_method: str = "stochastic_nonmonotone",
    line_search_config=None,
    absolute_work_dir: str = None,
    sample_size: int = 30,
    random_seed: int = 1,
    sampling_method: str = "mc",
    evaluation_concurrency=1,
    covariance_regularization: float = 1e-8,
    restart_files_to_keep: int = 10,
    elbo_scaling_factor="diag_mean",
    elbo_relative_tolerance: float = None,
    baseline_method: str = None,
    bounded_parameter_handling: str = "transform",
    transform_interior_margin: float = 1e-6,
    transform_map: str = "sigmoid",
    min_physical_variational_std_fraction: float = 1e-6,
    dispatcher=None,
    *,
    absolute_vi_directory: str = None,
    score_function_entropy_strategy: str = "analytic",
    variational_distribution: str = None,
    max_covariance_log_step: float = 1.0,
    create_run_directories: bool = True,
    sample_reuse_config=None,
):
    """Run VI, adding a true full-covariance Gaussian variational family."""
    distribution = _resolve_variational_distribution(
        variational_distribution,
        prior_parameter_space,
        initial_variational_parameter_space,
    )
    if distribution != "full_covariance":
        return _FALLBACK_RUN_VI(
            model,
            prior_parameter_space,
            observations,
            observations_covariance,
            parameter_mins=parameter_mins,
            parameter_maxes=parameter_maxes,
            initial_variational_parameter_space=initial_variational_parameter_space,
            restart_file=restart_file,
            optimizer_method=optimizer_method,
            optimizer_config=optimizer_config,
            line_search_method=line_search_method,
            line_search_config=line_search_config,
            absolute_work_dir=absolute_work_dir,
            sample_size=sample_size,
            random_seed=random_seed,
            sampling_method=sampling_method,
            evaluation_concurrency=evaluation_concurrency,
            covariance_regularization=covariance_regularization,
            restart_files_to_keep=restart_files_to_keep,
            elbo_scaling_factor=elbo_scaling_factor,
            elbo_relative_tolerance=elbo_relative_tolerance,
            baseline_method=baseline_method,
            bounded_parameter_handling=bounded_parameter_handling,
            transform_interior_margin=transform_interior_margin,
            transform_map=transform_map,
            min_physical_variational_std_fraction=min_physical_variational_std_fraction,
            dispatcher=dispatcher,
            absolute_vi_directory=absolute_vi_directory,
            score_function_entropy_strategy=score_function_entropy_strategy,
            create_run_directories=create_run_directories,
            sample_reuse_config=sample_reuse_config,
        )
    if sample_reuse_config is not None:
        raise NotImplementedError(
            "Sample reuse is not yet defined for the full-covariance variational family."
        )
    if absolute_vi_directory is not None:
        if absolute_work_dir is not None:
            raise TypeError("Specify only 'absolute_work_dir', not both directory arguments.")
        warnings.warn(
            "'absolute_vi_directory' is deprecated; use 'absolute_work_dir' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        absolute_work_dir = absolute_vi_directory
    with _directory_policy._run_directory_policy(create_run_directories):
        return _run_full_covariance_vi(
            model=model,
            prior_parameter_space=prior_parameter_space,
            observations=observations,
            observations_covariance=observations_covariance,
            parameter_mins=parameter_mins,
            parameter_maxes=parameter_maxes,
            initial_variational_parameter_space=initial_variational_parameter_space,
            restart_file=restart_file,
            optimizer_method=optimizer_method,
            optimizer_config=optimizer_config,
            line_search_method=line_search_method,
            line_search_config=line_search_config,
            absolute_work_dir=absolute_work_dir,
            sample_size=sample_size,
            random_seed=random_seed,
            sampling_method=sampling_method,
            evaluation_concurrency=evaluation_concurrency,
            covariance_regularization=covariance_regularization,
            restart_files_to_keep=restart_files_to_keep,
            elbo_scaling_factor=elbo_scaling_factor,
            elbo_relative_tolerance=elbo_relative_tolerance,
            baseline_method=baseline_method,
            bounded_parameter_handling=bounded_parameter_handling,
            transform_interior_margin=transform_interior_margin,
            transform_map=transform_map,
            min_physical_variational_std_fraction=min_physical_variational_std_fraction,
            dispatcher=dispatcher,
            score_function_entropy_strategy=score_function_entropy_strategy,
            max_covariance_log_step=max_covariance_log_step,
        )


# MF-VI is installed below in a separate commit so the single-fidelity path can
# be exercised independently while preserving the public fallback.
def run_mf_vi(*args, variational_distribution=None, max_covariance_log_step=1.0, **kwargs):
    distribution = _resolve_variational_distribution(
        variational_distribution,
        kwargs.get("prior_parameter_space", args[2] if len(args) > 2 else None),
        kwargs.get("initial_variational_parameter_space"),
    )
    if distribution == "full_covariance":
        raise NotImplementedError("Full-covariance MF-VI wiring is not installed yet.")
    return _FALLBACK_RUN_MF_VI(*args, **kwargs)


def mf_vi_with_auto_rom(*args, variational_distribution=None, max_covariance_log_step=1.0, **kwargs):
    distribution = _resolve_variational_distribution(
        variational_distribution,
        kwargs.get("prior_parameter_space", args[1] if len(args) > 1 else None),
        kwargs.get("initial_variational_parameter_space"),
    )
    if distribution == "full_covariance":
        raise NotImplementedError("Full-covariance auto-ROM MF-VI wiring is not installed yet.")
    return _FALLBACK_AUTO_MF_VI(*args, **kwargs)
