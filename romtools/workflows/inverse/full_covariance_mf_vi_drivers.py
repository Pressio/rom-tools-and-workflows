"""Full-covariance multifidelity variational inference.

The MF control variate acts on the complete packed Gaussian natural-score
contribution ``[delta, svec(delta delta.T - Sigma)]``. Analytic entropy is
added once after the stochastic HF/LF control-variate estimator.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from romtools.hpc.dispatchers import resolve_dispatcher
from romtools.workflows.inverse import mf_vi_drivers as _mf
from romtools.workflows.inverse import vi_drivers as _vi
from romtools.workflows.inverse import vi_run_directory_policy as _directory_policy
from romtools.workflows.inverse.full_covariance_vi import (
    covariance_from_cholesky,
    entropy as full_covariance_entropy,
    log_density as full_covariance_log_density,
    marginal_std_from_cholesky,
    natural_score_terms,
    packed_direction_to_covariance,
    retract_covariance,
    robust_cholesky,
    svec,
    svec_inverse,
)
from romtools.workflows.inverse.full_covariance_vi_drivers import (
    _FALLBACK_AUTO_MF_VI,
    _FALLBACK_RUN_MF_VI,
    _enforce_full_covariance_scale_bounds,
    _extract_prior_and_initial_moments,
    _gradient_norm,
    _new_history,
    _representation_from_cholesky,
    _resolve_variational_distribution,
    _update_vector,
)
from romtools.workflows.inverse.vi_optimization_methods import (
    AdamSolver,
    SteepestDescentSolver,
    VIAdamOptimizerConfig,
    VIGradientOptimizerConfig,
    VINewtonOptimizerConfig,
    VILegacyLineSearchConfig,
    VIStochasticNonmonotoneLineSearchConfig,
    _resolve_line_search_config,
    _resolve_optimizer_config,
)


def _loo(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return np.zeros_like(values)
    return (np.sum(values) - values) / float(values.size - 1)


def _optimal_baseline(values: np.ndarray, scores: np.ndarray) -> np.ndarray:
    numerator = np.mean(values[:, None] * scores**2, axis=0)
    denominator = np.mean(scores**2, axis=0)
    result = np.zeros(scores.shape[1], dtype=float)
    nonzero = denominator > 0.0
    result[nonzero] = numerator[nonzero] / denominator[nonzero]
    return result


def _component_standard_error(terms: np.ndarray) -> np.ndarray:
    if terms.shape[0] <= 1:
        return np.zeros(terms.shape[1], dtype=float)
    return np.std(terms, axis=0, ddof=1) / np.sqrt(terms.shape[0])


def _mfmc_standard_error(high_terms, low_base, low_extra, alpha, mode):
    high_flat = high_terms.reshape(high_terms.shape[0], -1)
    if low_extra is None or low_extra.shape[0] == 0:
        return _component_standard_error(high_flat)
    low_base_flat = low_base.reshape(low_base.shape[0], -1)
    low_extra_flat = low_extra.reshape(low_extra.shape[0], -1)
    ratio = low_extra_flat.shape[0] / (
        low_base_flat.shape[0] + low_extra_flat.shape[0]
    )
    mode = _mf._normalize_mfmc_control_variate_mode(mode)
    if mode == "matrix":
        beta = ratio * np.asarray(alpha)
        corrected_high = high_flat - low_base_flat @ beta
        projected_extra = low_extra_flat @ beta
        extra_se = _component_standard_error(projected_extra)
    elif mode == "scalar":
        beta = ratio * float(np.asarray(alpha))
        corrected_high = high_flat - beta * low_base_flat
        extra_se = abs(beta) * _component_standard_error(low_extra_flat)
    else:
        beta = ratio * np.asarray(alpha).reshape(-1)
        corrected_high = high_flat - beta[None, :] * low_base_flat
        extra_se = np.abs(beta) * _component_standard_error(low_extra_flat)
    return np.sqrt(_component_standard_error(corrected_high) ** 2 + extra_se**2)


def _ordinary_from_natural(packed_natural, dimensionality, cholesky):
    natural_mean = packed_natural[:dimensionality]
    natural_covariance_svec = packed_natural[dimensionality:]
    covariance = covariance_from_cholesky(cholesky)
    ordinary_mean = np.linalg.solve(covariance, natural_mean)
    natural_covariance = svec_inverse(natural_covariance_svec, dimensionality)
    left = np.linalg.solve(covariance, natural_covariance)
    ordinary_covariance = 0.5 * np.linalg.solve(covariance, left.T).T
    ordinary_covariance = 0.5 * (ordinary_covariance + ordinary_covariance.T)
    return np.concatenate([ordinary_mean, svec(ordinary_covariance)])


def _upgrade_mf_state(
    state,
    variational_mean,
    cholesky,
    baseline_method,
    use_mfmc_control_variate,
    mfmc_control_variate_mode,
    gradient_method,
    elbo_scaling_factor,
    score_function_entropy_strategy,
):
    n_fom = state["parameter_samples_fom"].shape[0]
    n_base = state["parameter_samples_rom_base"].shape[0]
    n_extra = state["parameter_samples_rom_only"].shape[0]
    optimizer_samples = np.asarray(state["optimizer_samples"])
    optimizer_fom = optimizer_samples[:n_fom]
    optimizer_base = optimizer_samples[n_fom:n_fom + n_base]
    optimizer_extra = optimizer_samples[n_fom + n_base:n_fom + n_base + n_extra]

    mean_fom, covariance_fom = natural_score_terms(
        optimizer_fom, variational_mean, cholesky
    )
    mean_base, covariance_base = natural_score_terms(
        optimizer_base, variational_mean, cholesky
    )
    if n_extra > 0:
        mean_extra, covariance_extra = natural_score_terms(
            optimizer_extra, variational_mean, cholesky
        )
    else:
        mean_extra = np.zeros((0, variational_mean.size))
        covariance_extra = np.zeros((0, covariance_fom.shape[1]))

    score_fom = np.hstack([mean_fom, covariance_fom])
    score_base = np.hstack([mean_base, covariance_base])
    score_extra = np.hstack([mean_extra, covariance_extra])

    weights_fom = elbo_scaling_factor * np.asarray(state["log_joint_terms_fom"])
    weights_base = elbo_scaling_factor * np.asarray(state["log_joint_terms_rom_base"])
    weights_extra = elbo_scaling_factor * np.asarray(state["log_joint_terms_rom_only"])
    entropy_strategy = score_function_entropy_strategy.strip().lower()
    log_q_fom = None
    if entropy_strategy == "joint":
        log_q_fom = full_covariance_log_density(
            optimizer_fom, variational_mean, cholesky
        )
        weights_fom -= elbo_scaling_factor * log_q_fom
        weights_base -= elbo_scaling_factor * full_covariance_log_density(
            optimizer_base, variational_mean, cholesky
        )
        if n_extra > 0:
            weights_extra -= elbo_scaling_factor * full_covariance_log_density(
                optimizer_extra, variational_mean, cholesky
            )
    elif entropy_strategy != "analytic":
        raise ValueError(
            "score_function_entropy_strategy must be 'analytic' or 'joint'"
        )

    baseline_method = baseline_method.strip().lower()
    if baseline_method == "optimal":
        baseline = _optimal_baseline(weights_fom, score_fom)
        centered_fom = weights_fom[:, None] - baseline[None, :]
        centered_base = weights_base[:, None] - baseline[None, :]
        centered_extra = weights_extra[:, None] - baseline[None, :]
    elif baseline_method == "loo":
        baseline = np.zeros(score_fom.shape[1], dtype=float)
        centered_fom = (weights_fom - _loo(weights_fom))[:, None]
        centered_base = (weights_base - _loo(weights_base))[:, None]
        centered_extra = (weights_extra - _loo(weights_extra))[:, None]
    elif baseline_method == "none":
        baseline = np.zeros(score_fom.shape[1], dtype=float)
        centered_fom = weights_fom[:, None]
        centered_base = weights_base[:, None]
        centered_extra = weights_extra[:, None]
    else:
        raise ValueError("baseline_method must be 'none', 'loo', or 'optimal'")

    high_terms = centered_fom * score_fom
    low_base_terms = centered_base * score_base
    low_extra_terms = centered_extra * score_extra if n_extra > 0 else None
    high_reference = None
    if entropy_strategy == "joint":
        high_reference = (
            np.abs(
                elbo_scaling_factor * np.asarray(state["log_joint_terms_fom"])
            )[:, None]
            + np.abs(elbo_scaling_factor * log_q_fom)[:, None]
        ) * np.abs(score_fom)

    natural_gradient, alpha = _mf._mfmc_gradient_estimator(
        high_terms,
        low_base_terms,
        low_extra_terms,
        use_mfmc_control_variate,
        mfmc_control_variate_mode,
        high_variance_reference_terms=high_reference,
    )
    natural_gradient = np.asarray(natural_gradient).reshape(-1)
    if entropy_strategy == "analytic":
        covariance = covariance_from_cholesky(cholesky)
        natural_gradient[variational_mean.size:] += (
            elbo_scaling_factor * svec(covariance)
        )

    standard_error = _mfmc_standard_error(
        high_terms,
        low_base_terms,
        low_extra_terms,
        alpha,
        mfmc_control_variate_mode,
    )
    noise_norm = float(np.linalg.norm(standard_error))
    signal_norm = float(np.linalg.norm(natural_gradient))
    if noise_norm == 0.0:
        snr = np.inf if signal_norm > 0.0 else 0.0
    else:
        snr = signal_norm / noise_norm
    ordinary_gradient = _ordinary_from_natural(
        natural_gradient, variational_mean.size, cholesky
    )
    if gradient_method == "natural":
        update = natural_gradient
    elif gradient_method == "standard":
        update = ordinary_gradient
    else:
        raise ValueError("gradient_method must be 'standard' or 'natural'")

    state = dict(state)
    dimensionality = variational_mean.size
    state.update(
        gradient_mean=ordinary_gradient[:dimensionality],
        gradient_covariance_svec=ordinary_gradient[dimensionality:],
        natural_gradient_mean=natural_gradient[:dimensionality],
        natural_gradient_covariance_svec=natural_gradient[dimensionality:],
        update_direction_mean=update[:dimensionality],
        update_direction_covariance_svec=update[dimensionality:],
        baseline_packed=baseline,
        gradient_standard_error=standard_error,
        gradient_signal_to_noise_ratio=float(snr),
        gradient_method=gradient_method,
        mfmc_alpha_packed=alpha,
    )
    state["entropy"] = full_covariance_entropy(cholesky, elbo_scaling_factor)
    if n_extra > 0:
        low_full = np.concatenate(
            [state["log_joint_terms_rom_base"], state["log_joint_terms_rom_only"]]
        )
        mean_joint = (
            np.mean(state["log_joint_terms_fom"])
            + np.mean(low_full)
            - np.mean(state["log_joint_terms_rom_base"])
        )
    else:
        mean_joint = np.mean(state["log_joint_terms_fom"])
    state["elbo"] = elbo_scaling_factor * float(mean_joint) + state["entropy"]
    return state


def _evaluate_mf_state(
    *,
    model,
    rom_model,
    rom_model_builder,
    observations,
    observations_covariance,
    iteration_directory,
    parameter_names,
    variational_mean,
    cholesky,
    prior_mean,
    prior_precision_operator,
    prior_covariance_log_det,
    fom_sample_size,
    rom_extra_sample_size,
    fom_evaluation_concurrency,
    rom_evaluation_concurrency,
    covariance_regularization,
    baseline_method,
    use_mfmc_control_variate,
    mfmc_control_variate_mode,
    elbo_scaling_factor,
    gradient_method,
    bounded_parameter_handling,
    min_variational_std,
    max_variational_std,
    rom_base_sampling_strategy,
    parameter_mins,
    parameter_maxes,
    transform_interior_margin,
    transform_map,
    rom_tolerance,
    max_rom_training_dirs,
    correlation_estimator,
    correlation_k_folds,
    training_dirs,
    training_parameters,
    training_qois,
    rom_training_dirs,
    rom_training_parameters,
    rom_training_qois,
    log_likelihood_precision_operator,
    sampling_method,
    dispatcher,
    score_function_entropy_strategy,
):
    _, _, variational_log_std, correlation_cholesky = (
        _representation_from_cholesky(cholesky)
    )
    state = _mf._evaluate_mf_vi_state(
        model=model,
        rom_model=rom_model,
        rom_model_builder=rom_model_builder,
        observations=observations,
        observations_covariance=observations_covariance,
        iteration_directory=iteration_directory,
        parameter_names=parameter_names,
        variational_mean=variational_mean,
        variational_log_std=variational_log_std,
        prior_mean=prior_mean,
        prior_precision_operator=prior_precision_operator,
        prior_covariance_log_det=prior_covariance_log_det,
        fom_sample_size=fom_sample_size,
        rom_extra_sample_size=rom_extra_sample_size,
        fom_evaluation_concurrency=fom_evaluation_concurrency,
        rom_evaluation_concurrency=rom_evaluation_concurrency,
        covariance_regularization=covariance_regularization,
        baseline_method=baseline_method,
        use_mfmc_control_variate=use_mfmc_control_variate,
        mfmc_control_variate_mode=mfmc_control_variate_mode,
        variational_correlation_cholesky=correlation_cholesky,
        elbo_scaling_factor=elbo_scaling_factor,
        gradient_method="standard",
        bounded_parameter_handling=bounded_parameter_handling,
        min_variational_std=min_variational_std,
        max_variational_std=max_variational_std,
        rom_base_sampling_strategy=rom_base_sampling_strategy,
        parameter_mins=parameter_mins,
        parameter_maxes=parameter_maxes,
        transform_interior_margin=transform_interior_margin,
        transform_map=transform_map,
        rom_tolerance=rom_tolerance,
        max_rom_training_dirs=max_rom_training_dirs,
        correlation_estimator=correlation_estimator,
        correlation_k_folds=correlation_k_folds,
        training_dirs=training_dirs,
        training_parameters=training_parameters,
        training_qois=training_qois,
        rom_training_dirs=rom_training_dirs,
        rom_training_parameters=rom_training_parameters,
        rom_training_qois=rom_training_qois,
        log_likelihood_precision_operator=log_likelihood_precision_operator,
        sampling_method=sampling_method,
        dispatcher=dispatcher,
        score_function_entropy_strategy=score_function_entropy_strategy,
    )
    return _upgrade_mf_state(
        state,
        variational_mean,
        cholesky,
        baseline_method,
        use_mfmc_control_variate,
        mfmc_control_variate_mode,
        gradient_method,
        elbo_scaling_factor,
        score_function_entropy_strategy,
    )


def _save_mf_restart(
    path,
    *,
    variational_mean,
    cholesky,
    state,
    iteration,
    step_size,
    bounded_parameter_handling,
    parameter_mins,
    parameter_maxes,
    transform_interior_margin,
    transform_map,
    optimization_method,
    gradient_method,
    sampling_method,
    baseline_method,
    score_function_entropy_strategy,
    mfmc_control_variate_mode,
    elbo_scaling_factor,
    accepted_elbo_history,
    max_covariance_log_step,
    adam_solver,
    dispatcher,
):
    data = {
        "variational_mean": _vi._get_persisted_variational_mean(
            variational_mean,
            bounded_parameter_handling,
            parameter_mins,
            parameter_maxes,
            transform_interior_margin,
            transform_map,
        ),
        "variational_mean_coordinates": "physical",
        "variational_cholesky_optimizer": cholesky,
        "variational_distribution": "full_covariance",
        "iteration": int(iteration),
        "step_size": float(step_size),
        "bounded_parameter_handling": bounded_parameter_handling,
        "transform_map": transform_map,
        "optimization_method": optimization_method,
        "gradient_method": gradient_method,
        "sampling_method": sampling_method,
        "baseline_method": baseline_method,
        "score_function_entropy_strategy": score_function_entropy_strategy,
        "mfmc_control_variate_mode": mfmc_control_variate_mode,
        "elbo_scaling_factor": float(elbo_scaling_factor),
        "accepted_elbo_history": np.asarray(accepted_elbo_history),
        "max_covariance_log_step": float(max_covariance_log_step),
        "rng_state": np.array(np.random.get_state(), dtype=object),
        "training_directories": np.array(state["training_dirs"]),
        "rom_training_directories": np.array(state["rom_training_dirs"]),
        "training_parameters": state["training_parameters"],
        "training_qois": state["training_qois"],
        "rom_training_parameters": state["rom_training_parameters"],
        "rom_training_qois": state["rom_training_qois"],
    }
    if adam_solver is not None:
        data.update(adam_solver.restart_state_dict())
        data["adam_gradient_method"] = gradient_method
    dispatcher.np_savez(path, **data)


def _append_history(history, mean, cholesky, state, cpu_time, step_size, **kwargs):
    history["variational_mean"].append(
        _vi._get_persisted_variational_mean(
            mean,
            kwargs["bounded_parameter_handling"],
            kwargs["parameter_mins"],
            kwargs["parameter_maxes"],
            kwargs["transform_interior_margin"],
            kwargs["transform_map"],
        )
    )
    history["variational_covariance"].append(covariance_from_cholesky(cholesky))
    history["relative_mse"].append(float(state["mean_relative_mse"]))
    history["loglikelihood"].append(float(np.mean(state["log_likelihoods_fom"])))
    history["elbo"].append(float(state["elbo"]))
    history["cpu_time_seconds"].append(float(cpu_time))
    history["accepted_step_size"].append(float(step_size))
    history["gradient"].append(_update_vector(state).copy())
    history["gradient_standard_error"].append(
        np.asarray(state["gradient_standard_error"]).copy()
    )


def _save_history(absolute_work_dir, history, state, dispatcher):
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
        vi_history_gradient_standard_error=np.asarray(
            history["gradient_standard_error"]
        ),
        mfmc_alpha_packed=np.asarray(state["mfmc_alpha_packed"]),
        variational_family="full_covariance",
    )


def _run_full_covariance_mf_vi(
    *,
    model,
    rom_model_builder,
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
    fom_sample_size,
    rom_extra_sample_size,
    rom_tolerance,
    max_rom_training_history,
    random_seed,
    sampling_method,
    fom_evaluation_concurrency,
    rom_evaluation_concurrency,
    covariance_regularization,
    restart_files_to_keep,
    correlation_estimator,
    correlation_k_folds,
    elbo_scaling_factor,
    elbo_relative_tolerance,
    baseline_method,
    use_mfmc_control_variate,
    mfmc_control_variate_mode,
    rom_base_sampling_strategy,
    bounded_parameter_handling,
    transform_interior_margin,
    transform_map,
    min_physical_variational_std_fraction,
    dispatcher,
    score_function_entropy_strategy,
    max_covariance_log_step,
):
    dispatcher = resolve_dispatcher(dispatcher)
    dispatcher.require_supported_concurrency(fom_evaluation_concurrency)
    if absolute_work_dir is None:
        absolute_work_dir = os.getcwd() + "/work/"
    if max_covariance_log_step <= 0.0:
        raise ValueError("max_covariance_log_step must be positive")
    if fom_sample_size <= 1 or rom_extra_sample_size < 0:
        raise ValueError(
            "fom_sample_size must exceed one and rom_extra_sample_size be nonnegative"
        )

    optimization_method, config = _resolve_optimizer_config(
        optimizer_method,
        optimizer_config,
        VIGradientOptimizerConfig(),
        VINewtonOptimizerConfig(newton_regularization=1e-8),
        VIAdamOptimizerConfig(),
    )
    if optimization_method == "newton":
        raise NotImplementedError(
            "Full-covariance Newton/Hessian support is intentionally out of scope"
        )
    gradient_method = config.gradient_method.strip().lower()
    if gradient_method not in ("standard", "natural"):
        raise ValueError("gradient_method must be 'standard' or 'natural'")
    gradient_norm_tolerance = config.gradient_norm_tolerance
    max_iterations = config.max_iterations
    min_variational_std = config.min_variational_std
    max_variational_std = config.max_variational_std

    if optimization_method == "adam":
        if line_search_config is not None:
            raise ValueError("line_search_config is not supported with Adam")
        line_search_method = "legacy"
        line_config = VILegacyLineSearchConfig(
            initial_step_size=1.0,
            max_step_size=1.0,
            step_size_growth_factor=1.0,
            step_size_decay_factor=1.0,
            max_step_size_decrease_trys=0,
            relaxation_parameter=1.0,
            line_search_sample_growth_factor=1.0,
            log_std_learning_rate_factor=1.0,
        )
    else:
        line_search_method, line_config = _resolve_line_search_config(
            line_search_method,
            line_search_config,
            VILegacyLineSearchConfig(
                step_size_growth_factor=1.05,
                relaxation_parameter=10.0,
            ),
            VIStochasticNonmonotoneLineSearchConfig(
                step_size_growth_factor=1.05,
                relaxation_parameter=10.0,
            ),
        )
    step_size = min(line_config.initial_step_size, line_config.max_step_size)
    sample_growth = line_config.line_search_sample_growth_factor
    nonmonotone_window = getattr(
        line_config, "line_search_nonmonotone_window", 1
    )
    armijo = getattr(line_config, "line_search_armijo_coefficient", 0.0)
    uncertainty = getattr(line_config, "line_search_uncertainty_sigma", 0.0)

    bounded_parameter_handling = _vi._normalize_bounded_parameter_handling(
        bounded_parameter_handling
    )
    transform_map = _vi._normalize_transform_map(transform_map)
    sampling_method = _vi._normalize_sampling_method(sampling_method)
    baseline_method = _vi._normalize_baseline_method(
        "loo" if baseline_method is None else baseline_method
    )
    score_function_entropy_strategy = _vi._normalize_score_function_entropy_strategy(
        score_function_entropy_strategy
    )
    mfmc_control_variate_mode = _mf._normalize_mfmc_control_variate_mode(
        mfmc_control_variate_mode
    )
    rom_base_sampling_strategy = _mf._normalize_rom_base_sampling_strategy(
        rom_base_sampling_strategy
    )
    correlation_estimator = _mf._normalize_correlation_estimator(
        correlation_estimator
    )
    elbo_scaling_factor = _vi._resolve_elbo_scaling_factor(
        elbo_scaling_factor, observations_covariance
    )
    parameter_mins, parameter_maxes = _vi._resolve_parameter_bounds(
        parameter_mins, parameter_maxes
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
    log_likelihood_precision = _vi._compute_log_likelihood_precision_operator(
        observations_covariance, covariance_regularization
    )
    prior_precision, prior_log_det = _vi._compute_gaussian_log_density_data(
        prior_covariance
    )
    max_rom_training_dirs = int(max_rom_training_history * fom_sample_size)
    if max_rom_training_dirs < 1:
        raise ValueError("max_rom_training_history must be positive")

    history = _new_history()
    iteration = 0
    rom_model = None
    training_dirs = []
    training_parameters = None
    training_qois = None
    rom_training_dirs = []
    rom_training_parameters = None
    rom_training_qois = None

    if restart_file is None:
        np.random.seed(random_seed)
        optimizer_mean, optimizer_covariance = (
            _vi._convert_physical_moments_to_optimizer_moments(
                initial_mean,
                initial_covariance,
                bounded_parameter_handling,
                parameter_mins,
                parameter_maxes,
                transform_interior_margin,
                transform_map,
            )
        )
        mean = optimizer_mean
        cholesky = robust_cholesky(optimizer_covariance)
    else:
        restart_data = np.load(restart_file, allow_pickle=True)
        if str(restart_data["variational_distribution"].item()) != "full_covariance":
            raise ValueError("restart_file is not a full-covariance MF-VI restart")
        if "rng_state" in restart_data:
            np.random.set_state(tuple(restart_data["rng_state"].tolist()))
        else:
            np.random.seed(random_seed)
        iteration = int(restart_data["iteration"])
        step_size = min(
            float(restart_data["step_size"]), line_config.max_step_size
        )
        mean = _vi._restore_variational_mean_from_restart(
            restart_data,
            bounded_parameter_handling,
            parameter_mins,
            parameter_maxes,
            transform_interior_margin,
            transform_map,
        )
        cholesky = np.asarray(
            restart_data["variational_cholesky_optimizer"], dtype=float
        )
        training_dirs = restart_data["training_directories"].tolist()
        rom_training_dirs = restart_data["rom_training_directories"].tolist()
        training_parameters = restart_data["training_parameters"]
        training_qois = restart_data["training_qois"]
        rom_training_parameters = restart_data["rom_training_parameters"]
        rom_training_qois = restart_data["rom_training_qois"]
        rom_model = rom_model_builder.build_from_training_dirs(
            f"{absolute_work_dir}/iteration_{iteration}",
            rom_training_dirs,
            rom_training_parameters,
            rom_training_qois,
        )

    cholesky = _enforce_full_covariance_scale_bounds(
        mean,
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
    state = _evaluate_mf_state(
        model=model,
        rom_model=rom_model,
        rom_model_builder=rom_model_builder,
        observations=observations,
        observations_covariance=observations_covariance,
        iteration_directory=f"{absolute_work_dir}/iteration_{iteration}",
        parameter_names=parameter_names,
        variational_mean=mean,
        cholesky=cholesky,
        prior_mean=prior_mean,
        prior_precision_operator=prior_precision,
        prior_covariance_log_det=prior_log_det,
        fom_sample_size=fom_sample_size,
        rom_extra_sample_size=rom_extra_sample_size,
        fom_evaluation_concurrency=fom_evaluation_concurrency,
        rom_evaluation_concurrency=rom_evaluation_concurrency,
        covariance_regularization=covariance_regularization,
        baseline_method=baseline_method,
        use_mfmc_control_variate=use_mfmc_control_variate,
        mfmc_control_variate_mode=mfmc_control_variate_mode,
        elbo_scaling_factor=elbo_scaling_factor,
        gradient_method=gradient_method,
        bounded_parameter_handling=bounded_parameter_handling,
        min_variational_std=min_variational_std,
        max_variational_std=max_variational_std,
        rom_base_sampling_strategy=rom_base_sampling_strategy,
        parameter_mins=parameter_mins,
        parameter_maxes=parameter_maxes,
        transform_interior_margin=transform_interior_margin,
        transform_map=transform_map,
        rom_tolerance=rom_tolerance,
        max_rom_training_dirs=max_rom_training_dirs,
        correlation_estimator=correlation_estimator,
        correlation_k_folds=correlation_k_folds,
        training_dirs=training_dirs,
        training_parameters=training_parameters,
        training_qois=training_qois,
        rom_training_dirs=rom_training_dirs,
        rom_training_parameters=rom_training_parameters,
        rom_training_qois=rom_training_qois,
        log_likelihood_precision_operator=log_likelihood_precision,
        sampling_method=sampling_method,
        dispatcher=dispatcher,
        score_function_entropy_strategy=score_function_entropy_strategy,
    )
    accepted_elbo_history = [float(state["elbo"])]
    initial_elbo_reference = float(state["elbo"])

    solver = (
        AdamSolver.from_config(config)
        if optimization_method == "adam"
        else SteepestDescentSolver()
    )
    if optimization_method == "adam":
        solver.parameter_dimension = dimensionality
        if restart_file is not None and "adam_iteration" in restart_data:
            solver.load_restart_state_dict(restart_data)
    adam_solver = solver if optimization_method == "adam" else None

    gradient_norm = _gradient_norm(state)
    _append_history(
        history,
        mean,
        cholesky,
        state,
        0.0,
        np.nan,
        bounded_parameter_handling=bounded_parameter_handling,
        parameter_mins=parameter_mins,
        parameter_maxes=parameter_maxes,
        transform_interior_margin=transform_interior_margin,
        transform_map=transform_map,
    )
    _save_mf_restart(
        f"{absolute_work_dir}/iteration_{iteration}/restart.npz",
        variational_mean=mean,
        cholesky=cholesky,
        state=state,
        iteration=iteration,
        step_size=step_size,
        bounded_parameter_handling=bounded_parameter_handling,
        parameter_mins=parameter_mins,
        parameter_maxes=parameter_maxes,
        transform_interior_margin=transform_interior_margin,
        transform_map=transform_map,
        optimization_method=optimization_method,
        gradient_method=gradient_method,
        sampling_method=sampling_method,
        baseline_method=baseline_method,
        score_function_entropy_strategy=score_function_entropy_strategy,
        mfmc_control_variate_mode=mfmc_control_variate_mode,
        elbo_scaling_factor=elbo_scaling_factor,
        accepted_elbo_history=accepted_elbo_history,
        max_covariance_log_step=max_covariance_log_step,
        adam_solver=adam_solver,
        dispatcher=dispatcher,
    )
    _vi._prune_old_restart_files(
        absolute_work_dir, restart_files_to_keep, dispatcher
    )

    iteration += 1
    failed_steps = 0
    elbo_converged = False
    while iteration < max_iterations and gradient_norm > gradient_norm_tolerance:
        current_fom_size = max(
            2, int(np.ceil(fom_sample_size * sample_growth**failed_steps))
        )
        current_rom_extra = (
            0
            if rom_extra_sample_size == 0
            else max(
                1,
                int(
                    np.ceil(
                        rom_extra_sample_size * sample_growth**failed_steps
                    )
                ),
            )
        )
        direction = _update_vector(state)
        if optimization_method == "adam":
            direction = solver.step(direction, fisher_diagonal=None)
        else:
            direction = solver.step(direction)
        direction_mean = direction[:dimensionality]
        direction_covariance = direction[dimensionality:]
        candidate_mean = mean + step_size * direction_mean
        candidate_cholesky, covariance_scale = retract_covariance(
            cholesky,
            packed_direction_to_covariance(
                direction_covariance, dimensionality
            ),
            step_size,
            max_covariance_log_step,
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
        candidate = _evaluate_mf_state(
            model=model,
            rom_model=state["rom_model"],
            rom_model_builder=rom_model_builder,
            observations=observations,
            observations_covariance=observations_covariance,
            iteration_directory=f"{absolute_work_dir}/iteration_{iteration}",
            parameter_names=parameter_names,
            variational_mean=candidate_mean,
            cholesky=candidate_cholesky,
            prior_mean=prior_mean,
            prior_precision_operator=prior_precision,
            prior_covariance_log_det=prior_log_det,
            fom_sample_size=current_fom_size,
            rom_extra_sample_size=current_rom_extra,
            fom_evaluation_concurrency=fom_evaluation_concurrency,
            rom_evaluation_concurrency=rom_evaluation_concurrency,
            covariance_regularization=covariance_regularization,
            baseline_method=baseline_method,
            use_mfmc_control_variate=use_mfmc_control_variate,
            mfmc_control_variate_mode=mfmc_control_variate_mode,
            elbo_scaling_factor=elbo_scaling_factor,
            gradient_method=gradient_method,
            bounded_parameter_handling=bounded_parameter_handling,
            min_variational_std=min_variational_std,
            max_variational_std=max_variational_std,
            rom_base_sampling_strategy=rom_base_sampling_strategy,
            parameter_mins=parameter_mins,
            parameter_maxes=parameter_maxes,
            transform_interior_margin=transform_interior_margin,
            transform_map=transform_map,
            rom_tolerance=rom_tolerance,
            max_rom_training_dirs=max_rom_training_dirs,
            correlation_estimator=correlation_estimator,
            correlation_k_folds=correlation_k_folds,
            training_dirs=state["training_dirs"],
            training_parameters=state["training_parameters"],
            training_qois=state["training_qois"],
            rom_training_dirs=state["rom_training_dirs"],
            rom_training_parameters=state["rom_training_parameters"],
            rom_training_qois=state["rom_training_qois"],
            log_likelihood_precision_operator=log_likelihood_precision,
            sampling_method=sampling_method,
            dispatcher=dispatcher,
            score_function_entropy_strategy=score_function_entropy_strategy,
        )
        ordinary_gradient = np.concatenate(
            [state["gradient_mean"], state["gradient_covariance_svec"]]
        )
        effective_direction = np.concatenate(
            [direction_mean, covariance_scale * direction_covariance]
        )
        predicted_slope = float(
            np.dot(ordinary_gradient, effective_direction)
        )

        if optimization_method == "adam":
            accept = True
        elif line_search_method == "legacy":
            allowed_drop = (
                line_config.relaxation_parameter - 1.0
            ) * abs(state["elbo"])
            accept = candidate["elbo"] >= state["elbo"] - allowed_drop
        else:
            reference = max(
                accepted_elbo_history[-nonmonotone_window:]
            )
            target = (
                reference
                + armijo * step_size * max(predicted_slope, 0.0)
            )
            current_se = _mf._compute_state_elbo_standard_error(
                state, elbo_scaling_factor
            )
            candidate_se = _mf._compute_state_elbo_standard_error(
                candidate, elbo_scaling_factor
            )
            target -= uncertainty * np.sqrt(
                current_se**2 + candidate_se**2
            )
            accept = candidate["elbo"] >= target

        if accept:
            mean = candidate_mean
            cholesky = candidate_cholesky
            state = candidate
            accepted_elbo_history.append(float(state["elbo"]))
            accepted_step = step_size
            failed_steps = 0
            if optimization_method != "adam":
                step_size = min(
                    step_size * line_config.step_size_growth_factor,
                    line_config.max_step_size,
                )
            gradient_norm = _gradient_norm(state)
            _append_history(
                history,
                mean,
                cholesky,
                state,
                float(iteration),
                accepted_step,
                bounded_parameter_handling=bounded_parameter_handling,
                parameter_mins=parameter_mins,
                parameter_maxes=parameter_maxes,
                transform_interior_margin=transform_interior_margin,
                transform_map=transform_map,
            )
            _save_mf_restart(
                f"{absolute_work_dir}/iteration_{iteration}/restart.npz",
                variational_mean=mean,
                cholesky=cholesky,
                state=state,
                iteration=iteration,
                step_size=step_size,
                bounded_parameter_handling=bounded_parameter_handling,
                parameter_mins=parameter_mins,
                parameter_maxes=parameter_maxes,
                transform_interior_margin=transform_interior_margin,
                transform_map=transform_map,
                optimization_method=optimization_method,
                gradient_method=gradient_method,
                sampling_method=sampling_method,
                baseline_method=baseline_method,
                score_function_entropy_strategy=score_function_entropy_strategy,
                mfmc_control_variate_mode=mfmc_control_variate_mode,
                elbo_scaling_factor=elbo_scaling_factor,
                accepted_elbo_history=accepted_elbo_history,
                max_covariance_log_step=max_covariance_log_step,
                adam_solver=adam_solver,
                dispatcher=dispatcher,
            )
            _vi._prune_old_restart_files(
                absolute_work_dir, restart_files_to_keep, dispatcher
            )
            relative_elbo = float(state["elbo"]) / (
                initial_elbo_reference + 1e-16
            )
            print(
                f"Iteration: {iteration}, Relative MSE: "
                f"{state['mean_relative_mse']:.5f}, ELBO: {state['elbo']:.5f}, "
                f"ROM err: {state['rom_error']:.5f}, Gradient norm: "
                f"{gradient_norm:.5f}, covariance scale: {covariance_scale:.3f}"
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
            step_size /= line_config.step_size_decay_factor
            if failed_steps > line_config.max_step_size_decrease_trys:
                break

    _save_history(absolute_work_dir, history, state, dispatcher)
    final_std = marginal_std_from_cholesky(cholesky)
    if elbo_converged:
        print("ELBO relative-improvement tolerance reached!")
    return (
        mean,
        final_std,
        state["parameter_samples_fom"],
        state["qois_fom"],
    )


def run_mf_vi(
    model,
    rom_model_builder,
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
    fom_sample_size: int = 10,
    rom_extra_sample_size: int = 30,
    rom_tolerance: float = 0.005,
    max_rom_training_history: int = 1,
    random_seed: int = 1,
    sampling_method: str = "mc",
    fom_evaluation_concurrency=10,
    rom_evaluation_concurrency=1,
    covariance_regularization: float = 1e-7,
    restart_files_to_keep: int = 10,
    correlation_estimator: str = "in_sample",
    correlation_k_folds: int = 5,
    elbo_scaling_factor="diag_mean",
    elbo_relative_tolerance: float = None,
    baseline_method: str = None,
    use_mfmc_control_variate: bool = True,
    mfmc_control_variate_mode: str = "componentwise",
    rom_base_sampling_strategy: str = "coupled",
    bounded_parameter_handling: str = "transform",
    transform_interior_margin: float = 1e-8,
    transform_map: str = "sigmoid",
    min_physical_variational_std_fraction: float = 1e-8,
    dispatcher=None,
    *,
    absolute_vi_directory: str = None,
    score_function_entropy_strategy: str = "analytic",
    variational_distribution: str = None,
    max_covariance_log_step: float = 1.0,
    create_run_directories: bool = True,
    sample_reuse_config=None,
):
    """Run MF-VI with an optional true full-covariance Gaussian family."""
    distribution = _resolve_variational_distribution(
        variational_distribution,
        prior_parameter_space,
        initial_variational_parameter_space,
    )
    if distribution != "full_covariance":
        return _FALLBACK_RUN_MF_VI(
            model,
            rom_model_builder,
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
            fom_sample_size=fom_sample_size,
            rom_extra_sample_size=rom_extra_sample_size,
            rom_tolerance=rom_tolerance,
            max_rom_training_history=max_rom_training_history,
            random_seed=random_seed,
            sampling_method=sampling_method,
            fom_evaluation_concurrency=fom_evaluation_concurrency,
            rom_evaluation_concurrency=rom_evaluation_concurrency,
            covariance_regularization=covariance_regularization,
            restart_files_to_keep=restart_files_to_keep,
            correlation_estimator=correlation_estimator,
            correlation_k_folds=correlation_k_folds,
            elbo_scaling_factor=elbo_scaling_factor,
            elbo_relative_tolerance=elbo_relative_tolerance,
            baseline_method=baseline_method,
            use_mfmc_control_variate=use_mfmc_control_variate,
            mfmc_control_variate_mode=mfmc_control_variate_mode,
            rom_base_sampling_strategy=rom_base_sampling_strategy,
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
            "Sample reuse is not yet defined for full-covariance MF-VI"
        )
    if absolute_vi_directory is not None:
        if absolute_work_dir is not None:
            raise TypeError(
                "Specify only 'absolute_work_dir', not both directory arguments."
            )
        warnings.warn(
            "'absolute_vi_directory' is deprecated; use 'absolute_work_dir'.",
            DeprecationWarning,
            stacklevel=2,
        )
        absolute_work_dir = absolute_vi_directory
    with _directory_policy._run_directory_policy(create_run_directories):
        return _run_full_covariance_mf_vi(
            model=model,
            rom_model_builder=rom_model_builder,
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
            fom_sample_size=fom_sample_size,
            rom_extra_sample_size=rom_extra_sample_size,
            rom_tolerance=rom_tolerance,
            max_rom_training_history=max_rom_training_history,
            random_seed=random_seed,
            sampling_method=sampling_method,
            fom_evaluation_concurrency=fom_evaluation_concurrency,
            rom_evaluation_concurrency=rom_evaluation_concurrency,
            covariance_regularization=covariance_regularization,
            restart_files_to_keep=restart_files_to_keep,
            correlation_estimator=correlation_estimator,
            correlation_k_folds=correlation_k_folds,
            elbo_scaling_factor=elbo_scaling_factor,
            elbo_relative_tolerance=elbo_relative_tolerance,
            baseline_method=baseline_method,
            use_mfmc_control_variate=use_mfmc_control_variate,
            mfmc_control_variate_mode=mfmc_control_variate_mode,
            rom_base_sampling_strategy=rom_base_sampling_strategy,
            bounded_parameter_handling=bounded_parameter_handling,
            transform_interior_margin=transform_interior_margin,
            transform_map=transform_map,
            min_physical_variational_std_fraction=min_physical_variational_std_fraction,
            dispatcher=dispatcher,
            score_function_entropy_strategy=score_function_entropy_strategy,
            max_covariance_log_step=max_covariance_log_step,
        )


def mf_vi_with_auto_rom(
    *args,
    variational_distribution=None,
    max_covariance_log_step=1.0,
    **kwargs,
):
    """Delegate the existing auto-ROM API unless full covariance is requested."""
    prior = kwargs.get(
        "prior_parameter_space", args[1] if len(args) > 1 else None
    )
    distribution = _resolve_variational_distribution(
        variational_distribution,
        prior,
        kwargs.get("initial_variational_parameter_space"),
    )
    if distribution == "full_covariance":
        raise NotImplementedError(
            "For full-covariance MF-VI, construct the ROM builder explicitly and "
            "call run_mf_vi. Auto-ROM wiring will be added separately."
        )
    return _FALLBACK_AUTO_MF_VI(*args, **kwargs)
