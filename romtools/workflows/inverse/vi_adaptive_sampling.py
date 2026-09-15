"""Adaptive small-batch sampling for Newton VI and MFVI.

The public wrappers in this module are opt-in. When ``adaptive_sample_config``
is ``None`` they delegate to the existing VI/MFVI wrapper stack unchanged.
When enabled for Newton, the current Monte Carlo batch is enriched until a
delete-one jackknife estimate of the covariance of the regularized Newton step
is sufficiently small, or a configured maximum sample size is reached.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import inspect
from typing import Optional

import numpy as np

from romtools.hpc.dispatchers import resolve_dispatcher, resolve_local_dispatcher
from romtools.workflows.inverse import mf_vi_drivers as _mf
from romtools.workflows.inverse import vi_drivers as _vi
from romtools.workflows.inverse.vi_optimization_methods import (
    VILegacyLineSearchConfig,
    VINewtonOptimizerConfig,
    _resolve_line_search_config,
    _resolve_optimizer_config,
)
from romtools.workflows.inverse.vi_run_directory_policy import (
    mf_vi_with_auto_rom as _BASE_MF_VI_WITH_AUTO_ROM,
    run_mf_vi as _BASE_RUN_MF_VI,
    run_vi as _BASE_RUN_VI,
)
from romtools.workflows.inverse import vi_sample_reuse as _reuse


_ORIGINAL_RUN_VI = _reuse._ORIGINAL_RUN_VI
_ORIGINAL_RUN_MF_VI = _reuse._ORIGINAL_RUN_MF_VI
_ORIGINAL_MF_VI_WITH_AUTO_ROM = _reuse._ORIGINAL_MF_VI_WITH_AUTO_ROM
_ORIGINAL_EVALUATE_VI_STATE = _reuse._ORIGINAL_EVALUATE_VI_STATE
_ORIGINAL_EVALUATE_VI_CANDIDATE = _reuse._ORIGINAL_EVALUATE_VI_CANDIDATE
_ORIGINAL_EVALUATE_MF_VI_STATE = _reuse._ORIGINAL_EVALUATE_MF_VI_STATE


@dataclass(frozen=True)
class VIAdaptiveSampleConfig:
    """Configuration for jackknife-based adaptive Newton sample enrichment.

    The acceptance rule is

    ``sigma_step <= absolute_tolerance + relative_tolerance * max(||step||, step_norm_floor)``.

    The delete-one jackknife is applied to the *regularized Newton step*, so the
    diagnostic includes model-dependent gradient variance, Hessian variance,
    their correlation, and (for MFVI) the fitted control-variate coefficients.
    """

    enabled: bool = True
    max_sample_size: int = 64
    growth_factor: float = 2.0
    relative_tolerance: float = 0.25
    absolute_tolerance: float = 0.0
    step_norm_floor: float = 5.0e-2
    disable_backtracking: bool = False

    def __post_init__(self):
        if self.max_sample_size < 2:
            raise ValueError("max_sample_size must be at least 2")
        if not np.isfinite(self.growth_factor) or self.growth_factor <= 1.0:
            raise ValueError("growth_factor must be finite and greater than 1")
        if self.relative_tolerance < 0.0 or not np.isfinite(self.relative_tolerance):
            raise ValueError("relative_tolerance must be finite and non-negative")
        if self.absolute_tolerance < 0.0 or not np.isfinite(self.absolute_tolerance):
            raise ValueError("absolute_tolerance must be finite and non-negative")
        if self.step_norm_floor < 0.0 or not np.isfinite(self.step_norm_floor):
            raise ValueError("step_norm_floor must be finite and non-negative")


@dataclass
class _AdaptiveContext:
    config: VIAdaptiveSampleConfig
    newton_config: VINewtonOptimizerConfig


_ACTIVE_CONTEXT: ContextVar[Optional[_AdaptiveContext]] = ContextVar(
    "romtools_vi_adaptive_sample_context", default=None
)


def _bind(signature_source, args, kwargs):
    signature = inspect.signature(signature_source)
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


def _resolve_newton_context(signature_source, args, kwargs, config):
    if config is None or not config.enabled:
        return None
    arguments = _bind(signature_source, args, kwargs)
    optimizer_method = arguments.get("optimizer_method", "gradient")
    optimizer_config = arguments.get("optimizer_config", None)
    method, resolved = _resolve_optimizer_config(
        optimizer_method,
        optimizer_config,
        default_newton_config=VINewtonOptimizerConfig(),
    )
    if method != "newton":
        raise ValueError("adaptive_sample_config is currently supported only for optimizer_method='newton'")
    sampling_method = str(arguments.get("sampling_method", "mc")).strip().lower()
    if sampling_method not in ("mc", "montecarlo", "monte_carlo"):
        raise NotImplementedError(
            "Jackknife adaptive sampling currently supports sampling_method='mc' only."
        )
    sample_reuse_config = kwargs.get("sample_reuse_config")
    if sample_reuse_config is not None and getattr(sample_reuse_config, "enabled", False):
        raise NotImplementedError(
            "adaptive_sample_config is for standard VI/MFVI and cannot be combined with sample_reuse_config."
        )
    return _AdaptiveContext(config=config, newton_config=resolved)


def _single_candidate_line_search(signature_source, args, kwargs):
    """Disable repeated Newton backtracking while retaining the next-state evaluation."""
    arguments = _bind(signature_source, args, kwargs)
    method = arguments.get("line_search_method", "legacy")
    config = arguments.get("line_search_config", None)
    _, resolved = _resolve_line_search_config(method, config)
    permissive = VILegacyLineSearchConfig(
        initial_step_size=resolved.initial_step_size,
        max_step_size=resolved.max_step_size,
        step_size_growth_factor=resolved.step_size_growth_factor,
        step_size_decay_factor=1.0,
        max_step_size_decrease_trys=0,
        relaxation_parameter=1.0e12,
        line_search_objective="elbo",
        line_search_sample_growth_factor=1.0,
        log_std_learning_rate_factor=resolved.log_std_learning_rate_factor,
    )
    kwargs = dict(kwargs)
    kwargs["line_search_method"] = "legacy"
    kwargs["line_search_config"] = permissive
    return args, kwargs


def _newton_step(state, variational_log_std, min_variational_std, max_variational_std, ctx):
    variational_std, _ = _vi._compute_variational_std(
        variational_log_std, min_variational_std, max_variational_std
    )
    metric_scale = _vi._compute_newton_metric_scale(
        ctx.newton_config.newton_metric, variational_std
    )
    mean_step, log_std_step = _vi._compute_newton_step(
        state,
        ctx.newton_config.newton_regularization,
        newton_hessian_type=ctx.newton_config.newton_hessian_type,
        metric_scale=metric_scale,
    )
    return np.concatenate([mean_step, log_std_step])


def _jackknife_covariance(steps: np.ndarray) -> np.ndarray:
    """Classical delete-one jackknife covariance of a vector statistic."""
    steps = np.asarray(steps, dtype=float)
    if steps.ndim != 2 or steps.shape[0] < 2:
        raise ValueError("jackknife steps must have shape (N, d) with N >= 2")
    count = steps.shape[0]
    centered = steps - np.mean(steps, axis=0, keepdims=True)
    return ((count - 1.0) / count) * (centered.T @ centered)


def _diagnostics(nominal_step, jackknife_steps, ctx):
    covariance = _jackknife_covariance(jackknife_steps)
    sigma = float(np.sqrt(max(np.trace(covariance), 0.0)))
    step_norm = float(np.linalg.norm(nominal_step))
    scale = max(step_norm, ctx.config.step_norm_floor)
    relative = sigma / max(scale, np.finfo(float).eps)
    threshold = ctx.config.absolute_tolerance + ctx.config.relative_tolerance * scale
    return {
        "jackknife_step_covariance": covariance,
        "jackknife_step_sigma": sigma,
        "jackknife_step_relative_uncertainty": relative,
        "jackknife_step_threshold": float(threshold),
        "jackknife_step_accepted": bool(sigma <= threshold),
    }


def _attach_diagnostics(state, diagnostics, sample_size, stages, max_reached):
    state.update(diagnostics)
    state["adaptive_sample_size"] = int(sample_size)
    state["adaptive_enrichment_stages"] = int(stages)
    state["adaptive_max_sample_size_reached"] = bool(max_reached)
    print(
        "Adaptive sampling: "
        f"N={sample_size}, sigma_step={diagnostics['jackknife_step_sigma']:.5e}, "
        f"relative={diagnostics['jackknife_step_relative_uncertainty']:.5e}, "
        f"threshold={diagnostics['jackknife_step_threshold']:.5e}, "
        f"accepted={diagnostics['jackknife_step_accepted']}"
    )
    return state


def _append_vi_results(results, extra):
    qois = np.hstack([results["qois"], extra["qois"]])
    errors = np.hstack([results["errors"], extra["errors"]])
    return {"qois": qois, "mean-qoi": np.mean(qois, axis=1), "errors": errors}


def _subset_vi_results(results, keep):
    qois = results["qois"][:, keep]
    errors = results["errors"][:, keep]
    return {"qois": qois, "mean-qoi": np.mean(qois, axis=1), "errors": errors}


def _build_vi_state(a, optimizer_samples, parameter_samples, results):
    return _vi._build_vi_state_from_results(
        optimizer_samples=optimizer_samples,
        parameter_samples=parameter_samples,
        iteration_results=results,
        observations=a["observations"],
        observations_covariance=a["observations_covariance"],
        variational_mean=a["variational_mean"],
        variational_log_std=a["variational_log_std"],
        prior_mean=a["prior_mean"],
        prior_precision_operator=a["prior_precision_operator"],
        prior_covariance_log_det=a["prior_covariance_log_det"],
        covariance_regularization=a["covariance_regularization"],
        baseline_method=a["baseline_method"],
        gradient_method=a["gradient_method"],
        bounded_parameter_handling=a["bounded_parameter_handling"],
        min_variational_std=a["min_variational_std"],
        max_variational_std=a["max_variational_std"],
        parameter_mins=a["parameter_mins"],
        parameter_maxes=a["parameter_maxes"],
        transform_interior_margin=a.get("transform_interior_margin", 0.0),
        transform_map=a.get("transform_map", "sigmoid"),
        variational_correlation_cholesky=a.get("variational_correlation_cholesky"),
        elbo_scaling_factor=a.get("elbo_scaling_factor", 1.0),
        log_likelihood_precision_operator=a.get("log_likelihood_precision_operator"),
        score_function_entropy_strategy=a.get("score_function_entropy_strategy", "analytic"),
    )


def _vi_jackknife(a, state, optimizer_samples, parameter_samples, results, ctx):
    sample_count = optimizer_samples.shape[0]
    nominal_step = _newton_step(
        state,
        a["variational_log_std"],
        a["min_variational_std"],
        a["max_variational_std"],
        ctx,
    )
    steps = []
    for deleted in range(sample_count):
        keep = np.ones(sample_count, dtype=bool)
        keep[deleted] = False
        subset_state = _build_vi_state(
            a,
            optimizer_samples[keep],
            parameter_samples[keep],
            _subset_vi_results(results, keep),
        )
        steps.append(
            _newton_step(
                subset_state,
                a["variational_log_std"],
                a["min_variational_std"],
                a["max_variational_std"],
                ctx,
            )
        )
    return _diagnostics(nominal_step, np.asarray(steps), ctx)


def _adaptive_vi_state_impl(a, ctx, standard_normal_samples=None):
    sample_size = int(a["sample_size"])
    max_size = max(sample_size, int(ctx.config.max_sample_size))
    optimizer_samples, parameter_samples = _vi._draw_parameter_samples(
        a["variational_mean"],
        a["variational_log_std"],
        sample_size,
        a["min_variational_std"],
        a["max_variational_std"],
        a["bounded_parameter_handling"],
        a["parameter_mins"],
        a["parameter_maxes"],
        transform_interior_margin=a.get("transform_interior_margin", 0.0),
        transform_map=a.get("transform_map", "sigmoid"),
        standard_normal_samples=standard_normal_samples,
        variational_correlation_cholesky=a.get("variational_correlation_cholesky"),
        sampling_method=a.get("sampling_method", "mc"),
    )
    results = _vi._run_vi_iteration_samples(
        a["model"],
        a["observations"],
        a["run_directory_base"],
        a["parameter_names"],
        parameter_samples,
        a["evaluation_concurrency"],
        a.get("dispatcher"),
    )
    state = _build_vi_state(a, optimizer_samples, parameter_samples, results)
    stages = 0
    while True:
        diagnostics = _vi_jackknife(a, state, optimizer_samples, parameter_samples, results, ctx)
        if diagnostics["jackknife_step_accepted"] or sample_size >= max_size:
            return (
                _attach_diagnostics(
                    state,
                    diagnostics,
                    sample_size,
                    stages,
                    sample_size >= max_size and not diagnostics["jackknife_step_accepted"],
                ),
                optimizer_samples,
                parameter_samples,
                results,
            )
        target = min(
            max_size,
            max(sample_size + 1, int(np.ceil(sample_size * ctx.config.growth_factor))),
        )
        extra_count = target - sample_size
        extra_optimizer, extra_parameters = _vi._draw_parameter_samples(
            a["variational_mean"],
            a["variational_log_std"],
            extra_count,
            a["min_variational_std"],
            a["max_variational_std"],
            a["bounded_parameter_handling"],
            a["parameter_mins"],
            a["parameter_maxes"],
            transform_interior_margin=a.get("transform_interior_margin", 0.0),
            transform_map=a.get("transform_map", "sigmoid"),
            variational_correlation_cholesky=a.get("variational_correlation_cholesky"),
            sampling_method=a.get("sampling_method", "mc"),
        )
        extra_results = _vi._run_vi_iteration_samples(
            a["model"],
            a["observations"],
            f"{a['run_directory_base']}adaptive_{stages + 1}_",
            a["parameter_names"],
            extra_parameters,
            a["evaluation_concurrency"],
            a.get("dispatcher"),
        )
        optimizer_samples = np.vstack([optimizer_samples, extra_optimizer])
        parameter_samples = np.vstack([parameter_samples, extra_parameters])
        results = _append_vi_results(results, extra_results)
        sample_size = target
        stages += 1
        state = _build_vi_state(a, optimizer_samples, parameter_samples, results)


def _adaptive_vi_state(*args, **kwargs):
    ctx = _ACTIVE_CONTEXT.get()
    if ctx is None:
        return _ORIGINAL_EVALUATE_VI_STATE(*args, **kwargs)
    a = _bind(_ORIGINAL_EVALUATE_VI_STATE, args, kwargs)
    state, _, _, _ = _adaptive_vi_state_impl(a, ctx)
    return state


def _adaptive_vi_candidate(*args, **kwargs):
    ctx = _ACTIVE_CONTEXT.get()
    if ctx is None:
        return _ORIGINAL_EVALUATE_VI_CANDIDATE(*args, **kwargs)
    a = _bind(_ORIGINAL_EVALUATE_VI_CANDIDATE, args, kwargs)
    if a.get("line_search_objective", "elbo") != "elbo":
        raise NotImplementedError(
            "adaptive_sample_config currently requires line_search_objective='elbo'."
        )
    state_args = dict(a)
    state_args.pop("line_search_objective", None)
    standard = state_args.pop("standard_normal_samples", None)
    state_args["sampling_method"] = "mc"
    state, optimizer_samples, parameter_samples, results = _adaptive_vi_state_impl(
        state_args, ctx, standard_normal_samples=standard
    )
    return {
        "optimizer_samples": optimizer_samples,
        "parameter_samples": parameter_samples,
        "iteration_results": results,
        "mean_relative_mse": state["mean_relative_mse"],
        "state": state,
    }


def _mf_arrays(state):
    n_fom = state["parameter_samples_fom"].shape[0]
    n_base = state["parameter_samples_rom_base"].shape[0]
    n_extra = state["parameter_samples_rom_only"].shape[0]
    optimizer = state["optimizer_samples"]
    return (
        optimizer[:n_fom],
        optimizer[n_fom:n_fom + n_base],
        optimizer[n_fom + n_base:n_fom + n_base + n_extra],
    )


def _mf_estimator_state(
    a,
    optimizer_fom,
    optimizer_base,
    optimizer_extra,
    log_joint_fom,
    log_joint_base,
    log_joint_extra,
):
    variational_std, _ = _vi._compute_variational_std(
        a["variational_log_std"], a["min_variational_std"], a["max_variational_std"]
    )
    gradient = _mf._compute_mfmc_reinforce_gradients(
        optimizer_fom,
        optimizer_base,
        optimizer_extra,
        a["variational_mean"],
        variational_std,
        log_joint_fom,
        log_joint_base,
        log_joint_extra,
        a["baseline_method"],
        a["use_mfmc_control_variate"],
        a["mfmc_control_variate_mode"],
        a.get("variational_correlation_cholesky"),
        a.get("elbo_scaling_factor", 1.0),
        a.get("score_function_entropy_strategy", "analytic"),
    )
    hessian_full = _mf._compute_mfmc_reinforce_hessian_full(
        optimizer_fom,
        optimizer_base,
        optimizer_extra,
        a["variational_mean"],
        variational_std,
        log_joint_fom,
        log_joint_base,
        log_joint_extra,
        a["baseline_method"],
        a["use_mfmc_control_variate"],
        a["mfmc_control_variate_mode"],
        a.get("variational_correlation_cholesky"),
        a.get("elbo_scaling_factor", 1.0),
    )
    dimensionality = a["variational_mean"].size
    return {
        "gradient_mean": gradient[0],
        "gradient_log_std": gradient[1],
        "hessian_diagonal_mean": np.diag(hessian_full[:dimensionality, :dimensionality]),
        "hessian_diagonal_log_std": np.diag(hessian_full[dimensionality:, dimensionality:]),
        "hessian_full": hessian_full,
        "mfmc_alpha_mean": gradient[4],
        "mfmc_alpha_log_std": gradient[5],
        "gradient_signal_to_noise_ratio": gradient[6],
        "gradient_standard_error": gradient[7],
    }


def _mf_jackknife(a, state, ctx):
    optimizer_fom, optimizer_base, optimizer_extra = _mf_arrays(state)
    sample_count = optimizer_fom.shape[0]
    nominal_step = _newton_step(
        state,
        a["variational_log_std"],
        a["min_variational_std"],
        a["max_variational_std"],
        ctx,
    )
    steps = []
    for deleted in range(sample_count):
        keep = np.ones(sample_count, dtype=bool)
        keep[deleted] = False
        subset_state = _mf_estimator_state(
            a,
            optimizer_fom[keep],
            optimizer_base[keep],
            optimizer_extra,
            state["log_joint_terms_fom"][keep],
            state["log_joint_terms_rom_base"][keep],
            state["log_joint_terms_rom_only"],
        )
        steps.append(
            _newton_step(
                subset_state,
                a["variational_log_std"],
                a["min_variational_std"],
                a["max_variational_std"],
                ctx,
            )
        )
    return _diagnostics(nominal_step, np.asarray(steps), ctx)


def _recompute_mf_state(
    a,
    state,
    optimizer_fom,
    parameter_fom,
    fom_results,
    optimizer_base,
    parameter_base,
    base_results,
    optimizer_extra,
    parameter_extra,
    extra_results,
):
    precision = a.get("log_likelihood_precision_operator")
    fom_ll, fom_misfits = _vi._compute_log_likelihoods(
        fom_results["errors"],
        a["observations_covariance"],
        a["covariance_regularization"],
        precision,
    )
    base_ll, _ = _vi._compute_log_likelihoods(
        base_results["errors"],
        a["observations_covariance"],
        a["covariance_regularization"],
        precision,
    )
    if optimizer_extra.shape[0] > 0:
        extra_ll, _ = _vi._compute_log_likelihoods(
            extra_results["errors"],
            a["observations_covariance"],
            a["covariance_regularization"],
            precision,
        )
    else:
        extra_ll = np.zeros(0)

    def joint(ll, parameters, optimizer):
        if optimizer.shape[0] == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        return _vi._compute_log_prior_and_joint_terms(
            ll,
            parameters,
            optimizer,
            a["prior_mean"],
            a["prior_precision_operator"],
            a["prior_covariance_log_det"],
            a["bounded_parameter_handling"],
            a["parameter_mins"],
            a["parameter_maxes"],
            a.get("transform_interior_margin", 0.0),
            a.get("transform_map", "sigmoid"),
        )

    fom_prior, fom_jac, fom_joint = joint(fom_ll, parameter_fom, optimizer_fom)
    base_prior, base_jac, base_joint = joint(base_ll, parameter_base, optimizer_base)
    extra_prior, extra_jac, extra_joint = joint(extra_ll, parameter_extra, optimizer_extra)
    estimator = _mf_estimator_state(
        a,
        optimizer_fom,
        optimizer_base,
        optimizer_extra,
        fom_joint,
        base_joint,
        extra_joint,
    )

    variational_std, clipped_log_std = _vi._compute_variational_std(
        a["variational_log_std"], a["min_variational_std"], a["max_variational_std"]
    )
    update_mean, update_log, normalized_gradient_method = _vi._compute_update_directions(
        estimator["gradient_mean"],
        estimator["gradient_log_std"],
        variational_std,
        a["gradient_method"],
    )
    dimensionality = a["variational_mean"].size
    entropy = np.sum(clipped_log_std) + 0.5 * dimensionality * (1.0 + np.log(2.0 * np.pi))
    corr = a.get("variational_correlation_cholesky")
    if corr is not None:
        entropy += np.sum(np.log(np.diag(corr)))
    entropy *= a.get("elbo_scaling_factor", 1.0)
    if extra_joint.size:
        mf_mean_joint = (
            np.mean(fom_joint)
            + np.mean(np.concatenate([base_joint, extra_joint]))
            - np.mean(base_joint)
        )
    else:
        mf_mean_joint = np.mean(fom_joint)
    elbo = a.get("elbo_scaling_factor", 1.0) * mf_mean_joint + entropy

    state.update(
        {
            "optimizer_samples": np.vstack([optimizer_fom, optimizer_base, optimizer_extra]),
            "parameter_samples": np.vstack([parameter_fom, parameter_base, parameter_extra]),
            "parameter_samples_fom": parameter_fom,
            "parameter_samples_rom_base": parameter_base,
            "parameter_samples_rom_only": parameter_extra,
            "qois_fom": fom_results["qois"],
            "mean_qoi_fom": np.mean(fom_results["qois"], axis=1),
            "errors_fom": fom_results["errors"],
            "qois_rom_base": base_results["qois"],
            "qois_rom_coupled": base_results["qois"],
            "qois_rom_only": extra_results["qois"],
            "log_likelihoods_fom": fom_ll,
            "log_priors_fom": fom_prior,
            "log_joint_terms_fom": fom_joint,
            "log_transform_jacobian_fom": fom_jac,
            "log_likelihoods_rom_base": base_ll,
            "log_priors_rom_base": base_prior,
            "log_joint_terms_rom_base": base_joint,
            "log_transform_jacobian_rom_base": base_jac,
            "log_likelihoods_rom_coupled": base_ll,
            "log_likelihoods_rom_only": extra_ll,
            "log_priors_rom_only": extra_prior,
            "log_joint_terms_rom_only": extra_joint,
            "log_transform_jacobian_rom_only": extra_jac,
            "mean_misfit": float(np.mean(fom_misfits)),
            "mean_relative_mse": float(
                np.mean(_vi._compute_relative_mse(fom_results["errors"], a["observations"]))
            ),
            "entropy": entropy,
            "elbo": elbo,
            "gradient_mean": estimator["gradient_mean"],
            "gradient_log_std": estimator["gradient_log_std"],
            "hessian_diagonal_mean": estimator["hessian_diagonal_mean"],
            "hessian_diagonal_log_std": estimator["hessian_diagonal_log_std"],
            "hessian_full": estimator["hessian_full"],
            "update_direction_mean": update_mean,
            "update_direction_log_std": update_log,
            "gradient_method": normalized_gradient_method,
            "gradient_signal_to_noise_ratio": estimator["gradient_signal_to_noise_ratio"],
            "gradient_standard_error": estimator["gradient_standard_error"],
            "mfmc_alpha_mean": estimator["mfmc_alpha_mean"],
            "mfmc_alpha_log_std": estimator["mfmc_alpha_log_std"],
            "rom_error": _mf._compute_rom_relative_error(
                np.mean(base_results["qois"], axis=1)[:, None],
                np.mean(fom_results["qois"], axis=1)[:, None],
            ),
        }
    )
    return state


def _adaptive_mf_state(*args, **kwargs):
    ctx = _ACTIVE_CONTEXT.get()
    if ctx is None:
        return _ORIGINAL_EVALUATE_MF_VI_STATE(*args, **kwargs)
    a = _bind(_ORIGINAL_EVALUATE_MF_VI_STATE, args, kwargs)
    if a["rom_base_sampling_strategy"] != "coupled":
        raise NotImplementedError(
            "Adaptive MFVI currently requires rom_base_sampling_strategy='coupled'."
        )
    if a["correlation_estimator"] != "in_sample":
        raise NotImplementedError(
            "Adaptive MFVI currently requires correlation_estimator='in_sample'."
        )

    state = _ORIGINAL_EVALUATE_MF_VI_STATE(*args, **kwargs)
    sample_size = state["parameter_samples_fom"].shape[0]
    max_size = max(sample_size, int(ctx.config.max_sample_size))
    stages = 0
    dispatcher = resolve_dispatcher(a.get("dispatcher"))
    rom_dispatcher = resolve_local_dispatcher(dispatcher)

    while True:
        diagnostics = _mf_jackknife(a, state, ctx)
        if diagnostics["jackknife_step_accepted"] or sample_size >= max_size:
            return _attach_diagnostics(
                state,
                diagnostics,
                sample_size,
                stages,
                sample_size >= max_size and not diagnostics["jackknife_step_accepted"],
            )

        target = min(
            max_size,
            max(sample_size + 1, int(np.ceil(sample_size * ctx.config.growth_factor))),
        )
        extra_count = target - sample_size
        extra_optimizer, extra_parameters = _vi._draw_parameter_samples(
            a["variational_mean"],
            a["variational_log_std"],
            extra_count,
            a["min_variational_std"],
            a["max_variational_std"],
            a["bounded_parameter_handling"],
            a["parameter_mins"],
            a["parameter_maxes"],
            transform_interior_margin=a.get("transform_interior_margin", 0.0),
            transform_map=a.get("transform_map", "sigmoid"),
            variational_correlation_cholesky=a.get("variational_correlation_cholesky"),
            sampling_method=a.get("sampling_method", "mc"),
        )
        fom_extra = _mf.run_vi_iteration(
            a["model"],
            a["observations"],
            f"{a['iteration_directory']}/adaptive_fom_{stages + 1}_",
            a["parameter_names"],
            extra_parameters,
            a["fom_evaluation_concurrency"],
            dispatcher,
        )
        rom_extra_coupled = _mf.run_vi_iteration(
            state["rom_model"],
            a["observations"],
            f"{a['iteration_directory']}/adaptive_rom_{stages + 1}_",
            a["parameter_names"],
            extra_parameters,
            a["rom_evaluation_concurrency"],
            rom_dispatcher,
        )

        optimizer_fom, optimizer_base, optimizer_only = _mf_arrays(state)
        parameter_fom = state["parameter_samples_fom"]
        parameter_base = state["parameter_samples_rom_base"]
        parameter_only = state["parameter_samples_rom_only"]
        fom_results = {
            "qois": np.hstack([state["qois_fom"], fom_extra["qois"]]),
            "errors": np.hstack([state["errors_fom"], fom_extra["errors"]]),
        }
        fom_results["mean-qoi"] = np.mean(fom_results["qois"], axis=1)
        base_results = {
            "qois": np.hstack([state["qois_rom_base"], rom_extra_coupled["qois"]]),
            "errors": np.hstack(
                [
                    a["observations"][:, None] - state["qois_rom_base"],
                    rom_extra_coupled["errors"],
                ]
            ),
        }
        base_results["mean-qoi"] = np.mean(base_results["qois"], axis=1)
        only_results = {
            "qois": state["qois_rom_only"],
            "errors": a["observations"][:, None] - state["qois_rom_only"],
        }
        only_results["mean-qoi"] = (
            np.mean(only_results["qois"], axis=1)
            if only_results["qois"].shape[1]
            else fom_results["mean-qoi"]
        )

        optimizer_fom = np.vstack([optimizer_fom, extra_optimizer])
        optimizer_base = np.vstack([optimizer_base, extra_optimizer])
        parameter_fom = np.vstack([parameter_fom, extra_parameters])
        parameter_base = np.vstack([parameter_base, extra_parameters])

        new_dirs, new_parameters, new_qois = _mf._build_iteration_training_data(
            f"{a['iteration_directory']}/adaptive_fom_{stages + 1}_",
            extra_parameters,
            fom_extra,
        )
        state["training_dirs"] = list(state["training_dirs"]) + list(new_dirs)
        state["training_parameters"] = np.vstack([state["training_parameters"], new_parameters])
        state["training_qois"] = np.vstack([state["training_qois"], new_qois])

        state = _recompute_mf_state(
            a,
            state,
            optimizer_fom,
            parameter_fom,
            fom_results,
            optimizer_base,
            parameter_base,
            base_results,
            optimizer_only,
            parameter_only,
            only_results,
        )

        if state["rom_error"] >= a["rom_tolerance"]:
            state["rom_training_dirs"] = state["training_dirs"][-a["max_rom_training_dirs"]:]
            state["rom_training_parameters"] = state["training_parameters"][
                -a["max_rom_training_dirs"]:
            ]
            state["rom_training_qois"] = state["training_qois"][-a["max_rom_training_dirs"]:]
            state["rom_model"] = a["rom_model_builder"].build_from_training_dirs(
                a["iteration_directory"],
                state["rom_training_dirs"],
                state["rom_training_parameters"],
                state["rom_training_qois"],
            )
            base_results = _mf.run_vi_iteration(
                state["rom_model"],
                a["observations"],
                f"{a['iteration_directory']}/adaptive_rom_rebuild_{stages + 1}_",
                a["parameter_names"],
                parameter_base,
                a["rom_evaluation_concurrency"],
                rom_dispatcher,
            )
            if parameter_only.shape[0] > 0:
                only_results = _mf.run_vi_iteration(
                    state["rom_model"],
                    a["observations"],
                    f"{a['iteration_directory']}/adaptive_rom_only_rebuild_{stages + 1}_",
                    a["parameter_names"],
                    parameter_only,
                    a["rom_evaluation_concurrency"],
                    rom_dispatcher,
                )
            state = _recompute_mf_state(
                a,
                state,
                optimizer_fom,
                parameter_fom,
                fom_results,
                optimizer_base,
                parameter_base,
                base_results,
                optimizer_only,
                parameter_only,
                only_results,
            )
            state["rom_rebuilt_this_iteration"] = True

        sample_size = target
        stages += 1


@contextmanager
def _adaptive_context(ctx):
    if ctx is None:
        yield
        return
    token = _ACTIVE_CONTEXT.set(ctx)
    previous_vi_state = _vi._evaluate_vi_state
    previous_vi_candidate = _vi._evaluate_vi_candidate_for_line_search
    previous_mf_state = _mf._evaluate_mf_vi_state
    _vi._evaluate_vi_state = _adaptive_vi_state
    _vi._evaluate_vi_candidate_for_line_search = _adaptive_vi_candidate
    _mf._evaluate_mf_vi_state = _adaptive_mf_state
    try:
        yield
    finally:
        _vi._evaluate_vi_state = previous_vi_state
        _vi._evaluate_vi_candidate_for_line_search = previous_vi_candidate
        _mf._evaluate_mf_vi_state = previous_mf_state
        _ACTIVE_CONTEXT.reset(token)


def run_vi(*args, adaptive_sample_config: Optional[VIAdaptiveSampleConfig] = None, **kwargs):
    """Run standard VI with optional jackknife adaptive Newton sampling."""
    ctx = _resolve_newton_context(_ORIGINAL_RUN_VI, args, kwargs, adaptive_sample_config)
    if ctx is not None and ctx.config.disable_backtracking:
        args, kwargs = _single_candidate_line_search(_ORIGINAL_RUN_VI, args, kwargs)
    with _adaptive_context(ctx):
        return _BASE_RUN_VI(*args, **kwargs)


def run_mf_vi(*args, adaptive_sample_config: Optional[VIAdaptiveSampleConfig] = None, **kwargs):
    """Run standard MFVI with optional jackknife adaptive Newton sampling."""
    ctx = _resolve_newton_context(_ORIGINAL_RUN_MF_VI, args, kwargs, adaptive_sample_config)
    if ctx is not None and ctx.config.disable_backtracking:
        args, kwargs = _single_candidate_line_search(_ORIGINAL_RUN_MF_VI, args, kwargs)
    with _adaptive_context(ctx):
        return _BASE_RUN_MF_VI(*args, **kwargs)


def mf_vi_with_auto_rom(*args, adaptive_sample_config: Optional[VIAdaptiveSampleConfig] = None, **kwargs):
    """Run auto-ROM MFVI with optional jackknife adaptive Newton sampling."""
    ctx = _resolve_newton_context(
        _ORIGINAL_MF_VI_WITH_AUTO_ROM, args, kwargs, adaptive_sample_config
    )
    if ctx is not None and ctx.config.disable_backtracking:
        args, kwargs = _single_candidate_line_search(
            _ORIGINAL_MF_VI_WITH_AUTO_ROM, args, kwargs
        )
    with _adaptive_context(ctx):
        return _BASE_MF_VI_WITH_AUTO_ROM(*args, **kwargs)
