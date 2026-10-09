"""ROM-only Newton substeps for multifidelity variational inference.

Each substep estimates BOTH the ROM ELBO gradient and Hessian at its own
variational state. The surrogate is frozen for the duration of the substeps.
No high-fidelity model calls or multifidelity control variates occur here.
"""
from __future__ import annotations

import numpy as np


def validate_rom_substeps(start: int, end: int | None, count: int) -> None:
    if int(start) != start or start < 0:
        raise ValueError("rom_substep_start_iteration must be a nonnegative integer")
    if int(count) != count or count < 0:
        raise ValueError("num_rom_substeps must be a nonnegative integer")
    if end is not None and (int(end) != end or end < start):
        raise ValueError(
            "rom_substep_end_iteration must be an integer greater than or "
            "equal to rom_substep_start_iteration"
        )


def rom_substeps_enabled(outer_iteration: int, start: int,
                         end: int | None, count: int) -> bool:
    return count > 0 and start <= outer_iteration and (
        end is None or outer_iteration < end
    )


def restore_rom_substeps(restart_data, start, end, count):
    """Restore an existing run's schedule, matching MF-EKI restart semantics."""
    if "num_rom_substeps" in restart_data:
        start = int(restart_data["rom_substep_start_iteration"])
        saved_end = int(restart_data["rom_substep_end_iteration"])
        end = None if saved_end < 0 else saved_end
        count = int(restart_data["num_rom_substeps"])
    validate_rom_substeps(start, end, count)
    return start, end, count


def rom_substep_restart_data(start, end, count):
    return dict(
        rom_substep_start_iteration=int(start),
        rom_substep_end_iteration=-1 if end is None else int(end),
        num_rom_substeps=int(count),
    )


def _rom_newton_config(outer_method, optimizer_config, *, full_covariance=False):
    """Use the outer Newton curvature settings, otherwise safe Newton defaults.

    Inner steps are Newton steps even when the outer MFVI optimizer is
    gradient/Adam, so no outer Adam moments or lagged curvature are mutated.
    """
    from romtools.workflows.inverse.vi_optimization_methods import VINewtonOptimizerConfig
    if outer_method == "newton":
        return optimizer_config
    return VINewtonOptimizerConfig(
        newton_regularization=1e-2,
        newton_hessian_type="diagonal",
        newton_metric="natural" if full_covariance else "standard",
    )


def apply_diagonal_rom_substeps(
    *,
    rom_model,
    candidate_mean,
    candidate_log_std,
    outer_iteration,
    num_rom_substeps,
    step_size,
    outer_method,
    optimizer_config,
    max_mean_update_std,
    max_log_std_update,
    min_variational_std,
    max_variational_std,
    observations,
    observations_covariance,
    parameter_names,
    prior_mean,
    prior_precision_operator,
    prior_covariance_log_det,
    sample_size,
    evaluation_concurrency,
    covariance_regularization,
    baseline_method,
    bounded_parameter_handling,
    parameter_mins,
    parameter_maxes,
    transform_interior_margin,
    transform_map,
    min_physical_variational_std_fraction,
    variational_correlation_cholesky,
    elbo_scaling_factor,
    log_likelihood_precision_operator,
    sampling_method,
    score_function_entropy_strategy,
    directory,
    dispatcher,
):
    from romtools.workflows.inverse import vi_drivers as vi
    from romtools.workflows.inverse import vi_sample_reuse as reuse

    config = _rom_newton_config(outer_method, optimizer_config)
    mean = np.asarray(candidate_mean, dtype=float).copy()
    log_std = np.asarray(candidate_log_std, dtype=float).copy()
    for substep in range(num_rom_substeps):
        # Use the ORIGINAL single-fidelity evaluator: the ABRIS wrapper must
        # never consult/refresh the high-fidelity archive on a ROM-only step.
        rom_state = reuse._ORIGINAL_EVALUATE_VI_STATE(
            model=rom_model,
            observations=observations,
            observations_covariance=observations_covariance,
            run_directory_base=(
                f"{directory}/rom_substep_{substep}/run_rom_"
            ),
            parameter_names=parameter_names,
            variational_mean=mean,
            variational_log_std=log_std,
            prior_mean=prior_mean,
            prior_precision_operator=prior_precision_operator,
            prior_covariance_log_det=prior_covariance_log_det,
            sample_size=max(2, sample_size),
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
            variational_correlation_cholesky=variational_correlation_cholesky,
            elbo_scaling_factor=elbo_scaling_factor,
            log_likelihood_precision_operator=log_likelihood_precision_operator,
            sampling_method=sampling_method,
            dispatcher=dispatcher,
            score_function_entropy_strategy=score_function_entropy_strategy,
        )
        std, _ = vi._compute_variational_std(
            log_std, min_variational_std, max_variational_std
        )
        metric_scale = vi._compute_newton_metric_scale(
            config.newton_metric, std
        )
        # Gradient AND Hessian are recomputed from ROM samples at every
        # substep; use the same regularized solve as ordinary VI Newton.
        direction_mean, direction_log_std = vi._compute_newton_step(
            rom_state,
            config.newton_regularization,
            newton_hessian_type=config.newton_hessian_type,
            newton_regularization_strategy=config.newton_regularization_strategy,
            metric_scale=metric_scale,
        )
        mean += vi._limit_mean_update(
            direction_mean, step_size, std, max_mean_update_std
        )
        log_std += np.clip(
            step_size * direction_log_std,
            -max_log_std_update, max_log_std_update,
        )
        log_std = vi._clip_variational_log_std(
            log_std, min_variational_std, max_variational_std
        )
        log_std = vi._enforce_variational_log_std_bounds(
            mean, log_std, min_variational_std, max_variational_std,
            bounded_parameter_handling, parameter_mins, parameter_maxes,
            transform_interior_margin, min_physical_variational_std_fraction,
            transform_map,
        )
        print(
            f"  ROM-only VI Newton substep {substep + 1}/{num_rom_substeps} "
            f"after outer iteration {outer_iteration}, "
            f"gradient norm: {np.linalg.norm(np.r_[rom_state['gradient_mean'], rom_state['gradient_log_std']]):.5e}"
        )
    return mean, log_std


def apply_full_covariance_rom_substeps(
    *,
    rom_model,
    candidate_mean,
    candidate_cholesky,
    outer_iteration,
    num_rom_substeps,
    step_size,
    outer_method,
    optimizer_config,
    max_covariance_log_step,
    min_variational_std,
    max_variational_std,
    min_physical_variational_std_fraction,
    observations,
    observations_covariance,
    parameter_names,
    prior_mean,
    prior_precision_operator,
    prior_covariance_log_det,
    sample_size,
    evaluation_concurrency,
    covariance_regularization,
    baseline_method,
    bounded_parameter_handling,
    parameter_mins,
    parameter_maxes,
    transform_interior_margin,
    transform_map,
    elbo_scaling_factor,
    log_likelihood_precision_operator,
    sampling_method,
    score_function_entropy_strategy,
    directory,
    dispatcher,
):
    from romtools.workflows.inverse import full_covariance_vi_drivers as fcvi
    from romtools.workflows.inverse import full_covariance_mf_vi_drivers as fcmf
    from romtools.workflows.inverse import full_covariance_newton as fcnewton
    from romtools.workflows.inverse.full_covariance_vi import (
        packed_direction_to_covariance, retract_covariance,
    )

    config = _rom_newton_config(
        outer_method, optimizer_config, full_covariance=True
    )
    mean = np.asarray(candidate_mean, dtype=float).copy()
    cholesky = np.asarray(candidate_cholesky, dtype=float).copy()
    for substep in range(num_rom_substeps):
        rom_state = fcvi._evaluate_single_fidelity_state(
            model=rom_model,
            observations=observations,
            observations_covariance=observations_covariance,
            run_directory_base=f"{directory}/rom_substep_{substep}/run_rom_",
            parameter_names=parameter_names,
            variational_mean=mean,
            cholesky=cholesky,
            prior_mean=prior_mean,
            prior_precision_operator=prior_precision_operator,
            prior_covariance_log_det=prior_covariance_log_det,
            sample_size=max(2, sample_size),
            evaluation_concurrency=evaluation_concurrency,
            covariance_regularization=covariance_regularization,
            baseline_method=baseline_method,
            gradient_method="natural",
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
        hessian = fcnewton.estimate_ordinary_hessian(
            rom_state["optimizer_samples"],
            mean, cholesky,
            elbo_scaling_factor * np.asarray(rom_state["log_joint_terms"]),
            baseline_method,
            elbo_scaling_factor,
        )
        direction = fcnewton._newton_step_from_hessian(
            rom_state, cholesky, config, hessian
        )
        dimension = mean.size
        mean += step_size * direction[:dimension]
        cholesky, _ = retract_covariance(
            cholesky,
            packed_direction_to_covariance(direction[dimension:], dimension),
            step_size, max_covariance_log_step,
        )
        cholesky = fcmf._enforce_full_covariance_scale_bounds(
            mean, cholesky, min_variational_std, max_variational_std,
            bounded_parameter_handling, parameter_mins, parameter_maxes,
            transform_interior_margin, min_physical_variational_std_fraction,
            transform_map,
        )
        print(
            f"  ROM-only full-covariance VI Newton substep "
            f"{substep + 1}/{num_rom_substeps} after outer iteration "
            f"{outer_iteration}, gradient norm: "
            f"{np.linalg.norm(np.r_[rom_state['gradient_mean'], rom_state['gradient_covariance_svec']]):.5e}"
        )
    return mean, cholesky
