"""ROM-only optimizer substeps for multifidelity variational inference.

Substeps inherit the outer optimizer method and configuration. Adam substeps
reuse a frozen snapshot of the outer optimizer's moments and iteration counter,
without changing them. Every ROM gradient produces a hypothetical Adam step
from that same snapshot. Newton substeps recompute both the gradient and
Hessian at each variational state. No high-fidelity model calls or
multifidelity control variates occur in these steps.
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


class _FrozenAdamOptimizer:
    """Produce independent hypothetical Adam updates from one fixed state.

    The supplied outer state is copied once; before *each* ROM-only gradient,
    the shadow optimizer is restored to that same snapshot. Thus the ROM
    gradients influence the parameter update but cannot change the moment
    history or Adam iteration count used by subsequent ROM substeps.
    """

    def __init__(self, config, outer_solver, parameter_dimension=None):
        from romtools.workflows.inverse.vi_optimization_methods import AdamSolver

        if not isinstance(outer_solver, AdamSolver):
            raise TypeError("Adam ROM substeps require the outer AdamSolver")
        if (parameter_dimension is not None
                and outer_solver.parameter_dimension not in (None, parameter_dimension)):
            raise ValueError("Outer Adam parameter dimension does not match ROM")
        self._snapshot = outer_solver.restart_state_dict()
        self._solver = AdamSolver.from_config(config)
        self._solver.parameter_dimension = outer_solver.parameter_dimension
        # Validate configuration and snapshot as soon as the ROM sequence starts.
        self._solver.load_restart_state_dict(self._snapshot)

    def step(self, gradient, fisher_diagonal=None):
        self._solver.load_restart_state_dict(self._snapshot)
        return self._solver.step(gradient, fisher_diagonal=fisher_diagonal)


def _rom_optimizer(outer_method, optimizer_config, *,
                   parameter_dimension=None, outer_adam_solver=None):
    """Use the configured method while keeping outer optimizer state intact."""
    from romtools.workflows.inverse.vi_optimization_methods import (
        SteepestDescentSolver,
    )

    method = outer_method.strip().lower()
    if method not in ("gradient", "adam", "newton"):
        raise ValueError(f"Unsupported ROM-only optimizer method: {outer_method}")
    if method == "adam":
        solver = _FrozenAdamOptimizer(
            optimizer_config, outer_adam_solver,
            parameter_dimension=parameter_dimension,
        )
    elif method == "gradient":
        solver = SteepestDescentSolver()
    else:
        solver = None
    return method, solver


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
    outer_adam_solver=None,
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

    method, solver = _rom_optimizer(
        outer_method, optimizer_config, outer_adam_solver=outer_adam_solver,
    )
    gradient_method = (
        optimizer_config.gradient_method.strip().lower()
        if method in ("gradient", "adam") else "standard"
    )
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
            sampling_method=sampling_method,
            dispatcher=dispatcher,
            score_function_entropy_strategy=score_function_entropy_strategy,
        )
        std, _ = vi._compute_variational_std(
            log_std, min_variational_std, max_variational_std
        )
        if method == "newton":
            metric_scale = vi._compute_newton_metric_scale(
                optimizer_config.newton_metric, std
            )
            # Newton uses both fresh ROM gradient and Hessian every substep.
            direction_mean, direction_log_std = vi._compute_newton_step(
                rom_state,
                optimizer_config.newton_regularization,
                newton_hessian_type=optimizer_config.newton_hessian_type,
                newton_regularization_strategy=optimizer_config.newton_regularization_strategy,
                newton_additive_regularization=optimizer_config.newton_additive_regularization,
                newton_regularization_epsilon=optimizer_config.newton_regularization_epsilon,
                newton_fallback_learning_rate=optimizer_config.newton_fallback_learning_rate,
                metric_scale=metric_scale,
            )
        else:
            if method == "adam":
                # Match outer mean-field Adam's Fisher preconditioning.
                # Each gradient uses a hypothetical step from frozen moments.
                gradient = np.r_[
                    rom_state["gradient_mean"], rom_state["gradient_log_std"]
                ]
                fisher = vi._compute_adam_fisher_diagonal(
                    log_std, min_variational_std, max_variational_std,
                    gradient_method,
                )
                direction = solver.step(gradient, fisher_diagonal=fisher)
            else:
                direction = solver.step(np.r_[
                    rom_state["update_direction_mean"],
                    rom_state["update_direction_log_std"],
                ])
            dimension = mean.size
            direction_mean = direction[:dimension]
            direction_log_std = direction[dimension:]
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
            f"  ROM-only VI {method} substep {substep + 1}/{num_rom_substeps} "
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
    outer_adam_solver=None,
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

    mean = np.asarray(candidate_mean, dtype=float).copy()
    method, solver = _rom_optimizer(
        outer_method, optimizer_config, parameter_dimension=mean.size,
        outer_adam_solver=outer_adam_solver,
    )
    gradient_method = (
        optimizer_config.gradient_method.strip().lower()
        if method in ("gradient", "adam") else "natural"
    )
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
        if method == "newton":
            hessian = fcnewton.estimate_ordinary_hessian(
                rom_state["optimizer_samples"],
                mean, cholesky,
                elbo_scaling_factor * np.asarray(rom_state["log_joint_terms"]),
                baseline_method,
                elbo_scaling_factor,
            )
            direction = fcnewton._newton_step_from_hessian(
                rom_state, cholesky, optimizer_config, hessian
            )
        else:
            gradient = fcvi._update_vector(rom_state)
            direction = (
                solver.step(gradient, fisher_diagonal=None)
                if method == "adam" else solver.step(gradient)
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
            f"  ROM-only full-covariance VI {method} substep "
            f"{substep + 1}/{num_rom_substeps} after outer iteration "
            f"{outer_iteration}, gradient norm: "
            f"{np.linalg.norm(np.r_[rom_state['gradient_mean'], rom_state['gradient_covariance_svec']]):.5e}"
        )
    return mean, cholesky
