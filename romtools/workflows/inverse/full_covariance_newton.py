"""Natural-coordinate Newton support for full-covariance Gaussian VI/MF-VI.

This module augments the full-covariance score-function implementation with a
second-order curvature model in ``(mu, svec(Sigma))`` coordinates. Newton
steps can be formed in ordinary coordinates or in locally Fisher-whitened
coordinates; the latter is selected with ``newton_metric='natural'``.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy

import numpy as np

from romtools.workflows.inverse import mf_vi_drivers as _mf
from romtools.workflows.inverse import full_covariance_vi_drivers as _fc_vi
from romtools.workflows.inverse import full_covariance_mf_vi_drivers as _fc_mf
from romtools.workflows.inverse import full_covariance_auto_mf_vi as _fc_auto
from romtools.workflows.inverse.full_covariance_vi import (
    covariance_from_cholesky,
    svec,
    svec_inverse,
    svec_size,
)
from romtools.workflows.inverse.vi_optimization_methods import (
    NewtonSolver,
    VIGradientOptimizerConfig,
    VINewtonOptimizerConfig,
    _normalize_newton_curvature_strategy,
    _normalize_newton_hessian_type,
    _normalize_newton_metric,
)


def _symmetric_basis(dimensionality: int) -> list[np.ndarray]:
    """Return the Frobenius-orthonormal symmetric basis induced by ``svec``."""
    basis = []
    size = svec_size(dimensionality)
    for index in range(size):
        coordinate = np.zeros(size, dtype=float)
        coordinate[index] = 1.0
        basis.append(svec_inverse(coordinate, dimensionality))
    return basis


def _ordinary_score_and_logq_hessian_terms(
    optimizer_samples: np.ndarray,
    mean: np.ndarray,
    cholesky: np.ndarray,
):
    """Return Gaussian scores and log-density Hessians in ``(mu, svec(Sigma))``."""
    optimizer_samples = np.asarray(optimizer_samples, dtype=float)
    mean = np.asarray(mean, dtype=float)
    covariance = covariance_from_cholesky(cholesky)
    precision = np.linalg.solve(covariance, np.eye(mean.size))
    dimensionality = mean.size
    covariance_size = svec_size(dimensionality)
    total_size = dimensionality + covariance_size
    basis = _symmetric_basis(dimensionality)

    scores = np.empty((optimizer_samples.shape[0], total_size), dtype=float)
    hessians = np.empty(
        (optimizer_samples.shape[0], total_size, total_size), dtype=float
    )

    for sample_index, sample in enumerate(optimizer_samples):
        delta = sample - mean
        score_mean = precision @ delta
        score_covariance_matrix = 0.5 * (
            np.outer(score_mean, score_mean) - precision
        )
        scores[sample_index] = np.concatenate(
            [score_mean, svec(score_covariance_matrix)]
        )

        hessian = np.zeros((total_size, total_size), dtype=float)
        hessian[:dimensionality, :dimensionality] = -precision
        for basis_index, basis_matrix in enumerate(basis):
            column = dimensionality + basis_index
            cross = -precision @ basis_matrix @ score_mean
            hessian[:dimensionality, column] = cross
            hessian[column, :dimensionality] = cross

            derivative_covariance_score = 0.5 * (
                -precision @ basis_matrix @ np.outer(score_mean, score_mean)
                - np.outer(score_mean, score_mean) @ basis_matrix @ precision
                + precision @ basis_matrix @ precision
            )
            hessian[dimensionality:, column] = svec(
                0.5
                * (
                    derivative_covariance_score
                    + derivative_covariance_score.T
                )
            )
        hessians[sample_index] = 0.5 * (hessian + hessian.T)

    return scores, hessians


def _entropy_hessian(cholesky: np.ndarray, scaling_factor: float) -> np.ndarray:
    """Return the analytic Gaussian-entropy Hessian in ordinary coordinates."""
    covariance = covariance_from_cholesky(cholesky)
    precision = np.linalg.solve(covariance, np.eye(covariance.shape[0]))
    dimensionality = covariance.shape[0]
    covariance_size = svec_size(dimensionality)
    result = np.zeros(
        (dimensionality + covariance_size, dimensionality + covariance_size),
        dtype=float,
    )
    for basis_index, basis_matrix in enumerate(_symmetric_basis(dimensionality)):
        derivative = -0.5 * scaling_factor * precision @ basis_matrix @ precision
        result[dimensionality:, dimensionality + basis_index] = svec(
            0.5 * (derivative + derivative.T)
        )
    return 0.5 * (result + result.T)


def _center_hessian_weights(
    values: np.ndarray,
    curvature_terms: np.ndarray,
    baseline_method: str,
) -> np.ndarray:
    """Center scalar curvature weights with an unbiased LOO baseline when asked."""
    values = np.asarray(values, dtype=float)
    method = str(baseline_method).strip().lower()
    if method == "none":
        return values
    if method == "loo":
        if values.size <= 1:
            return values
        loo = (np.sum(values) - values) / float(values.size - 1)
        return values - loo
    if method == "optimal":
        squared_norm = np.sum(curvature_terms**2, axis=(1, 2))
        denominator = float(np.sum(squared_norm))
        baseline = 0.0 if denominator <= 0.0 else float(
            np.sum(values * squared_norm) / denominator
        )
        return values - baseline
    raise ValueError("baseline_method must be 'none', 'loo', or 'optimal'")


def estimate_ordinary_hessian(
    optimizer_samples: np.ndarray,
    mean: np.ndarray,
    cholesky: np.ndarray,
    scaled_log_joint_terms: np.ndarray,
    baseline_method: str,
    elbo_scaling_factor: float,
) -> np.ndarray:
    """Estimate full-covariance ELBO curvature using log-joint score terms.

    The optional joint entropy strategy affects the gradient estimator. Newton
    curvature follows the existing romtools split estimator: log-joint score
    curvature plus analytic Gaussian-entropy curvature.
    """
    scores, logq_hessians = _ordinary_score_and_logq_hessian_terms(
        optimizer_samples, mean, cholesky
    )
    curvature_kernel = (
        scores[:, :, None] * scores[:, None, :] + logq_hessians
    )
    centered = _center_hessian_weights(
        np.asarray(scaled_log_joint_terms, dtype=float),
        curvature_kernel,
        baseline_method,
    )
    hessian = np.mean(centered[:, None, None] * curvature_kernel, axis=0)
    hessian += _entropy_hessian(cholesky, elbo_scaling_factor)
    hessian = 0.5 * (hessian + hessian.T)
    return np.nan_to_num(hessian, nan=0.0, posinf=0.0, neginf=0.0)


def _fisher_whitening_map(cholesky: np.ndarray) -> np.ndarray:
    """Return ``S`` such that ``S.T @ F @ S = I`` for the Gaussian Fisher metric."""
    cholesky = np.asarray(cholesky, dtype=float)
    dimensionality = cholesky.shape[0]
    covariance_size = svec_size(dimensionality)
    total_size = dimensionality + covariance_size
    result = np.zeros((total_size, total_size), dtype=float)
    result[:dimensionality, :dimensionality] = cholesky
    for basis_index, basis_matrix in enumerate(_symmetric_basis(dimensionality)):
        tangent = np.sqrt(2.0) * cholesky @ basis_matrix @ cholesky.T
        result[
            dimensionality:, dimensionality + basis_index
        ] = svec(0.5 * (tangent + tangent.T))
    return result


def _newton_step_from_hessian(
    state: dict,
    cholesky: np.ndarray,
    config: VINewtonOptimizerConfig,
    hessian: np.ndarray,
) -> np.ndarray:
    gradient = np.concatenate(
        [state["gradient_mean"], state["gradient_covariance_svec"]]
    )
    metric = _normalize_newton_metric(config.newton_metric)
    transform = (
        _fisher_whitening_map(cholesky)
        if metric == "natural"
        else np.eye(gradient.size)
    )
    transformed_gradient = transform.T @ gradient
    transformed_hessian = transform.T @ hessian @ transform
    hessian_type = _normalize_newton_hessian_type(config.newton_hessian_type)
    solver = NewtonSolver(
        regularization=config.newton_regularization,
        hessian_type=hessian_type,
    )
    if hessian_type == "diagonal":
        transformed_hessian = np.diag(transformed_hessian)
    step_local = solver.step(transformed_gradient, transformed_hessian)
    return transform @ step_local


def _validate_newton_config(optimizer_config) -> VINewtonOptimizerConfig:
    if optimizer_config is None:
        config = VINewtonOptimizerConfig()
    elif not isinstance(optimizer_config, VINewtonOptimizerConfig):
        raise TypeError(
            "optimizer_config for optimizer_method='newton' must be "
            "VINewtonOptimizerConfig"
        )
    else:
        config = copy.deepcopy(optimizer_config)
    strategy = _normalize_newton_curvature_strategy(config.newton_curvature_strategy)
    if strategy != "same_sample":
        raise NotImplementedError(
            "Full-covariance Newton currently supports "
            "newton_curvature_strategy='same_sample'."
        )
    return config


def _gradient_config_from_newton(config: VINewtonOptimizerConfig):
    return VIGradientOptimizerConfig(
        gradient_method="natural",
        gradient_norm_tolerance=config.gradient_norm_tolerance,
        max_iterations=config.max_iterations,
        max_log_std_update=config.max_log_std_update,
        min_variational_std=config.min_variational_std,
        max_variational_std=config.max_variational_std,
    )


def _natural_gradient_norm(state: dict) -> float:
    return float(
        np.linalg.norm(
            np.concatenate(
                [
                    state["natural_gradient_mean"],
                    state["natural_gradient_covariance_svec"],
                ]
            )
        )
    )


@contextmanager
def _patched_single_fidelity_newton(config: VINewtonOptimizerConfig):
    original_upgrade = _fc_vi._upgrade_single_fidelity_state
    original_gradient_norm = _fc_vi._gradient_norm

    def upgrade(*args, **kwargs):
        state = original_upgrade(*args, **kwargs)
        mean = args[1] if len(args) > 1 else kwargs["variational_mean"]
        cholesky = args[2] if len(args) > 2 else kwargs["cholesky"]
        baseline_method = args[3] if len(args) > 3 else kwargs["baseline_method"]
        elbo_scaling_factor = (
            args[5] if len(args) > 5 else kwargs["elbo_scaling_factor"]
        )
        hessian = estimate_ordinary_hessian(
            state["optimizer_samples"],
            mean,
            cholesky,
            elbo_scaling_factor * np.asarray(state["log_joint_terms"]),
            baseline_method,
            elbo_scaling_factor,
        )
        step = _newton_step_from_hessian(state, cholesky, config, hessian)
        dimensionality = np.asarray(mean).size
        state["update_direction_mean"] = step[:dimensionality]
        state["update_direction_covariance_svec"] = step[dimensionality:]
        state["newton_hessian_full"] = hessian
        state["newton_metric"] = _normalize_newton_metric(config.newton_metric)
        return state

    _fc_vi._upgrade_single_fidelity_state = upgrade
    _fc_vi._gradient_norm = _natural_gradient_norm
    try:
        yield
    finally:
        _fc_vi._upgrade_single_fidelity_state = original_upgrade
        _fc_vi._gradient_norm = original_gradient_norm


def _mf_curvature_terms(
    optimizer_samples: np.ndarray,
    mean: np.ndarray,
    cholesky: np.ndarray,
    scaled_log_joint_terms: np.ndarray,
    baseline_method: str,
):
    scores, logq_hessians = _ordinary_score_and_logq_hessian_terms(
        optimizer_samples, mean, cholesky
    )
    kernel = scores[:, :, None] * scores[:, None, :] + logq_hessians
    centered = _center_hessian_weights(
        np.asarray(scaled_log_joint_terms, dtype=float), kernel, baseline_method
    )
    return centered[:, None, None] * kernel


def estimate_mf_ordinary_hessian(
    state: dict,
    mean: np.ndarray,
    cholesky: np.ndarray,
    baseline_method: str,
    use_mfmc_control_variate: bool,
    mfmc_control_variate_mode: str,
    elbo_scaling_factor: float,
) -> np.ndarray:
    n_fom = state["parameter_samples_fom"].shape[0]
    n_base = state["parameter_samples_rom_base"].shape[0]
    n_extra = state["parameter_samples_rom_only"].shape[0]
    optimizer_samples = np.asarray(state["optimizer_samples"])
    optimizer_fom = optimizer_samples[:n_fom]
    optimizer_base = optimizer_samples[n_fom:n_fom + n_base]
    optimizer_extra = optimizer_samples[n_fom + n_base:n_fom + n_base + n_extra]

    high = _mf_curvature_terms(
        optimizer_fom,
        mean,
        cholesky,
        elbo_scaling_factor * np.asarray(state["log_joint_terms_fom"]),
        baseline_method,
    )
    low_base = _mf_curvature_terms(
        optimizer_base,
        mean,
        cholesky,
        elbo_scaling_factor * np.asarray(state["log_joint_terms_rom_base"]),
        baseline_method,
    )
    low_extra = None
    if n_extra > 0:
        low_extra = _mf_curvature_terms(
            optimizer_extra,
            mean,
            cholesky,
            elbo_scaling_factor * np.asarray(state["log_joint_terms_rom_only"]),
            baseline_method,
        )
    estimate, _ = _mf._mfmc_gradient_estimator(
        high,
        low_base,
        low_extra,
        use_mfmc_control_variate,
        mfmc_control_variate_mode,
    )
    estimate = np.asarray(estimate, dtype=float)
    estimate += _entropy_hessian(cholesky, elbo_scaling_factor)
    return 0.5 * (estimate + estimate.T)


@contextmanager
def _patched_mf_newton(config: VINewtonOptimizerConfig):
    original_upgrade = _fc_mf._upgrade_mf_state
    original_gradient_norm = _fc_mf._gradient_norm

    def upgrade(*args, **kwargs):
        state = original_upgrade(*args, **kwargs)
        mean = args[1] if len(args) > 1 else kwargs["variational_mean"]
        cholesky = args[2] if len(args) > 2 else kwargs["cholesky"]
        baseline_method = args[3] if len(args) > 3 else kwargs["baseline_method"]
        use_control = (
            args[4] if len(args) > 4 else kwargs["use_mfmc_control_variate"]
        )
        control_mode = (
            args[5] if len(args) > 5 else kwargs["mfmc_control_variate_mode"]
        )
        elbo_scaling_factor = (
            args[7] if len(args) > 7 else kwargs["elbo_scaling_factor"]
        )
        hessian = estimate_mf_ordinary_hessian(
            state,
            mean,
            cholesky,
            baseline_method,
            use_control,
            control_mode,
            elbo_scaling_factor,
        )
        step = _newton_step_from_hessian(state, cholesky, config, hessian)
        dimensionality = np.asarray(mean).size
        state["update_direction_mean"] = step[:dimensionality]
        state["update_direction_covariance_svec"] = step[dimensionality:]
        state["newton_hessian_full"] = hessian
        state["newton_metric"] = _normalize_newton_metric(config.newton_metric)
        return state

    _fc_mf._upgrade_mf_state = upgrade
    _fc_mf._gradient_norm = _natural_gradient_norm
    try:
        yield
    finally:
        _fc_mf._upgrade_mf_state = original_upgrade
        _fc_mf._gradient_norm = original_gradient_norm


def run_vi(*args, **kwargs):
    """Run full-covariance VI with score-function Newton curvature."""
    config = _validate_newton_config(kwargs.get("optimizer_config"))
    call_kwargs = dict(kwargs)
    call_kwargs["optimizer_method"] = "gradient"
    call_kwargs["optimizer_config"] = _gradient_config_from_newton(config)
    with _patched_single_fidelity_newton(config):
        return _fc_vi.run_vi(*args, **call_kwargs)


def run_mf_vi(*args, **kwargs):
    """Run full-covariance MF-VI with multifidelity Newton curvature."""
    config = _validate_newton_config(kwargs.get("optimizer_config"))
    call_kwargs = dict(kwargs)
    call_kwargs["optimizer_method"] = "gradient"
    call_kwargs["optimizer_config"] = _gradient_config_from_newton(config)
    with _patched_mf_newton(config):
        return _fc_mf.run_mf_vi(*args, **call_kwargs)


def mf_vi_with_auto_rom(*args, **kwargs):
    """Run auto-GP full-covariance MF-VI with natural-coordinate Newton."""
    original_run_mf_vi = _fc_auto.run_mf_vi
    _fc_auto.run_mf_vi = run_mf_vi
    try:
        return _fc_auto.mf_vi_with_auto_rom(*args, **kwargs)
    finally:
        _fc_auto.run_mf_vi = original_run_mf_vi
