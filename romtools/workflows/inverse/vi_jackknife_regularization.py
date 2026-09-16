"""Jackknife-driven adaptive regularization for independent-Hessian Newton VI.

This module provides an experimental wrapper around the existing VI and MFVI
Newton drivers. The gradient and Hessian estimators remain statistically
independent via ``newton_curvature_strategy='independent'``. The independent
Hessian batch is delete-one jackknifed, each replicate is mapped into the same
natural/standard metric coordinates used by the Newton solve, and the
regularization is expressed through the relative spectral uncertainty

    rho_H = sigma_H / ||H||_2.

The applied dimensional regularization is

    lambda = clip(scale * rho_H * ||H||_2, minimum, maximum),

where

    sigma_H^2 = (N-1)/N * sum_i ||H_{(-i)} - mean(H_{(-j)})||_2^2.

The Hessians in these expressions are sign-projected (absolute eigenvalues) but
*not* regularized, so the statistical uncertainty is measured before damping.
No extra model evaluations are required beyond the independent Hessian batch.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from romtools.workflows.inverse import mf_vi_drivers as _mf
from romtools.workflows.inverse import vi_adaptive_sampling as _adaptive
from romtools.workflows.inverse import vi_drivers as _vi
from romtools.workflows.inverse.vi_optimization_methods import (
    VINewtonOptimizerConfig,
    _resolve_optimizer_config,
)


@dataclass(frozen=True)
class VIJackknifeRegularizationConfig:
    """Configuration for independent-Hessian jackknife regularization."""

    enabled: bool = True
    scale: float = 1.0
    minimum: float = 1.0e-8
    maximum: float = 1.0e-1

    def __post_init__(self):
        if not np.isfinite(self.scale) or self.scale < 0.0:
            raise ValueError("scale must be finite and non-negative")
        if not np.isfinite(self.minimum) or self.minimum < 0.0:
            raise ValueError("minimum must be finite and non-negative")
        if not np.isfinite(self.maximum) or self.maximum <= 0.0:
            raise ValueError("maximum must be finite and positive")
        if self.maximum < self.minimum:
            raise ValueError("maximum must be greater than or equal to minimum")


@dataclass
class _RegularizationContext:
    config: VIJackknifeRegularizationConfig
    newton_config: VINewtonOptimizerConfig
    current_regularization: Optional[float] = None
    current_sigma: Optional[float] = None
    current_hessian_magnitude: Optional[float] = None
    current_relative_uncertainty: Optional[float] = None
    history: list[float] = field(default_factory=list)
    sigma_history: list[float] = field(default_factory=list)
    hessian_magnitude_history: list[float] = field(default_factory=list)
    relative_uncertainty_history: list[float] = field(default_factory=list)


_ACTIVE_CONTEXT: ContextVar[Optional[_RegularizationContext]] = ContextVar(
    "romtools_vi_jackknife_regularization_context", default=None
)
_LAST_REGULARIZATION_HISTORY: list[float] = []
_LAST_SIGMA_HISTORY: list[float] = []
_LAST_HESSIAN_MAGNITUDE_HISTORY: list[float] = []
_LAST_RELATIVE_UNCERTAINTY_HISTORY: list[float] = []


def get_last_jackknife_regularization_history() -> np.ndarray:
    """Return regularization values selected by the most recent wrapper run."""
    return np.asarray(_LAST_REGULARIZATION_HISTORY, dtype=float).copy()


def get_last_jackknife_hessian_sigma_history() -> np.ndarray:
    """Return absolute Hessian uncertainty from the most recent wrapper run."""
    return np.asarray(_LAST_SIGMA_HISTORY, dtype=float).copy()


def get_last_jackknife_hessian_magnitude_history() -> np.ndarray:
    """Return projected Hessian spectral norms from the most recent wrapper run."""
    return np.asarray(_LAST_HESSIAN_MAGNITUDE_HISTORY, dtype=float).copy()


def get_last_jackknife_relative_uncertainty_history() -> np.ndarray:
    """Return sigma_H / ||H||_2 from the most recent wrapper run."""
    return np.asarray(_LAST_RELATIVE_UNCERTAINTY_HISTORY, dtype=float).copy()


def _resolve_context(signature_source, args, kwargs, config):
    if config is None or not config.enabled:
        return None
    arguments = _adaptive._bind(signature_source, args, kwargs)
    optimizer_method = arguments.get("optimizer_method", "gradient")
    optimizer_config = arguments.get("optimizer_config", None)
    method, resolved = _resolve_optimizer_config(
        optimizer_method,
        optimizer_config,
        default_newton_config=VINewtonOptimizerConfig(),
    )
    if method != "newton":
        raise ValueError(
            "jackknife_regularization_config requires optimizer_method='newton'"
        )
    if resolved.newton_curvature_strategy != "independent":
        raise ValueError(
            "jackknife_regularization_config requires "
            "newton_curvature_strategy='independent'"
        )
    if resolved.newton_hessian_type != "full":
        raise NotImplementedError(
            "Jackknife adaptive regularization currently supports "
            "newton_hessian_type='full' only."
        )
    if (
        resolved.newton_hessian_num_samples is not None
        and resolved.newton_hessian_num_samples < 3
    ):
        raise ValueError("newton_hessian_num_samples must be at least 3")
    return _RegularizationContext(config=config, newton_config=resolved)


def _project_unregularized_hessian(hessian: np.ndarray) -> np.ndarray:
    """Apply the Newton absolute-eigenvalue projection without a damping floor."""
    hessian = np.nan_to_num(
        np.asarray(hessian, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    sym_hessian = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym_hessian)
    return (eigenvectors @ np.diag(np.abs(eigenvalues))) @ eigenvectors.T


def _metric_projected_hessian(hessian: np.ndarray, arguments: dict, ctx) -> np.ndarray:
    variational_std, _ = _vi._compute_variational_std(
        arguments["variational_log_std"],
        arguments["min_variational_std"],
        arguments["max_variational_std"],
    )
    metric_scale = _vi._compute_newton_metric_scale(
        ctx.newton_config.newton_metric, variational_std
    )
    transformed = np.asarray(hessian, dtype=float)
    if metric_scale is not None:
        transformed = (
            metric_scale[:, None] * transformed * metric_scale[None, :]
        )
    return _project_unregularized_hessian(transformed)


def _jackknife_spectral_sigma(hessians: list[np.ndarray]) -> float:
    if len(hessians) < 2:
        raise ValueError("At least two delete-one Hessians are required")
    stack = np.stack(hessians, axis=0)
    mean_hessian = np.mean(stack, axis=0)
    squared_norms = [
        float(np.linalg.norm(hessian - mean_hessian, ord=2) ** 2)
        for hessian in stack
    ]
    count = stack.shape[0]
    return float(np.sqrt(((count - 1.0) / count) * np.sum(squared_norms)))


def _relative_hessian_uncertainty(sigma: float, hessian_magnitude: float) -> float:
    denominator = max(float(hessian_magnitude), np.finfo(float).tiny)
    return float(sigma) / denominator


def _vi_hessian_sigma(arguments: dict, state: dict, ctx) -> float:
    optimizer_samples = np.asarray(state["optimizer_samples"])
    parameter_samples = np.asarray(state["parameter_samples"])
    sample_count = optimizer_samples.shape[0]
    if sample_count < 3:
        raise ValueError("Independent Hessian jackknife requires at least 3 samples")

    iteration_results = {
        "qois": np.asarray(state["qois"]),
        "mean-qoi": np.asarray(state["mean_qoi"]),
        "errors": np.asarray(state["errors"]),
    }
    hessians = []
    for deleted in range(sample_count):
        keep = np.ones(sample_count, dtype=bool)
        keep[deleted] = False
        subset_state = _adaptive._build_vi_state(
            arguments,
            optimizer_samples[keep],
            parameter_samples[keep],
            _adaptive._subset_vi_results(iteration_results, keep),
        )
        raw_hessian = _vi._get_state_hessian(
            subset_state, ctx.newton_config.newton_hessian_type
        )
        hessians.append(
            _metric_projected_hessian(raw_hessian, arguments, ctx)
        )
    return _jackknife_spectral_sigma(hessians)


def _mf_hessian_sigma(arguments: dict, state: dict, ctx) -> float:
    optimizer_fom, optimizer_base, optimizer_extra = _adaptive._mf_arrays(state)
    sample_count = optimizer_fom.shape[0]
    if sample_count < 3:
        raise ValueError("Independent MFVI Hessian jackknife requires at least 3 FOM samples")

    hessians = []
    for deleted in range(sample_count):
        keep = np.ones(sample_count, dtype=bool)
        keep[deleted] = False
        subset_state = _adaptive._mf_estimator_state(
            arguments,
            optimizer_fom[keep],
            optimizer_base[keep],
            optimizer_extra,
            state["log_joint_terms_fom"][keep],
            state["log_joint_terms_rom_base"][keep],
            state["log_joint_terms_rom_only"],
        )
        raw_hessian = _vi._get_state_hessian(
            subset_state, ctx.newton_config.newton_hessian_type
        )
        hessians.append(
            _metric_projected_hessian(raw_hessian, arguments, ctx)
        )
    return _jackknife_spectral_sigma(hessians)


def _select_regularization(ctx, sigma: float, hessian_magnitude: float) -> float:
    relative_uncertainty = _relative_hessian_uncertainty(sigma, hessian_magnitude)
    dimensional_shift = ctx.config.scale * relative_uncertainty * hessian_magnitude
    regularization = float(np.clip(
        dimensional_shift,
        ctx.config.minimum,
        ctx.config.maximum,
    ))
    ctx.current_sigma = float(sigma)
    ctx.current_hessian_magnitude = float(hessian_magnitude)
    ctx.current_relative_uncertainty = float(relative_uncertainty)
    ctx.current_regularization = regularization
    ctx.sigma_history.append(float(sigma))
    ctx.hessian_magnitude_history.append(float(hessian_magnitude))
    ctx.relative_uncertainty_history.append(float(relative_uncertainty))
    ctx.history.append(regularization)
    print(
        "Jackknife Hessian regularization: "
        f"sigma_H={sigma:.6e}, ||H||_2={hessian_magnitude:.6e}, "
        f"rho_H={relative_uncertainty:.6e}, lambda={regularization:.6e}"
    )
    return regularization


@contextmanager
def _regularization_context(ctx):
    global _LAST_REGULARIZATION_HISTORY, _LAST_SIGMA_HISTORY
    global _LAST_HESSIAN_MAGNITUDE_HISTORY, _LAST_RELATIVE_UNCERTAINTY_HISTORY
    if ctx is None:
        yield
        return

    previous_vi_state = _vi._evaluate_vi_state
    previous_mf_state = _mf._evaluate_mf_vi_state
    previous_vi_get_hessian = _vi._get_state_hessian
    previous_mf_get_hessian = _mf._get_state_hessian
    previous_vi_newton_step = _vi._compute_newton_step
    previous_mf_newton_step = _mf._compute_newton_step

    def evaluate_vi_state(*args, **kwargs):
        state = previous_vi_state(*args, **kwargs)
        arguments = _adaptive._bind(previous_vi_state, args, kwargs)
        run_base = str(arguments.get("run_directory_base", ""))
        if "hessian_run_" in run_base:
            sigma = _vi_hessian_sigma(arguments, state, ctx)
            raw_hessian = previous_vi_get_hessian(
                state, ctx.newton_config.newton_hessian_type
            )
            projected = _metric_projected_hessian(raw_hessian, arguments, ctx)
            state["jackknife_hessian_sigma"] = sigma
            state["jackknife_hessian_magnitude"] = float(
                np.linalg.norm(projected, ord=2)
            )
        return state

    def evaluate_mf_state(*args, **kwargs):
        state = previous_mf_state(*args, **kwargs)
        arguments = _adaptive._bind(previous_mf_state, args, kwargs)
        iteration_directory = str(arguments.get("iteration_directory", ""))
        if iteration_directory.rstrip("/").endswith("/hessian"):
            sigma = _mf_hessian_sigma(arguments, state, ctx)
            raw_hessian = previous_mf_get_hessian(
                state, ctx.newton_config.newton_hessian_type
            )
            projected = _metric_projected_hessian(raw_hessian, arguments, ctx)
            state["jackknife_hessian_sigma"] = sigma
            state["jackknife_hessian_magnitude"] = float(
                np.linalg.norm(projected, ord=2)
            )
        return state

    def vi_get_hessian(state, newton_hessian_type):
        hessian = previous_vi_get_hessian(state, newton_hessian_type)
        if "jackknife_hessian_sigma" in state:
            _select_regularization(
                ctx,
                float(state["jackknife_hessian_sigma"]),
                float(state["jackknife_hessian_magnitude"]),
            )
        return hessian

    def mf_get_hessian(state, newton_hessian_type):
        hessian = previous_mf_get_hessian(state, newton_hessian_type)
        if "jackknife_hessian_sigma" in state:
            _select_regularization(
                ctx,
                float(state["jackknife_hessian_sigma"]),
                float(state["jackknife_hessian_magnitude"]),
            )
        return hessian

    def vi_newton_step(
        state,
        newton_regularization,
        newton_hessian_type="diagonal",
        metric_scale=None,
        hessian=None,
    ):
        regularization = (
            ctx.current_regularization
            if hessian is not None and ctx.current_regularization is not None
            else newton_regularization
        )
        return previous_vi_newton_step(
            state,
            regularization,
            newton_hessian_type=newton_hessian_type,
            metric_scale=metric_scale,
            hessian=hessian,
        )

    def mf_newton_step(
        state,
        newton_regularization,
        newton_hessian_type="diagonal",
        metric_scale=None,
        hessian=None,
    ):
        regularization = (
            ctx.current_regularization
            if hessian is not None and ctx.current_regularization is not None
            else newton_regularization
        )
        return previous_mf_newton_step(
            state,
            regularization,
            newton_hessian_type=newton_hessian_type,
            metric_scale=metric_scale,
            hessian=hessian,
        )

    token = _ACTIVE_CONTEXT.set(ctx)
    _vi._evaluate_vi_state = evaluate_vi_state
    _mf._evaluate_mf_vi_state = evaluate_mf_state
    _vi._get_state_hessian = vi_get_hessian
    _mf._get_state_hessian = mf_get_hessian
    _vi._compute_newton_step = vi_newton_step
    _mf._compute_newton_step = mf_newton_step
    try:
        yield
    finally:
        _vi._evaluate_vi_state = previous_vi_state
        _mf._evaluate_mf_vi_state = previous_mf_state
        _vi._get_state_hessian = previous_vi_get_hessian
        _mf._get_state_hessian = previous_mf_get_hessian
        _vi._compute_newton_step = previous_vi_newton_step
        _mf._compute_newton_step = previous_mf_newton_step
        _ACTIVE_CONTEXT.reset(token)
        _LAST_REGULARIZATION_HISTORY = list(ctx.history)
        _LAST_SIGMA_HISTORY = list(ctx.sigma_history)
        _LAST_HESSIAN_MAGNITUDE_HISTORY = list(ctx.hessian_magnitude_history)
        _LAST_RELATIVE_UNCERTAINTY_HISTORY = list(ctx.relative_uncertainty_history)


def run_vi(
    *args,
    jackknife_regularization_config: Optional[VIJackknifeRegularizationConfig] = None,
    **kwargs,
):
    """Run VI with optional independent-Hessian jackknife regularization."""
    ctx = _resolve_context(
        _adaptive._ORIGINAL_RUN_VI,
        args,
        kwargs,
        jackknife_regularization_config,
    )
    with _regularization_context(ctx):
        return _adaptive.run_vi(*args, adaptive_sample_config=None, **kwargs)


def run_mf_vi(
    *args,
    jackknife_regularization_config: Optional[VIJackknifeRegularizationConfig] = None,
    **kwargs,
):
    """Run MFVI with optional independent-Hessian jackknife regularization."""
    ctx = _resolve_context(
        _adaptive._ORIGINAL_RUN_MF_VI,
        args,
        kwargs,
        jackknife_regularization_config,
    )
    with _regularization_context(ctx):
        return _adaptive.run_mf_vi(*args, adaptive_sample_config=None, **kwargs)
