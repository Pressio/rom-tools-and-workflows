"""Lagged-curvature support for full-covariance natural Newton VI/MF-VI.

This layer extends :mod:`full_covariance_newton` with the same accepted-state
exponential Hessian averaging used by the diagonal Newton implementation.
Rejected line-search candidates do not update the running Hessian.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy

import numpy as np

from romtools.workflows.inverse import full_covariance_newton as _base
from romtools.workflows.inverse.vi_optimization_methods import (
    VINewtonOptimizerConfig,
    _normalize_newton_curvature_strategy,
)


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

    strategy = _normalize_newton_curvature_strategy(
        config.newton_curvature_strategy
    )
    if strategy not in ("same_sample", "lagged"):
        raise NotImplementedError(
            "Full-covariance Newton currently supports "
            "newton_curvature_strategy='same_sample' or 'lagged'."
        )
    if strategy == "lagged":
        beta = float(config.newton_hessian_averaging_factor)
        if not np.isfinite(beta) or not 0.0 <= beta < 1.0:
            raise ValueError(
                "newton_hessian_averaging_factor must be finite and in [0, 1)."
            )
    return config


@contextmanager
def _accepted_state_hessian_averaging(config, estimator_name: str):
    """Average raw Hessians only when their evaluated state is accepted.

    The full-covariance drivers evaluate Newton information for line-search
    candidates before knowing whether the candidate will be accepted.  We form
    a prospective exponentially averaged Hessian for every candidate, but only
    commit it when the driver subsequently asks for that state's gradient norm;
    that call occurs for the initial state and after an accepted line-search
    candidate, not for rejected candidates.
    """
    strategy = _normalize_newton_curvature_strategy(
        config.newton_curvature_strategy
    )
    if strategy == "same_sample":
        yield
        return

    beta = float(config.newton_hessian_averaging_factor)
    original_estimator = getattr(_base, estimator_name)
    original_natural_gradient_norm = _base._natural_gradient_norm

    running_hessian = None
    pending_hessian_id = None
    pending_running_hessian = None

    def averaged_estimator(*args, **kwargs):
        nonlocal pending_hessian_id, pending_running_hessian
        current = np.asarray(original_estimator(*args, **kwargs), dtype=float)
        if running_hessian is None:
            prospective = current.copy()
        else:
            prospective = beta * running_hessian + (1.0 - beta) * current
        prospective = 0.5 * (prospective + prospective.T)
        pending_hessian_id = id(prospective)
        pending_running_hessian = prospective.copy()
        return prospective

    def commit_gradient_norm(state):
        nonlocal running_hessian, pending_hessian_id, pending_running_hessian
        state_hessian = state.get("newton_hessian_full")
        if (
            state_hessian is not None
            and pending_hessian_id is not None
            and id(state_hessian) == pending_hessian_id
        ):
            running_hessian = pending_running_hessian.copy()
            pending_hessian_id = None
            pending_running_hessian = None
        return original_natural_gradient_norm(state)

    setattr(_base, estimator_name, averaged_estimator)
    _base._natural_gradient_norm = commit_gradient_norm
    try:
        yield
    finally:
        setattr(_base, estimator_name, original_estimator)
        _base._natural_gradient_norm = original_natural_gradient_norm


def run_vi(*args, **kwargs):
    """Run full-covariance VI with same-sample or lagged Newton curvature."""
    config = _validate_newton_config(kwargs.get("optimizer_config"))
    call_kwargs = dict(kwargs)
    call_kwargs["optimizer_method"] = "gradient"
    call_kwargs["optimizer_config"] = _base._gradient_config_from_newton(config)
    with _accepted_state_hessian_averaging(config, "estimate_ordinary_hessian"):
        with _base._patched_single_fidelity_newton(config):
            return _base._fc_vi.run_vi(*args, **call_kwargs)


def run_mf_vi(*args, **kwargs):
    """Run full-covariance MF-VI with same-sample or lagged Newton curvature."""
    config = _validate_newton_config(kwargs.get("optimizer_config"))
    call_kwargs = dict(kwargs)
    call_kwargs["optimizer_method"] = "gradient"
    call_kwargs["optimizer_config"] = _base._gradient_config_from_newton(config)
    with _accepted_state_hessian_averaging(
        config, "estimate_mf_ordinary_hessian"
    ):
        with _base._patched_mf_newton(config):
            return _base._fc_mf.run_mf_vi(*args, **call_kwargs)


def mf_vi_with_auto_rom(*args, **kwargs):
    """Run auto-ROM full-covariance MF-VI with lagged-capable Newton."""
    original_run_mf_vi = _base._fc_auto.run_mf_vi
    _base._fc_auto.run_mf_vi = run_mf_vi
    try:
        return _base._fc_auto.mf_vi_with_auto_rom(*args, **kwargs)
    finally:
        _base._fc_auto.run_mf_vi = original_run_mf_vi
