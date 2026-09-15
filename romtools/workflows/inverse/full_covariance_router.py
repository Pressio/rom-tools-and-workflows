"""Public routing for Gaussian VI variational families.

``prior_parameter_space`` defines the Bayesian prior. The required
``initial_variational_parameter_space`` independently defines both the
variational family and its initial moments:

* ``GaussianParameterSpace`` -> diagonal/mean-field VI;
* ``MultivariateGaussianParameterSpace`` -> true full-covariance VI.

A multivariate variational initializer is therefore never interpreted as the
legacy fixed-correlation family.
"""

from __future__ import annotations

import inspect

from romtools.workflows.inverse import vi_drivers as _vi_drivers
from romtools.workflows.inverse import mf_vi_drivers as _mf_vi_drivers
from romtools.workflows.inverse.vi_run_directory_policy import (
    run_vi as _legacy_run_vi,
    run_mf_vi as _legacy_run_mf_vi,
    mf_vi_with_auto_rom as _legacy_auto_mf_vi,
)
from romtools.workflows.inverse.full_covariance_vi_drivers import (
    run_vi as _full_run_vi,
)
from romtools.workflows.inverse.full_covariance_mf_vi_drivers import (
    run_mf_vi as _full_run_mf_vi,
)
from romtools.workflows.inverse.full_covariance_auto_mf_vi import (
    mf_vi_with_auto_rom as _full_auto_mf_vi,
)
from romtools.workflows.inverse.full_covariance_newton import (
    run_vi as _full_newton_run_vi,
    run_mf_vi as _full_newton_run_mf_vi,
    mf_vi_with_auto_rom as _full_newton_auto_mf_vi,
)


def _validate_independent_gaussian_parameter_spaces(
    prior_parameter_space,
    initial_variational_parameter_space=None,
):
    """Validate prior and variational initializer without coupling families."""
    if initial_variational_parameter_space is None:
        raise TypeError(
            "initial_variational_parameter_space is required for VI and MF-VI. "
            "Use GaussianParameterSpace for diagonal VI or "
            "MultivariateGaussianParameterSpace for full-covariance VI."
        )

    prior_names, prior_mean, prior_covariance, prior_distribution = (
        _vi_drivers._extract_gaussian_parameter_space(
            prior_parameter_space,
            argument_name="prior_parameter_space",
        )
    )
    (
        initial_names,
        initial_mean,
        initial_covariance,
        variational_distribution,
    ) = _vi_drivers._extract_gaussian_parameter_space(
        initial_variational_parameter_space,
        argument_name="initial_variational_parameter_space",
    )

    if list(prior_names) != list(initial_names):
        raise ValueError(
            "prior_parameter_space and initial_variational_parameter_space must "
            "define the same parameter names in the same order."
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


_vi_drivers._validate_gaussian_parameter_spaces = (
    _validate_independent_gaussian_parameter_spaces
)
_mf_vi_drivers._validate_gaussian_parameter_spaces = (
    _validate_independent_gaussian_parameter_spaces
)


def _bound_arguments(function, args, kwargs):
    return inspect.signature(function).bind_partial(*args, **kwargs).arguments


def _supplied_initializer(function, args, kwargs):
    """Return the explicitly supplied variational initializer, if any."""
    return _bound_arguments(function, args, kwargs).get(
        "initial_variational_parameter_space"
    )


def _optimizer_method(function, args, kwargs) -> str:
    """Return the requested optimizer method using the public driver's defaults."""
    arguments = _bound_arguments(function, args, kwargs)
    return str(arguments.get("optimizer_method", "gradient")).strip().lower()


def _uses_natural_newton(function, args, kwargs) -> bool:
    """Return whether the request is specifically full-covariance natural Newton."""
    arguments = _bound_arguments(function, args, kwargs)
    if str(arguments.get("optimizer_method", "gradient")).strip().lower() != "newton":
        return False
    config = arguments.get("optimizer_config")
    if config is None:
        return False
    return str(getattr(config, "newton_metric", "standard")).strip().lower() == "natural"


def _variational_family(initial_variational_parameter_space) -> str:
    """Infer the variational family exclusively from the initializer type."""
    _, _, _, distribution = _vi_drivers._extract_gaussian_parameter_space(
        initial_variational_parameter_space,
        argument_name="initial_variational_parameter_space",
    )
    if distribution == "diagonal":
        return "diagonal"
    if distribution == "multivariate":
        return "full_covariance"
    raise ValueError(f"Unsupported variational Gaussian family '{distribution}'.")


def _require_coupled_mf_base(function, args, kwargs):
    """Enforce sample-by-sample HF/LF coupling for the MF control variate."""
    arguments = _bound_arguments(function, args, kwargs)
    strategy = arguments.get("rom_base_sampling_strategy", "coupled")
    if str(strategy).strip().lower() != "coupled":
        raise ValueError(
            "Full-covariance MF-VI requires rom_base_sampling_strategy='coupled' "
            "so the HF contribution and LF base control variate share the same "
            "optimizer samples."
        )


def run_vi(*args, max_covariance_log_step=1.0, **kwargs):
    initializer = _supplied_initializer(_legacy_run_vi, args, kwargs)
    if initializer is None:
        return _legacy_run_vi(*args, **kwargs)
    family = _variational_family(initializer)
    if family == "diagonal":
        return _legacy_run_vi(*args, **kwargs)
    implementation = (
        _full_newton_run_vi
        if _uses_natural_newton(_legacy_run_vi, args, kwargs)
        else _full_run_vi
    )
    return implementation(
        *args,
        max_covariance_log_step=max_covariance_log_step,
        **kwargs,
    )


def run_mf_vi(*args, max_covariance_log_step=1.0, **kwargs):
    initializer = _supplied_initializer(_legacy_run_mf_vi, args, kwargs)
    if initializer is None:
        return _legacy_run_mf_vi(*args, **kwargs)
    family = _variational_family(initializer)
    if family == "diagonal":
        return _legacy_run_mf_vi(*args, **kwargs)
    _require_coupled_mf_base(_full_run_mf_vi, args, kwargs)
    implementation = (
        _full_newton_run_mf_vi
        if _uses_natural_newton(_legacy_run_mf_vi, args, kwargs)
        else _full_run_mf_vi
    )
    return implementation(
        *args,
        max_covariance_log_step=max_covariance_log_step,
        **kwargs,
    )


def mf_vi_with_auto_rom(*args, max_covariance_log_step=1.0, **kwargs):
    initializer = _supplied_initializer(_legacy_auto_mf_vi, args, kwargs)
    if initializer is None:
        return _legacy_auto_mf_vi(*args, **kwargs)
    family = _variational_family(initializer)
    if family == "diagonal":
        return _legacy_auto_mf_vi(*args, **kwargs)
    _require_coupled_mf_base(_full_auto_mf_vi, args, kwargs)
    implementation = (
        _full_newton_auto_mf_vi
        if _uses_natural_newton(_legacy_auto_mf_vi, args, kwargs)
        else _full_auto_mf_vi
    )
    return implementation(
        *args,
        max_covariance_log_step=max_covariance_log_step,
        **kwargs,
    )
