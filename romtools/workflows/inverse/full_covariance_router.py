"""Public routing for the opt-in full-covariance Gaussian VI family.

The pre-existing multivariate parameter-space behavior is retained when
``variational_distribution`` is omitted so existing callers and restart files
remain backward compatible. True full covariance is selected explicitly with
``variational_distribution='full_covariance'``. The old fixed-correlation
multivariate path can therefore be deprecated separately without making this
feature release a silent behavior change.
"""

from __future__ import annotations

import warnings

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


def _normalize_requested_family(value):
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in ("full", "full_covariance"):
        return "full_covariance"
    if normalized == "multivariate":
        warnings.warn(
            "variational_distribution='multivariate' is deprecated; use "
            "'full_covariance' for freely evolving covariance. Omitting "
            "variational_distribution retains the legacy fixed-correlation path.",
            DeprecationWarning,
            stacklevel=3,
        )
        return "full_covariance"
    if normalized in ("diagonal", "mean_field"):
        return "diagonal"
    raise ValueError(
        f"Unsupported variational_distribution '{value}'. Supported options "
        "are 'diagonal' and 'full_covariance'."
    )


def run_vi(*args, variational_distribution=None, max_covariance_log_step=1.0, **kwargs):
    family = _normalize_requested_family(variational_distribution)
    if family != "full_covariance":
        return _legacy_run_vi(*args, **kwargs)
    return _full_run_vi(
        *args,
        variational_distribution="full_covariance",
        max_covariance_log_step=max_covariance_log_step,
        **kwargs,
    )


def run_mf_vi(*args, variational_distribution=None, max_covariance_log_step=1.0, **kwargs):
    family = _normalize_requested_family(variational_distribution)
    if family != "full_covariance":
        return _legacy_run_mf_vi(*args, **kwargs)
    return _full_run_mf_vi(
        *args,
        variational_distribution="full_covariance",
        max_covariance_log_step=max_covariance_log_step,
        **kwargs,
    )


def mf_vi_with_auto_rom(
    *args,
    variational_distribution=None,
    max_covariance_log_step=1.0,
    **kwargs,
):
    family = _normalize_requested_family(variational_distribution)
    if family != "full_covariance":
        return _legacy_auto_mf_vi(*args, **kwargs)
    return _full_auto_mf_vi(
        *args,
        variational_distribution="full_covariance",
        max_covariance_log_step=max_covariance_log_step,
        **kwargs,
    )
