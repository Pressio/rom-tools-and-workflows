"""Run-directory policy for VI and MFVI model evaluations.

The VI drivers historically create a separate run directory for every model
sample. That remains the default because many file-backed models depend on
those directories. Lightweight in-memory models can opt out through the
public ``create_run_directories`` keyword without disabling VI restart,
history, or iteration-level output directories.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import inspect
import os
import warnings

from romtools.hpc.dispatchers import resolve_dispatcher
from romtools.workflows.inverse import mf_vi_drivers as _mf
from romtools.workflows.inverse import vi_drivers as _vi
from romtools.workflows.inverse.vi_sample_reuse import (
    _ORIGINAL_MF_VI_WITH_AUTO_ROM as _LEGACY_MF_VI_WITH_AUTO_ROM,
    _ORIGINAL_RUN_MF_VI as _LEGACY_RUN_MF_VI,
    _ORIGINAL_RUN_VI as _LEGACY_RUN_VI,
    mf_vi_with_auto_rom as _BASE_MF_VI_WITH_AUTO_ROM,
    run_mf_vi as _BASE_RUN_MF_VI,
    run_vi as _BASE_RUN_VI,
)


_UNSET = object()
_ACTIVE_CREATE_RUN_DIRECTORIES: ContextVar[object] = ContextVar(
    "romtools_vi_create_run_directories",
    default=None,
)

_ORIGINAL_VI_RUN_ITERATION = _vi.run_vi_iteration
_ORIGINAL_MF_VI_RUN_ITERATION = _mf.run_vi_iteration


def _resolve_create_run_directories(value) -> bool:
    inherited = _ACTIVE_CREATE_RUN_DIRECTORIES.get()
    if value is _UNSET:
        return True if inherited is None else bool(inherited)
    if not isinstance(value, bool):
        raise TypeError("create_run_directories must be a bool")
    return value


@contextmanager
def _run_directory_policy(create_run_directories):
    inherited = _ACTIVE_CREATE_RUN_DIRECTORIES.get()
    resolved = _resolve_create_run_directories(create_run_directories)
    if inherited is None and resolved:
        warnings.warn(
            "create_run_directories=True creates a separate run directory for "
            "every VI/MFVI model evaluation. This can cause significant "
            "filesystem overhead or inode pressure for lightweight or in-memory "
            "models. Set create_run_directories=False only when the model does "
            "not require per-evaluation run directories.",
            UserWarning,
            stacklevel=3,
        )
    token = _ACTIVE_CREATE_RUN_DIRECTORIES.set(resolved)
    try:
        yield resolved
    finally:
        _ACTIVE_CREATE_RUN_DIRECTORIES.reset(token)


def _prepare_initial_iteration_directory(signature_source, args, kwargs, resolved_policy):
    """Create iteration_0 when per-sample directories are disabled.

    Normally the first model evaluation creates ``iteration_0`` as a parent of
    its per-sample run directory. When ``create_run_directories=False`` that
    side effect intentionally disappears, but restart/history output still
    requires the iteration-level directory. Create only that directory here,
    once at the outermost policy layer, before the driver writes its first
    restart file.
    """
    inherited = _ACTIVE_CREATE_RUN_DIRECTORIES.get()
    if inherited is not None or resolved_policy:
        return

    signature = inspect.signature(signature_source)
    bind_kwargs = {
        key: value for key, value in kwargs.items()
        if key in signature.parameters
    }
    bound = signature.bind_partial(*args, **bind_kwargs)
    bound.apply_defaults()

    if bound.arguments.get("restart_file") is not None:
        return

    absolute_work_dir = bound.arguments.get("absolute_work_dir")
    if absolute_work_dir is None:
        absolute_work_dir = bound.arguments.get("absolute_vi_directory")
    if absolute_work_dir is None:
        absolute_work_dir = os.getcwd() + "/work/"

    dispatcher = resolve_dispatcher(bound.arguments.get("dispatcher"))
    dispatcher.create_empty_dir(f"{absolute_work_dir}/iteration_0")


def _directory_aware_vi_iteration(*args, **kwargs):
    kwargs.setdefault(
        "create_run_directories",
        _resolve_create_run_directories(_UNSET),
    )
    return _ORIGINAL_VI_RUN_ITERATION(*args, **kwargs)


def _directory_aware_mf_vi_iteration(*args, **kwargs):
    kwargs.setdefault(
        "create_run_directories",
        _resolve_create_run_directories(_UNSET),
    )
    return _ORIGINAL_MF_VI_RUN_ITERATION(*args, **kwargs)


# The original VI and MFVI drivers import run_vi_iteration into their module
# namespaces. Patch those module-level references once so all evaluation paths,
# including line searches and sample-reuse controllers, inherit the public
# directory policy. The resolved boolean is passed explicitly before any
# multiprocessing workers are launched.
_vi.run_vi_iteration = _directory_aware_vi_iteration
_mf.run_vi_iteration = _directory_aware_mf_vi_iteration


def run_vi(*args, create_run_directories=_UNSET, **kwargs):
    """Run VI with optional per-evaluation run-directory creation."""
    resolved = _resolve_create_run_directories(create_run_directories)
    _prepare_initial_iteration_directory(_LEGACY_RUN_VI, args, kwargs, resolved)
    with _run_directory_policy(resolved):
        return _BASE_RUN_VI(*args, **kwargs)


def run_mf_vi(*args, create_run_directories=_UNSET, **kwargs):
    """Run MFVI with optional per-evaluation run-directory creation."""
    resolved = _resolve_create_run_directories(create_run_directories)
    _prepare_initial_iteration_directory(_LEGACY_RUN_MF_VI, args, kwargs, resolved)
    with _run_directory_policy(resolved):
        return _BASE_RUN_MF_VI(*args, **kwargs)


def mf_vi_with_auto_rom(*args, create_run_directories=_UNSET, **kwargs):
    """Run auto-ROM MFVI with the same run-directory policy."""
    resolved = _resolve_create_run_directories(create_run_directories)
    _prepare_initial_iteration_directory(
        _LEGACY_MF_VI_WITH_AUTO_ROM,
        args,
        kwargs,
        resolved,
    )
    with _run_directory_policy(resolved):
        return _BASE_MF_VI_WITH_AUTO_ROM(*args, **kwargs)
