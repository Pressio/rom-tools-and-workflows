"""Run-directory policy for VI and MFVI model evaluations.

The VI drivers historically create a separate run directory for every model
sample.  That remains the default because many file-backed models depend on
those directories.  Lightweight in-memory models can opt out through the
public ``create_run_directories`` keyword without disabling VI restart,
history, or iteration-level output directories.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import warnings

from romtools.workflows.inverse._inverse_utils import run_vi_iteration as _run_vi_iteration


_UNSET = object()
_ACTIVE_CREATE_RUN_DIRECTORIES: ContextVar[object] = ContextVar(
    "romtools_vi_create_run_directories",
    default=None,
)


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


def run_vi_iteration(*args, **kwargs):
    """Pass the active policy explicitly before launching model workers."""
    kwargs.setdefault("create_run_directories", _resolve_create_run_directories(_UNSET))
    return _run_vi_iteration(*args, **kwargs)
