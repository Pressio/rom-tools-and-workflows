"""Helpers for resolving an optional dispatcher argument to a concrete dispatcher."""

from typing import Optional

from romtools.hpc.dispatchers.base_dispatcher import BaseDispatcher
from romtools.hpc.dispatchers.local_dispatcher import LocalDispatcher


def resolve_dispatcher(dispatcher: Optional[BaseDispatcher] = None) -> BaseDispatcher:
    """Fall back to local execution when the caller supplies no dispatcher."""
    return dispatcher if dispatcher is not None else LocalDispatcher()


def resolve_local_dispatcher(dispatcher: Optional[BaseDispatcher] = None) -> LocalDispatcher:
    """
    Return a local dispatcher, reusing the supplied one when it is already local.

    Use this for work that always runs in-process, so that it stays on the local
    machine even when the rest of the workflow is dispatched to a remote host.
    """
    if isinstance(dispatcher, LocalDispatcher):
        return dispatcher
    return LocalDispatcher()
