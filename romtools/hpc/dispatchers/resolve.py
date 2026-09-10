"""Helpers for resolving an optional dispatcher argument to a concrete dispatcher."""

from typing import Optional

from romtools.hpc.dispatchers.base_dispatcher import BaseDispatcher
from romtools.hpc.dispatchers.local_dispatcher import LocalDispatcher


# argv=[] because nobody asked for this dispatcher: reading the host program's
# command line would take its own switches as configuration.
def _default_dispatcher() -> LocalDispatcher:
    return LocalDispatcher(argv=[])


def resolve_dispatcher(dispatcher: Optional[BaseDispatcher] = None) -> BaseDispatcher:
    """Fall back to local execution when the caller supplies no dispatcher."""
    return dispatcher if dispatcher is not None else _default_dispatcher()


def resolve_local_dispatcher(dispatcher: Optional[BaseDispatcher] = None) -> LocalDispatcher:
    """
    Return a local dispatcher, reusing the supplied one when it is already local.

    Keeps in-process work on this machine even when the rest of the workflow is
    dispatched to a remote host.
    """
    if isinstance(dispatcher, LocalDispatcher):
        return dispatcher
    return _default_dispatcher()
