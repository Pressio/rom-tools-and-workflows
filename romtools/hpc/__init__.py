"""Tools for running ROM workflows on HPC resources.

Workflows take a dispatcher, which decides where model evaluations run:
LocalDispatcher keeps them in place, RemoteDispatcher sends them to a cluster
over SSH and brings the results back.
"""

from romtools.hpc.logger import Logger
from romtools.hpc.connection import Connection, Result
from romtools.hpc.configuration import Configuration, ConfigurationError
from romtools.hpc.dispatchers import (
    BaseDispatcher,
    LocalDispatcher,
    RemoteDispatcher,
    resolve_dispatcher,
    resolve_local_dispatcher,
)

__all__ = [
    "BaseDispatcher",
    "LocalDispatcher",
    "RemoteDispatcher",
    "resolve_dispatcher",
    "resolve_local_dispatcher",
    "Configuration",
    "ConfigurationError",
    "Connection",
    "Result",
    "Logger",
]
