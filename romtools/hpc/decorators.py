# hpctools/dispatch_decorator.py
from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, TypeVar, cast

## ----------------------------------------------------------------------------
## Dispatching work to remote host

T = TypeVar("T", bound=Callable[..., Any])

def dispatch(fn: T) -> T:
    """Execute the method on a remote host if a dispatcher is available."""
    @wraps(fn)
    def wrapper(self, run_directory: str, parameter_sample: Dict[str, Any]):
        dispatcher = getattr(self, "dispatcher", None)

        if dispatcher is None:
            return fn(self, run_directory, parameter_sample)

        return dispatcher.run(
            fn=fn,
            self_obj=self,
            run_directory=run_directory,
            parameter_sample=parameter_sample,
        )

    return cast(T, wrapper)

## ----------------------------------------------------------------------------
## Ensuring a connection exists before executing a method

def require_connection(func):
    """Ensure a connection exists before executing a method."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.conn is None:
            raise RuntimeError(f"No connection established. Call __connect_to_remote() before using {func.__name__}.")
        return func(self, *args, **kwargs)
    return wrapper
