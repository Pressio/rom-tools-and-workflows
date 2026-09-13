"""Backwards compatibility for workflow working-directory keywords."""

import functools
import inspect
import warnings


_UNSET = object()


def standardize_work_dir_argument(function, old_name):
    """Expose ``absolute_work_dir`` while accepting a deprecated old name."""
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    old_index = next(
        i for i, parameter in enumerate(parameters) if parameter.name == old_name
    )
    parameters[old_index] = parameters[old_index].replace(name="absolute_work_dir")

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        new_value = kwargs.pop("absolute_work_dir", _UNSET)
        old_value = kwargs.get(old_name, _UNSET)

        if new_value is not _UNSET:
            if old_value is not _UNSET or old_index < len(args):
                raise TypeError("work directory specified more than once")
            kwargs[old_name] = new_value
        elif old_value is not _UNSET:
            warnings.warn(
                f"'{old_name}' is deprecated; use 'absolute_work_dir' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        return function(*args, **kwargs)

    wrapper.__signature__ = signature.replace(parameters=parameters)
    return wrapper


def patch_work_dir_argument(module, function_name, old_name):
    """Apply the compatibility wrapper to a workflow module."""
    function = standardize_work_dir_argument(getattr(module, function_name), old_name)
    setattr(module, function_name, function)
    return function
