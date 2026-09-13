"""Compatibility helpers for standardized workflow working-directory arguments."""

import functools
import inspect
import sys
import warnings


_UNSET = object()


def standardize_work_dir_argument(function, deprecated_name):
    """Wrap a workflow so ``absolute_work_dir`` replaces a legacy keyword.

    Positional calls retain their historical behavior. Calls using the legacy
    keyword continue to work and emit a deprecation warning. Supplying both
    names is rejected as an ambiguous duplicate specification.
    """
    if getattr(function, "_romtools_work_dir_deprecated_name", None) == deprecated_name:
        return function

    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    deprecated_index = None

    for index, parameter in enumerate(parameters):
        if parameter.name == deprecated_name:
            deprecated_index = index
            parameters[index] = parameter.replace(name="absolute_work_dir")
            break

    if deprecated_index is None:
        raise ValueError(
            f"{function.__qualname__} has no parameter named {deprecated_name!r}"
        )

    @functools.wraps(function)
    def wrapper(*args, absolute_work_dir=_UNSET, **kwargs):
        legacy_keyword_supplied = deprecated_name in kwargs
        legacy_positional_supplied = deprecated_index < len(args)

        if legacy_keyword_supplied:
            if absolute_work_dir is not _UNSET:
                raise TypeError(
                    "Specify only 'absolute_work_dir'; do not also provide "
                    f"deprecated '{deprecated_name}'."
                )

            # Internal calls still use the implementation's legacy keyword.
            # Warn only when the deprecated spelling originates outside the
            # romtools workflow implementation itself.
            caller_module = sys._getframe(1).f_globals.get("__name__", "")
            if not caller_module.startswith("romtools.workflows"):
                warnings.warn(
                    f"'{deprecated_name}' is deprecated and will be removed in "
                    "a future release; use 'absolute_work_dir' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        elif absolute_work_dir is not _UNSET:
            if legacy_positional_supplied:
                raise TypeError(
                    "The work directory was provided both positionally and via "
                    "'absolute_work_dir'."
                )
            kwargs[deprecated_name] = absolute_work_dir

        return function(*args, **kwargs)

    wrapper.__signature__ = signature.replace(parameters=parameters)
    if wrapper.__doc__:
        wrapper.__doc__ = wrapper.__doc__.replace(deprecated_name, "absolute_work_dir")
    wrapper._romtools_work_dir_deprecated_name = deprecated_name
    return wrapper


def patch_work_dir_argument(module, function_name, deprecated_name):
    """Patch one public workflow function in-place and return the wrapper."""
    function = getattr(module, function_name)
    wrapped = standardize_work_dir_argument(function, deprecated_name)
    setattr(module, function_name, wrapped)
    return wrapped
