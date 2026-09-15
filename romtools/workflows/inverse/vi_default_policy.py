"""Centralized public defaults for VI and MF-VI optimization.

This module updates the shared configuration dataclasses used by both VI and
MF-VI.  It is imported by the public VI routing layer during inverse-package
initialization so direct and package-level imports observe the same defaults.
"""

from __future__ import annotations

from romtools.workflows.inverse.vi_optimization_methods import (
    VINewtonOptimizerConfig,
    VILegacyLineSearchConfig,
    VIStochasticNonmonotoneLineSearchConfig,
)


def _set_dataclass_default(cls, field_name: str, value) -> None:
    """Update a dataclass field and its generated ``__init__`` default."""
    fields = list(cls.__dataclass_fields__)
    field_index = fields.index(field_name)
    defaults = list(cls.__init__.__defaults__ or ())
    first_default_index = len(fields) - len(defaults)
    default_index = field_index - first_default_index
    if default_index < 0:
        raise RuntimeError(
            f"Cannot set default for non-default dataclass field {field_name!r}."
        )
    defaults[default_index] = value
    cls.__init__.__defaults__ = tuple(defaults)
    cls.__dataclass_fields__[field_name].default = value


# Newton defaults shared by single- and multifidelity VI.
_set_dataclass_default(VINewtonOptimizerConfig, "newton_metric", "natural")
_set_dataclass_default(VINewtonOptimizerConfig, "newton_regularization", 5.0e-4)
_set_dataclass_default(VINewtonOptimizerConfig, "newton_hessian_type", "full")

# Use a useful nonzero initial line-search scale by default.  The maximum step
# remains unconstrained unless the caller specifies it explicitly.
_set_dataclass_default(VILegacyLineSearchConfig, "initial_step_size", 0.1)
_set_dataclass_default(
    VIStochasticNonmonotoneLineSearchConfig,
    "initial_step_size",
    0.1,
)
