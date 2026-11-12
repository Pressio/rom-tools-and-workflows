"""Compatibility imports for shared adaptive-training utilities."""

from .._adaptive.validation import (
    KFoldValidation, SpaceFillingValidationContext, relative_l2_error,
    _build_rom, _error_value,
)
