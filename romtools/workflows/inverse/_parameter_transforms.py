"""Parameter transforms shared by inverse workflows."""

import numpy as np


def _normalize_bounded_parameter_handling(bounded_parameter_handling: str):
    handling = bounded_parameter_handling.strip().lower()
    if handling not in ('clip', 'transform'):
        raise ValueError(
            f"Unsupported bounded_parameter_handling '{bounded_parameter_handling}'. "
            "Supported options are 'clip' and 'transform'."
        )
    return handling


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped_values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped_values))


def _logit(values: np.ndarray) -> np.ndarray:
    return np.log(values / (1.0 - values))


def _normalize_transform_map(transform_map: str) -> str:
    normalized = transform_map.strip().lower()
    if normalized in ('sigmoid', 'logistic'):
        return 'sigmoid'
    if normalized in ('arctan', 'atan'):
        return 'arctan'
    raise ValueError(
        f"Unsupported transform_map '{transform_map}'. "
        "Supported options are 'sigmoid' and 'arctan'."
    )


def _transform_unit_from_optimizer(optimizer_values: np.ndarray,
                                   transform_map: str) -> np.ndarray:
    transform_map = _normalize_transform_map(transform_map)
    if transform_map == 'sigmoid':
        return _sigmoid(optimizer_values)
    return 0.5 + np.arctan(optimizer_values) / np.pi


def _inverse_transform_unit_to_optimizer(unit_values: np.ndarray,
                                         transform_map: str) -> np.ndarray:
    transform_map = _normalize_transform_map(transform_map)
    if transform_map == 'sigmoid':
        return _logit(unit_values)
    return np.tan(np.pi * (unit_values - 0.5))


def _compute_transform_jacobian_diagonal(optimizer_values: np.ndarray,
                                         parameter_mins: np.ndarray,
                                         parameter_maxes: np.ndarray,
                                         transform_interior_margin: float,
                                         transform_map: str = 'sigmoid') -> np.ndarray:
    margin_scale = 1.0 - 2.0 * transform_interior_margin
    if margin_scale <= 0.0:
        raise ValueError("transform_interior_margin must be in [0.0, 0.5).")
    transform_map = _normalize_transform_map(transform_map)
    bounded_unit = _transform_unit_from_optimizer(optimizer_values, transform_map)
    if transform_map == 'sigmoid':
        base_jacobian = bounded_unit * (1.0 - bounded_unit)
    else:
        base_jacobian = 1.0 / (np.pi * (1.0 + optimizer_values ** 2))
    return (
        (parameter_maxes - parameter_mins)
        * margin_scale
        * base_jacobian
    )


def _transform_optimizer_to_parameter(optimizer_samples: np.ndarray,
                                      parameter_mins: np.ndarray,
                                      parameter_maxes: np.ndarray,
                                      transform_interior_margin: float = 0.0,
                                      transform_map: str = 'sigmoid') -> np.ndarray:
    margin_scale = 1.0 - 2.0 * transform_interior_margin
    if margin_scale <= 0.0:
        raise ValueError("transform_interior_margin must be in [0.0, 0.5).")
    bounded_unit = transform_interior_margin + margin_scale * _transform_unit_from_optimizer(
        optimizer_samples,
        transform_map,
    )
    return parameter_mins[None, :] + (parameter_maxes - parameter_mins)[None, :] * bounded_unit


def _transform_parameter_to_optimizer(parameter_samples: np.ndarray,
                                      parameter_mins: np.ndarray,
                                      parameter_maxes: np.ndarray,
                                      transform_interior_margin: float = 0.0,
                                      transform_map: str = 'sigmoid') -> np.ndarray:
    parameter_ranges = parameter_maxes - parameter_mins
    unit_values = (parameter_samples - parameter_mins[None, :]) / parameter_ranges[None, :]
    margin_scale = 1.0 - 2.0 * transform_interior_margin
    if margin_scale <= 0.0:
        raise ValueError("transform_interior_margin must be in [0.0, 0.5).")
    unit_values = (unit_values - transform_interior_margin) / margin_scale
    unit_values = np.clip(unit_values, 1e-12, 1.0 - 1e-12)
    return _inverse_transform_unit_to_optimizer(unit_values, transform_map)
