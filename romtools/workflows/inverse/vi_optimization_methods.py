"""Optimization methods and configs used by inverse VI workflows."""

import copy
from dataclasses import dataclass, fields

import numpy as np


@dataclass
class VIGradientOptimizerConfig:
    gradient_method: str = 'standard'
    gradient_norm_tolerance: float = 1e-5
    max_iterations: int = 1000
    max_log_std_update: float = 0.5
    min_variational_std: float = 1e-6
    max_variational_std: float = 1e6


@dataclass
class VINewtonOptimizerConfig:
    gradient_norm_tolerance: float = 1e-5
    max_iterations: int = 1000
    max_log_std_update: float = 0.5
    min_variational_std: float = 1e-6
    max_variational_std: float = 1e6
    newton_metric: str = 'standard'
    newton_regularization: float = 1e-1


@dataclass
class VILegacyLineSearchConfig:
    initial_step_size: float = 1e-2
    max_step_size: float = np.inf
    step_size_growth_factor: float = 1.01
    step_size_decay_factor: float = 3.0
    max_step_size_decrease_trys: int = 5
    relaxation_parameter: float = 2.05
    line_search_objective: str = 'elbo'
    line_search_sample_growth_factor: float = 1.0
    log_std_learning_rate_factor: float = 1e-1


@dataclass
class VIStochasticNonmonotoneLineSearchConfig:
    initial_step_size: float = 1e-2
    max_step_size: float = np.inf
    step_size_growth_factor: float = 1.05
    step_size_decay_factor: float = 2.0
    max_step_size_decrease_trys: int = 5
    relaxation_parameter: float = 2.05
    line_search_objective: str = 'elbo'
    line_search_nonmonotone_window: int = 5
    line_search_armijo_coefficient: float = 1e-8
    line_search_uncertainty_sigma: float = 50.0
    line_search_sample_growth_factor: float = 2.0
    log_std_learning_rate_factor: float = 1.0


def _normalize_optimization_method(optimization_method: str):
    method = optimization_method.strip().lower()
    if method not in ('gradient', 'newton'):
        raise ValueError(
            f"Unsupported optimization_method '{optimization_method}'. "
            "Supported options are 'gradient' and 'newton'."
        )
    return method


def _normalize_line_search_objective(line_search_objective: str):
    objective = line_search_objective.strip().lower()
    if objective not in ('elbo', 'mse'):
        raise ValueError(
            f"Unsupported line_search_objective '{line_search_objective}'. "
            "Supported options are 'elbo' and 'mse'."
        )
    return objective


def _normalize_line_search_method(line_search_method: str):
    method = line_search_method.strip().lower()
    if method in ('legacy', 'basic'):
        return 'legacy'
    if method in ('stochastic_nonmonotone', 'stochastic', 'advanced'):
        return 'stochastic_nonmonotone'
    raise ValueError(
        f"Unsupported line_search_method '{line_search_method}'. "
        "Supported options are 'legacy' and 'stochastic_nonmonotone'."
    )


def _normalize_newton_metric(newton_metric: str):
    metric = newton_metric.strip().lower()
    if metric in ('standard', 'euclidean'):
        return 'standard'
    if metric in ('natural',):
        return 'natural'
    raise ValueError(
        f"Unsupported newton_metric '{newton_metric}'. "
        "Supported options are 'standard' and 'natural'."
    )


def _resolve_optimizer_config(optimizer_method: str,
                              optimizer_config,
                              legacy_kwargs: dict,
                              default_gradient_config: VIGradientOptimizerConfig = None,
                              default_newton_config: VINewtonOptimizerConfig = None):
    explicit_optimizer_config = optimizer_config is not None
    normalized_optimizer_method = _normalize_optimization_method(optimizer_method)
    gradient_config = copy.deepcopy(default_gradient_config) if default_gradient_config is not None else (
        VIGradientOptimizerConfig()
    )
    newton_config = copy.deepcopy(default_newton_config) if default_newton_config is not None else (
        VINewtonOptimizerConfig()
    )

    optimizer_type_by_method = {
        'gradient': VIGradientOptimizerConfig,
        'newton': VINewtonOptimizerConfig,
    }
    default_config_by_method = {
        'gradient': gradient_config,
        'newton': newton_config,
    }

    if optimizer_config is None:
        resolved_optimizer_config = copy.deepcopy(default_config_by_method[normalized_optimizer_method])
        explicit_optimizer_config = False
    else:
        expected_optimizer_type = optimizer_type_by_method[normalized_optimizer_method]
        if not isinstance(optimizer_config, expected_optimizer_type):
            raise TypeError(
                f"optimizer_config for optimizer_method='{normalized_optimizer_method}' "
                f"must be of type {expected_optimizer_type.__name__}."
            )
        resolved_optimizer_config = copy.deepcopy(optimizer_config)

    optimizer_field_names = {field.name for field in fields(type(resolved_optimizer_config))}
    for key in list(legacy_kwargs.keys()):
        if key in optimizer_field_names:
            if explicit_optimizer_config:
                raise ValueError(
                    f"Legacy optimizer kwarg '{key}' cannot be used with optimizer_config."
                )
            setattr(resolved_optimizer_config, key, legacy_kwargs.pop(key))

    return normalized_optimizer_method, resolved_optimizer_config


def _resolve_line_search_config(line_search_method: str,
                                line_search_config,
                                legacy_kwargs: dict,
                                default_legacy_config: VILegacyLineSearchConfig = None,
                                default_stochastic_config: VIStochasticNonmonotoneLineSearchConfig = None):
    normalized_line_search_method = _normalize_line_search_method(line_search_method)
    legacy_config = (
        copy.deepcopy(default_legacy_config)
        if default_legacy_config is not None else VILegacyLineSearchConfig()
    )
    stochastic_config = (
        copy.deepcopy(default_stochastic_config)
        if default_stochastic_config is not None else VIStochasticNonmonotoneLineSearchConfig()
    )
    line_search_type_by_method = {
        'legacy': VILegacyLineSearchConfig,
        'stochastic_nonmonotone': VIStochasticNonmonotoneLineSearchConfig,
    }
    default_config_by_method = {
        'legacy': legacy_config,
        'stochastic_nonmonotone': stochastic_config,
    }
    explicit_line_search_config = line_search_config is not None
    if line_search_config is None:
        resolved_line_search_config = copy.deepcopy(default_config_by_method[normalized_line_search_method])
        explicit_line_search_config = False
    else:
        expected_line_search_type = line_search_type_by_method[normalized_line_search_method]
        if not isinstance(line_search_config, expected_line_search_type):
            raise TypeError(
                f"line_search_config for line_search_method='{normalized_line_search_method}' "
                f"must be of type {expected_line_search_type.__name__}."
            )
        resolved_line_search_config = copy.deepcopy(line_search_config)

    line_search_field_names = {field.name for field in fields(type(resolved_line_search_config))}
    for key in list(legacy_kwargs.keys()):
        if key in line_search_field_names:
            if explicit_line_search_config:
                raise ValueError(
                    f"Legacy line-search kwarg '{key}' cannot be used with line_search_config."
                )
            setattr(resolved_line_search_config, key, legacy_kwargs.pop(key))
    return normalized_line_search_method, resolved_line_search_config


def _raise_for_unused_legacy_kwargs(legacy_kwargs: dict, function_name: str) -> None:
    if not legacy_kwargs:
        return
    unknown_kwargs = ", ".join(sorted(legacy_kwargs.keys()))
    raise TypeError(f"{function_name}() got unexpected keyword arguments: {unknown_kwargs}")


class SteepestDescentSolver:
    """Generic steepest-descent solver."""

    def step(self, gradient: np.ndarray) -> np.ndarray:
        return np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)


class NewtonSolver:
    """Generic Newton solver supporting diagonal or dense Hessians."""

    def __init__(self, regularization: float):
        self.regularization = regularization

    def _project_hessian(self, hessian) -> np.ndarray:
        hessian = np.nan_to_num(hessian, nan=0.0, posinf=0.0, neginf=0.0)
        if hessian.ndim == 1:
            projected_diagonal = np.maximum(np.abs(hessian), self.regularization)
            return np.diag(projected_diagonal)
        if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
            raise ValueError("Hessian must be a 1D diagonal or a square 2D matrix.")
        sym_hessian = 0.5 * (hessian + hessian.T)
        eigenvalues, eigenvectors = np.linalg.eigh(sym_hessian)
        projected_eigenvalues = np.maximum(np.abs(eigenvalues), self.regularization)
        return (eigenvectors @ np.diag(projected_eigenvalues)) @ eigenvectors.T

    def step(self,
             gradient: np.ndarray,
             hessian) -> np.ndarray:
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        projected_hessian = self._project_hessian(hessian)
        return np.linalg.solve(projected_hessian, gradient)
