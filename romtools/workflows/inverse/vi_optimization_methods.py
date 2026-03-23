"""Optimization methods and configs used by inverse VI workflows."""

import copy
from dataclasses import dataclass

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
    gradient_norm_tolerance: float = 5e-5
    max_iterations: int = 1000
    max_log_std_update: float = 0.5
    min_variational_std: float = 1e-8
    max_variational_std: float = 1e6
    newton_metric: str = 'standard'
    newton_regularization: float = 1e-2
    newton_hessian_type: str = 'diagonal'


@dataclass
class VIAdaGradOptimizerConfig:
    gradient_method: str = 'standard'
    gradient_norm_tolerance: float = 1e-5
    max_iterations: int = 1000
    max_log_std_update: float = 0.5
    min_variational_std: float = 1e-6
    max_variational_std: float = 1e6
    adagrad_epsilon: float = 1e-8
    initial_accumulator_value: float = 0.0


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
    log_std_learning_rate_factor: float = 0.1


@dataclass
class VIStochasticNonmonotoneLineSearchConfig:
    initial_step_size: float = 1e-2
    max_step_size: float = np.inf
    step_size_growth_factor: float = 1.05
    step_size_decay_factor: float = 2.0
    max_step_size_decrease_trys: int = 5
    relaxation_parameter: float = 3.05
    line_search_objective: str = 'elbo'
    line_search_nonmonotone_window: int = 5
    line_search_armijo_coefficient: float = 1e-6
    line_search_uncertainty_sigma: float = 4.0
    line_search_sample_growth_factor: float = 1.0
    log_std_learning_rate_factor: float = 1.0


def _normalize_optimization_method(optimization_method: str):
    method = optimization_method.strip().lower()
    if method not in ('gradient', 'adagrad', 'newton'):
        raise ValueError(
            f"Unsupported optimization_method '{optimization_method}'. "
            "Supported options are 'gradient', 'adagrad', and 'newton'."
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


def _normalize_newton_hessian_type(newton_hessian_type: str):
    hessian_type = newton_hessian_type.strip().lower()
    if hessian_type in ('diagonal', 'diag'):
        return 'diagonal'
    if hessian_type in ('full', 'dense'):
        return 'full'
    raise ValueError(
        f"Unsupported newton_hessian_type '{newton_hessian_type}'. "
        "Supported options are 'diagonal' and 'full'."
    )


def _resolve_optimizer_config(optimizer_method: str,
                              optimizer_config,
                              default_gradient_config: VIGradientOptimizerConfig = None,
                              default_adagrad_config: VIAdaGradOptimizerConfig = None,
                              default_newton_config: VINewtonOptimizerConfig = None):
    normalized_optimizer_method = _normalize_optimization_method(optimizer_method)
    gradient_config = copy.deepcopy(default_gradient_config) if default_gradient_config is not None else (
        VIGradientOptimizerConfig()
    )
    adagrad_config = copy.deepcopy(default_adagrad_config) if default_adagrad_config is not None else (
        VIAdaGradOptimizerConfig()
    )
    newton_config = copy.deepcopy(default_newton_config) if default_newton_config is not None else (
        VINewtonOptimizerConfig()
    )

    optimizer_type_by_method = {
        'gradient': VIGradientOptimizerConfig,
        'adagrad': VIAdaGradOptimizerConfig,
        'newton': VINewtonOptimizerConfig,
    }
    default_config_by_method = {
        'gradient': gradient_config,
        'adagrad': adagrad_config,
        'newton': newton_config,
    }

    if optimizer_config is None:
        resolved_optimizer_config = copy.deepcopy(default_config_by_method[normalized_optimizer_method])
    else:
        expected_optimizer_type = optimizer_type_by_method[normalized_optimizer_method]
        if not isinstance(optimizer_config, expected_optimizer_type):
            raise TypeError(
                f"optimizer_config for optimizer_method='{normalized_optimizer_method}' "
                f"must be of type {expected_optimizer_type.__name__}."
            )
        resolved_optimizer_config = copy.deepcopy(optimizer_config)

    return normalized_optimizer_method, resolved_optimizer_config


def _resolve_line_search_config(line_search_method: str,
                                line_search_config,
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

    return normalized_line_search_method, resolved_line_search_config


class SteepestDescentSolver:
    """Generic steepest-descent solver."""

    def step(self, gradient: np.ndarray) -> np.ndarray:
        return np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)


class AdaGradSolver:
    """Generic AdaGrad solver."""

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def step(self, gradient: np.ndarray, accumulator: np.ndarray) -> np.ndarray:
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        accumulator = np.nan_to_num(accumulator, nan=0.0, posinf=0.0, neginf=0.0)
        return gradient / np.sqrt(accumulator + self.epsilon)


class NewtonSolver:
    """Generic Newton solver supporting diagonal or dense Hessians."""

    def __init__(self, regularization: float, hessian_type: str = 'diagonal'):
        self.regularization = regularization
        self.hessian_type = _normalize_newton_hessian_type(hessian_type)

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
        print('Gradient and hessian norms:', np.linalg.norm(gradient),np.linalg.norm(projected_hessian))
        if self.hessian_type == 'full':
            return np.linalg.solve(projected_hessian, gradient)
        return np.linalg.solve(np.diag(np.diag(projected_hessian)), gradient)
