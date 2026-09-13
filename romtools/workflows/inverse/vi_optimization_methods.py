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
class VIAdamOptimizerConfig:
    """Configuration for Adam-based stochastic ELBO maximization.

    By default Adam is applied to the natural gradient. For the Gaussian
    variational parameterization used by the VI drivers this corresponds to
    preconditioning the mean gradient by ``std**2`` and the log-standard-
    deviation gradient by ``1/2``. When ``learning_rate`` is ``None``, the
    ABRIS convention ``0.1 / parameter_dimension`` is used.
    """

    gradient_method: str = 'natural'
    learning_rate: float = None
    learning_rate_scale: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
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
    max_mean_update_std: float = None
    min_variational_std: float = 1e-8
    max_variational_std: float = 1e6
    newton_metric: str = 'standard'
    newton_regularization: float = 1e-2
    newton_hessian_type: str = 'diagonal'
    newton_curvature_strategy: str = 'same_sample'
    newton_hessian_num_samples: int = None
    newton_hessian_averaging_factor: float = 0.9


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
    if method not in ('gradient', 'adam', 'newton'):
        raise ValueError(
            f"Unsupported optimization_method '{optimization_method}'. "
            "Supported options are 'gradient', 'adam', and 'newton'."
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


def _normalize_newton_curvature_strategy(strategy: str) -> str:
    """Normalize the stochastic Newton curvature sampling strategy."""
    normalized = strategy.strip().lower()
    if normalized in ('same_sample', 'independent', 'lagged'):
        return normalized
    raise ValueError(
        f"Unsupported newton_curvature_strategy '{strategy}'. "
        "Supported options are 'same_sample', 'independent', and 'lagged'."
    )


def _resolve_optimizer_config(optimizer_method: str,
                              optimizer_config,
                              default_gradient_config: VIGradientOptimizerConfig = None,
                              default_newton_config: VINewtonOptimizerConfig = None,
                              default_adam_config: VIAdamOptimizerConfig = None):
    normalized_optimizer_method = _normalize_optimization_method(optimizer_method)
    gradient_config = copy.deepcopy(default_gradient_config) if default_gradient_config is not None else (
        VIGradientOptimizerConfig()
    )
    newton_config = copy.deepcopy(default_newton_config) if default_newton_config is not None else (
        VINewtonOptimizerConfig()
    )
    adam_config = copy.deepcopy(default_adam_config) if default_adam_config is not None else (
        VIAdamOptimizerConfig()
    )

    optimizer_type_by_method = {
        'gradient': VIGradientOptimizerConfig,
        'adam': VIAdamOptimizerConfig,
        'newton': VINewtonOptimizerConfig,
    }
    default_config_by_method = {
        'gradient': gradient_config,
        'adam': adam_config,
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


class AdamSolver:
    """Stateful Adam ascent direction for stochastic VI gradients."""

    def __init__(self,
                 learning_rate: float = None,
                 learning_rate_scale: float = 0.1,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 parameter_dimension: int = None):
        if learning_rate is not None and learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive when provided.")
        if learning_rate_scale <= 0.0:
            raise ValueError("learning_rate_scale must be positive.")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1).")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1).")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if parameter_dimension is not None and parameter_dimension < 1:
            raise ValueError("parameter_dimension must be positive when provided.")

        self.learning_rate = learning_rate
        self.learning_rate_scale = learning_rate_scale
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.parameter_dimension = parameter_dimension
        self.first_moment = None
        self.second_moment = None
        self.iteration = 0

    @classmethod
    def from_config(cls, config: VIAdamOptimizerConfig):
        return cls(
            learning_rate=config.learning_rate,
            learning_rate_scale=config.learning_rate_scale,
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
        )

    def _resolved_learning_rate(self, gradient: np.ndarray) -> float:
        if self.learning_rate is not None:
            return float(self.learning_rate)
        parameter_dimension = self.parameter_dimension
        if parameter_dimension is None:
            if gradient.size % 2 != 0:
                raise ValueError(
                    "Cannot infer the VI parameter dimension from an odd-sized Adam gradient."
                )
            parameter_dimension = gradient.size // 2
        return self.learning_rate_scale / parameter_dimension

    def step(self, gradient: np.ndarray) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=float)
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        if self.first_moment is None:
            self.first_moment = np.zeros_like(gradient)
            self.second_moment = np.zeros_like(gradient)
        elif self.first_moment.shape != gradient.shape:
            raise ValueError("Adam gradient shape changed after optimizer initialization.")

        self.iteration += 1
        self.first_moment = (
            self.beta1 * self.first_moment + (1.0 - self.beta1) * gradient
        )
        self.second_moment = (
            self.beta2 * self.second_moment + (1.0 - self.beta2) * gradient ** 2
        )
        first_moment_hat = self.first_moment / (1.0 - self.beta1 ** self.iteration)
        second_moment_hat = self.second_moment / (1.0 - self.beta2 ** self.iteration)
        learning_rate = self._resolved_learning_rate(gradient)
        return learning_rate * first_moment_hat / (
            np.sqrt(second_moment_hat) + self.epsilon
        )

    def state_dict(self) -> dict:
        return {
            'iteration': self.iteration,
            'first_moment': None if self.first_moment is None else self.first_moment.copy(),
            'second_moment': None if self.second_moment is None else self.second_moment.copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.iteration = int(state['iteration'])
        first_moment = state.get('first_moment')
        second_moment = state.get('second_moment')
        self.first_moment = None if first_moment is None else np.asarray(first_moment).copy()
        self.second_moment = None if second_moment is None else np.asarray(second_moment).copy()
        if (self.first_moment is None) != (self.second_moment is None):
            raise ValueError("Adam first and second moments must either both be set or both be None.")
        if self.first_moment is not None and self.first_moment.shape != self.second_moment.shape:
            raise ValueError("Adam first and second moments must have matching shapes.")


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
        hessian = np.asarray(hessian)
        if self.hessian_type == 'diagonal':
            if hessian.ndim == 2:
                if hessian.shape[0] != hessian.shape[1]:
                    raise ValueError("Hessian must be a 1D diagonal or a square 2D matrix.")
                hessian = np.diag(hessian)
            elif hessian.ndim != 1:
                raise ValueError("Hessian must be a 1D diagonal or a square 2D matrix.")
            projected_diagonal = np.maximum(
                np.abs(np.nan_to_num(hessian, nan=0.0, posinf=0.0, neginf=0.0)),
                self.regularization,
            )
            print(
                'Gradient and hessian norms:',
                np.linalg.norm(gradient),
                np.linalg.norm(projected_diagonal),
            )
            return gradient / projected_diagonal

        projected_hessian = self._project_hessian(hessian)
        print(
            'Gradient and hessian norms:',
            np.linalg.norm(gradient),
            np.linalg.norm(projected_hessian),
        )
        return np.linalg.solve(projected_hessian, gradient)
