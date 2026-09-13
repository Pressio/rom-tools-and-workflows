"""Adam integration for the VI and MF-VI workflow drivers.

The existing VI drivers share a line-search based gradient path.  Adam does
not need that line search, so this module adapts the public ``run_vi`` and
``run_mf_vi`` entry points to reuse the existing state-evaluation machinery
while replacing the steepest-descent direction with a stateful Adam direction.
"""

from contextvars import ContextVar
import functools
import inspect
import warnings

import numpy as np

from romtools.workflows.inverse.vi_optimization_methods import (
    AdamSolver,
    SteepestDescentSolver,
    VIAdamOptimizerConfig,
    VIGradientOptimizerConfig,
    VILegacyLineSearchConfig,
)


_adam_config_context = ContextVar("romtools_vi_adam_config", default=None)


class _GradientSolverDispatch:
    """Construct either the standard gradient solver or the active Adam solver."""

    def __init__(self):
        adam_config = _adam_config_context.get()
        if adam_config is None:
            self._solver = SteepestDescentSolver()
        else:
            self._solver = AdamSolver.from_config(adam_config)

    def step(self, gradient: np.ndarray) -> np.ndarray:
        return self._solver.step(gradient)


def _adam_line_search_config() -> VILegacyLineSearchConfig:
    """Return a one-shot acceptance configuration for Adam updates.

    Adam supplies the complete learning-rate-scaled direction.  The legacy
    gradient path still evaluates the proposed state, which is useful because
    that evaluation becomes the state for the next optimizer iteration.  The
    line-search multiplier is therefore fixed at one and retries are disabled.
    """
    return VILegacyLineSearchConfig(
        initial_step_size=1.0,
        max_step_size=1.0,
        step_size_growth_factor=1.0,
        step_size_decay_factor=1.0,
        max_step_size_decrease_trys=0,
        relaxation_parameter=1e300,
        line_search_objective='elbo',
        line_search_sample_growth_factor=1.0,
        log_std_learning_rate_factor=1.0,
    )


def _validate_adam_config(config: VIAdamOptimizerConfig) -> VIAdamOptimizerConfig:
    if not isinstance(config, VIAdamOptimizerConfig):
        raise TypeError(
            "optimizer_config for optimizer_method='adam' must be of type "
            "VIAdamOptimizerConfig."
        )
    gradient_method = config.gradient_method.strip().lower()
    if gradient_method not in ('standard', 'natural'):
        raise ValueError(
            f"Unsupported Adam gradient_method '{config.gradient_method}'. "
            "Supported options are 'standard' and 'natural'."
        )
    # Construction validates the Adam-specific hyperparameters without
    # advancing optimizer state.
    AdamSolver.from_config(config)
    return config


def _gradient_config_from_adam(config: VIAdamOptimizerConfig) -> VIGradientOptimizerConfig:
    return VIGradientOptimizerConfig(
        gradient_method=config.gradient_method,
        gradient_norm_tolerance=config.gradient_norm_tolerance,
        max_iterations=config.max_iterations,
        max_log_std_update=config.max_log_std_update,
        min_variational_std=config.min_variational_std,
        max_variational_std=config.max_variational_std,
    )


def _make_adam_aware_driver(original_driver):
    signature = inspect.signature(original_driver)

    @functools.wraps(original_driver)
    def wrapped(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        optimizer_method = bound.arguments.get(
            'optimizer_method',
            signature.parameters['optimizer_method'].default,
        )
        if not isinstance(optimizer_method, str) or optimizer_method.strip().lower() != 'adam':
            return original_driver(*args, **kwargs)

        adam_config = bound.arguments.get('optimizer_config')
        if adam_config is None:
            adam_config = VIAdamOptimizerConfig()
        adam_config = _validate_adam_config(adam_config)

        if bound.arguments.get('restart_file') is not None:
            warnings.warn(
                "Adam optimizer moments are not currently stored in VI restart files; "
                "restarting an Adam VI run resets the Adam first- and second-moment state.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Reuse the mature gradient workflow for sampling, ELBO evaluation,
        # bounds handling, logging, and restart output.  The context-local
        # solver dispatch replaces only the gradient update rule.
        bound.arguments['optimizer_method'] = 'gradient'
        bound.arguments['optimizer_config'] = _gradient_config_from_adam(adam_config)
        bound.arguments['line_search_method'] = 'legacy'
        bound.arguments['line_search_config'] = _adam_line_search_config()

        token = _adam_config_context.set(adam_config)
        try:
            return original_driver(*bound.args, **bound.kwargs)
        finally:
            _adam_config_context.reset(token)

    wrapped._romtools_adam_aware = True
    return wrapped


def install_adam_optimizer_support(vi_drivers_module, mf_vi_drivers_module) -> None:
    """Install Adam dispatch on the VI and MF-VI public driver functions."""
    vi_drivers_module.SteepestDescentSolver = _GradientSolverDispatch
    mf_vi_drivers_module.SteepestDescentSolver = _GradientSolverDispatch

    if not getattr(vi_drivers_module.run_vi, '_romtools_adam_aware', False):
        vi_drivers_module.run_vi = _make_adam_aware_driver(vi_drivers_module.run_vi)
    if not getattr(mf_vi_drivers_module.run_mf_vi, '_romtools_adam_aware', False):
        mf_vi_drivers_module.run_mf_vi = _make_adam_aware_driver(mf_vi_drivers_module.run_mf_vi)
