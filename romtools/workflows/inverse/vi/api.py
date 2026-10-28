"""Public entry points for diagonal and full-covariance variational inference."""

import inspect
import os
import sys
import warnings
from typing import Optional

import numpy as np

from romtools.hpc.dispatchers import BaseDispatcher
from romtools.workflows.models import QoiModel
from romtools.workflows.inverse.mf_eki_drivers import GaussianProcessQoiModelBuilderWithTrainingData
from . import drivers, mf_drivers, full_covariance_drivers, full_covariance_mf_drivers
from .evaluation import _UNSET, _run_directory_policy
from .newton import FullCovarianceNewton, _gradient_config_from_newton
from .optimization import (
    VINewtonOptimizerConfig, VIGradientOptimizerConfig, VILegacyLineSearchConfig,
    VIStochasticNonmonotoneLineSearchConfig, _resolve_optimizer_config,
    _resolve_line_search_config,
)
from .sample_reuse import VISampleReuseConfig, make_controller

def _bound_arguments(function, args, kwargs):
    return inspect.signature(function).bind_partial(*args, **kwargs).arguments


def _replace_argument(function, args, kwargs, name, value):
    """Replace a positional-or-keyword argument without creating duplicates."""
    signature = inspect.signature(function)
    parameter_names = list(signature.parameters)
    parameter_index = parameter_names.index(name)
    if parameter_index < len(args):
        args = list(args)
        args[parameter_index] = value
        return tuple(args), kwargs
    kwargs = dict(kwargs)
    kwargs[name] = value
    return args, kwargs


def _ensure_default_newton_config(function, args, kwargs):
    """Materialize the shared Newton defaults before legacy driver resolution."""
    arguments = _bound_arguments(function, args, kwargs)
    method = str(arguments.get("optimizer_method", "gradient")).strip().lower()
    if method != "newton" or arguments.get("optimizer_config") is not None:
        return args, kwargs
    return _replace_argument(
        function,
        args,
        kwargs,
        "optimizer_config",
        VINewtonOptimizerConfig(),
    )


def _ensure_full_rom_training_history(function, args, kwargs):
    """Use all accumulated FOM training data unless a history limit is supplied."""
    arguments = _bound_arguments(function, args, kwargs)
    if (
        "max_rom_training_history" in arguments
        and arguments["max_rom_training_history"] is not None
    ):
        return args, kwargs

    signature = inspect.signature(function)
    fom_sample_size = arguments.get(
        "fom_sample_size",
        signature.parameters["fom_sample_size"].default,
    )
    fom_sample_size = max(int(fom_sample_size), 1)
    full_history = max(1, sys.maxsize // fom_sample_size)
    return _replace_argument(
        function,
        args,
        kwargs,
        "max_rom_training_history",
        full_history,
    )


def _optimizer_method(function, args, kwargs) -> str:
    """Return the requested optimizer method using the public driver's defaults."""
    arguments = _bound_arguments(function, args, kwargs)
    return str(arguments.get("optimizer_method", "gradient")).strip().lower()


def _uses_natural_newton(function, args, kwargs) -> bool:
    """Return whether the request is specifically full-covariance natural Newton."""
    arguments = _bound_arguments(function, args, kwargs)
    if str(arguments.get("optimizer_method", "gradient")).strip().lower() != "newton":
        return False
    config = arguments.get("optimizer_config")
    if config is None:
        config = VINewtonOptimizerConfig()
    return str(getattr(config, "newton_metric", "natural")).strip().lower() == "natural"


def _validate_no_line_search(function, args, kwargs):
    """Validate before full-covariance Newton is mapped to a gradient driver."""
    arguments = _bound_arguments(function, args, kwargs)
    method = str(arguments.get("line_search_method", "stochastic_nonmonotone")).strip().lower()
    if method == "none" and _optimizer_method(function, args, kwargs) != "newton":
        raise ValueError("line_search_method='none' requires optimizer_method='newton'")


def _reject_mean_field_only_newton_strategy(function, args, kwargs) -> None:
    """Reject Newton options that are not defined for full covariance."""
    arguments = _bound_arguments(function, args, kwargs)
    if str(arguments.get("optimizer_method", "gradient")).strip().lower() != "newton":
        return
    config = arguments.get("optimizer_config")
    if config is None:
        return
    strategy = str(config.newton_regularization_strategy).strip().lower()
    if strategy == "per_parameter":
        raise NotImplementedError(
            "newton_regularization_strategy='per_parameter' is supported only "
            "for mean-field Gaussian VI/MF-VI."
        )
    if config.newton_additive_regularization_max is not None:
        raise NotImplementedError(
            "An additive Newton damping maximum is supported only for mean-field Gaussian VI/MF-VI."
        )
    if config.newton_additive_regularization_growth_factor != 1.0:
        raise NotImplementedError(
            "Additive Newton damping growth is supported only for mean-field Gaussian VI/MF-VI."
        )
    if config.newton_adaptive_regularization:
        raise NotImplementedError(
            "Adaptive Newton regularization is supported only for mean-field "
            "Gaussian VI/MF-VI."
        )


def _variational_family(initial_variational_parameter_space) -> str:
    """Infer the variational family exclusively from the initializer type."""
    _, _, _, distribution = drivers._extract_gaussian_parameter_space(
        initial_variational_parameter_space,
        argument_name="initial_variational_parameter_space",
    )
    if distribution == "diagonal":
        return "diagonal"
    if distribution == "multivariate":
        return "full_covariance"
    raise ValueError(f"Unsupported variational Gaussian family '{distribution}'.")


def _run(args, kwargs, *, multifidelity, max_covariance_log_step,
         create_run_directories, sample_reuse_config):
    diagonal = mf_drivers.run_mf_vi if multifidelity else drivers.run_vi
    full = full_covariance_mf_drivers.run_mf_vi if multifidelity else full_covariance_drivers.run_vi
    _validate_no_line_search(diagonal, args, kwargs)
    args, kwargs = _ensure_default_newton_config(diagonal, args, kwargs)
    if multifidelity:
        args, kwargs = _ensure_full_rom_training_history(diagonal, args, kwargs)
    bound = inspect.signature(diagonal).bind(*args, **kwargs)
    initializer = bound.arguments.get("initial_variational_parameter_space")
    if initializer is None or _variational_family(initializer) == "diagonal":
        bound.apply_defaults()
        controller = make_controller(sample_reuse_config, bound.arguments,
                                     multifidelity=multifidelity)
        with _run_directory_policy(create_run_directories):
            return diagonal(*args, _reuse_controller=controller, **kwargs)
    _reject_mean_field_only_newton_strategy(diagonal, args, kwargs)
    call = dict(bound.arguments)
    if _uses_natural_newton(diagonal, args, kwargs):
        newton = FullCovarianceNewton(call.get("optimizer_config"), multifidelity=multifidelity)
        call["optimizer_method"] = "gradient"
        call["optimizer_config"] = _gradient_config_from_newton(newton.config)
        call["_newton"] = newton
        if multifidelity:
            call["rom_substep_newton_config"] = newton.config
    return full(**call, max_covariance_log_step=max_covariance_log_step,
                create_run_directories=create_run_directories,
                sample_reuse_config=sample_reuse_config)


def run_vi(*args, max_covariance_log_step=1.0, create_run_directories=_UNSET,
           sample_reuse_config=None, **kwargs):
    """Run Gaussian VI; the initializer determines the variational family."""
    return _run(args, kwargs, multifidelity=False,
                max_covariance_log_step=max_covariance_log_step,
                create_run_directories=create_run_directories,
                sample_reuse_config=sample_reuse_config)


def run_mf_vi(*args, max_covariance_log_step=1.0, create_run_directories=_UNSET,
              sample_reuse_config=None, **kwargs):
    """Run Gaussian multifidelity VI with an adaptive surrogate builder."""
    return _run(args, kwargs, multifidelity=True,
                max_covariance_log_step=max_covariance_log_step,
                create_run_directories=create_run_directories,
                sample_reuse_config=sample_reuse_config)


def mf_vi_with_auto_rom(model: QoiModel,
                        prior_parameter_space,
                        observations: np.ndarray,
                        observations_covariance: np.ndarray,
                        parameter_mins: np.ndarray = None,
                        parameter_maxes: np.ndarray = None,
                        initial_variational_parameter_space=None,
                        restart_file: str = None,
                        optimizer_method: str = 'gradient',
                        optimizer_config=None,
                        line_search_method: str = 'stochastic_nonmonotone',
                        line_search_config=None,
                        absolute_work_dir: str = None,
                        fom_sample_size: int = 10,
                        rom_extra_sample_size: int = 30,
                        rom_tolerance: float = 0.005,
                        max_rom_training_history: int = None,
                        random_seed: int = 1,
                        sampling_method: str = 'mc',
                        fom_evaluation_concurrency=1,
                        rom_evaluation_concurrency=1,
                        covariance_regularization: float = 1e-8,
                        restart_files_to_keep: int = 10,
                        rom_retraining_strategy: str = 'delayed',
                        gradient_gain_estimation_strategy: str = 'same_sample',
                        elbo_scaling_factor='diag_mean',
                        elbo_relative_tolerance: float = None,
                        baseline_method: str = None,
                        use_mfmc_control_variate: bool = True,
                        mfmc_control_variate_mode: str = 'componentwise',
                        bounded_parameter_handling: str = 'transform',
                        transform_interior_margin: float = 1e-8,
                        transform_map: str = 'sigmoid',
                        min_physical_variational_std_fraction: float = 1e-8,
                        rom_type: str = "gp",
                        rom_args: Optional[dict] = None,
                        dispatcher: Optional[BaseDispatcher] = None,
                        *,
                        absolute_vi_directory: str = None,
                        score_function_entropy_strategy: str = 'analytic',
                        rom_substep_start_iteration: int = 0,
                        rom_substep_end_iteration: Optional[int] = None,
                        num_rom_substeps: int = 0,
                        gain_num_folds: int = 5,
                        hessian_gain_estimation_strategy: str = "same_sample",
                        gain_sample_size=None,
                        mfmc_fixed_gain: float = 1.0,
                        max_covariance_log_step: float = 1.0,
                        create_run_directories: bool = _UNSET,
                        sample_reuse_config=None):
    """
    Wrapper around run_mf_vi that selects a default ROM surrogate by rom_type.
    Accepts the same ROM-retraining and gain-estimation options as run_mf_vi.
    """
    with _run_directory_policy(create_run_directories):
        if absolute_vi_directory is not None:
            warnings.warn(
                "'absolute_vi_directory' is deprecated; use 'absolute_work_dir' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if absolute_work_dir is not None:
                raise TypeError("Specify only 'absolute_work_dir', not both directory arguments.")
            absolute_work_dir = absolute_vi_directory
        if absolute_work_dir is None:
            absolute_work_dir = os.getcwd() + "/work/"

        parameter_mins, parameter_maxes = drivers._resolve_parameter_bounds(
            parameter_mins,
            parameter_maxes,
        )
        restart_file = drivers._resolve_restart_file(restart_file)
        diagonal = (initial_variational_parameter_space is None
                    or _variational_family(initial_variational_parameter_space) == "diagonal")
        if optimizer_method.strip().lower() == "newton" and optimizer_config is None:
            optimizer_config = VINewtonOptimizerConfig()
        if diagonal:
            resolved_optimizer_method, resolved_optimizer_config = _resolve_optimizer_config(
                optimizer_method,
                optimizer_config,
                VIGradientOptimizerConfig(),
                VINewtonOptimizerConfig(newton_regularization=1e-8),
            )
            resolved_line_search_method, resolved_line_search_config = _resolve_line_search_config(
                line_search_method,
                line_search_config,
                VILegacyLineSearchConfig(
                    step_size_growth_factor=1.05,
                    max_step_size_decrease_trys=10,
                    relaxation_parameter=1000.0,
                ),
                VIStochasticNonmonotoneLineSearchConfig(
                    step_size_growth_factor=1.05,
                    max_step_size_decrease_trys=10,
                    relaxation_parameter=1000.0,
                    line_search_nonmonotone_window=8,
                    line_search_uncertainty_sigma=5.0,
                ),
            )

        else:
            resolved_optimizer_method = optimizer_method
            resolved_optimizer_config = optimizer_config
            resolved_line_search_method = line_search_method
            resolved_line_search_config = line_search_config

        rom_args = {} if rom_args is None else dict(rom_args)
        rom_type_normalized = rom_type.strip().lower()
        if rom_type_normalized == "gp":
            variational_parameter_names = list(prior_parameter_space.get_names())
            rom_model_builder = GaussianProcessQoiModelBuilderWithTrainingData(
                parameter_names=variational_parameter_names,
                pod_energy_fraction=rom_args.get("pod_energy_fraction", 0.9999),
                max_pod_modes=rom_args.get("max_pod_modes"),
                kernel=rom_args.get("kernel"),
                noise_variance=rom_args.get("noise_variance"),
                auto_noise_variance=rom_args.get("auto_noise_variance", False),
                noise_variance_fraction=rom_args.get("noise_variance_fraction", 1e-6),
                tune_hyperparameters=rom_args.get("tune_hyperparameters", False),
                length_scale_grid=rom_args.get("length_scale_grid"),
                signal_variance_grid=rom_args.get("signal_variance_grid"),
                normalize_parameters=rom_args.get("normalize_parameters", False),
                normalize_targets=rom_args.get("normalize_targets", False),
            )
        else:
            raise ValueError(f"Unsupported rom_type '{rom_type}'.")

        return run_mf_vi(
            max_covariance_log_step=max_covariance_log_step,
            create_run_directories=create_run_directories,
            sample_reuse_config=sample_reuse_config,
            gain_sample_size=gain_sample_size,
            model=model,
            rom_model_builder=rom_model_builder,
            prior_parameter_space=prior_parameter_space,
            observations=observations,
            observations_covariance=observations_covariance,
            parameter_mins=parameter_mins,
            parameter_maxes=parameter_maxes,
            initial_variational_parameter_space=initial_variational_parameter_space,
            restart_file=restart_file,
            optimizer_method=resolved_optimizer_method,
            optimizer_config=resolved_optimizer_config,
            line_search_method=resolved_line_search_method,
            line_search_config=(None if resolved_optimizer_method == 'adam' and line_search_config is None
                                else resolved_line_search_config),
            absolute_work_dir=absolute_work_dir,
            fom_sample_size=fom_sample_size,
            rom_extra_sample_size=rom_extra_sample_size,
            rom_tolerance=rom_tolerance,
            max_rom_training_history=max_rom_training_history,
            random_seed=random_seed,
            sampling_method=sampling_method,
            fom_evaluation_concurrency=fom_evaluation_concurrency,
            rom_evaluation_concurrency=rom_evaluation_concurrency,
            covariance_regularization=covariance_regularization,
            restart_files_to_keep=restart_files_to_keep,
            rom_retraining_strategy=rom_retraining_strategy,
            gradient_gain_estimation_strategy=gradient_gain_estimation_strategy,
            hessian_gain_estimation_strategy=hessian_gain_estimation_strategy,
            gain_num_folds=gain_num_folds,
            elbo_scaling_factor=elbo_scaling_factor,
            elbo_relative_tolerance=elbo_relative_tolerance,
            baseline_method=baseline_method,
            use_mfmc_control_variate=use_mfmc_control_variate,
            mfmc_control_variate_mode=mfmc_control_variate_mode,
            bounded_parameter_handling=bounded_parameter_handling,
            transform_interior_margin=transform_interior_margin,
            transform_map=transform_map,
            min_physical_variational_std_fraction=min_physical_variational_std_fraction,
            dispatcher=dispatcher,
            score_function_entropy_strategy=score_function_entropy_strategy,
            rom_substep_start_iteration=rom_substep_start_iteration,
            rom_substep_end_iteration=rom_substep_end_iteration,
            num_rom_substeps=num_rom_substeps,
            mfmc_fixed_gain=mfmc_fixed_gain,
        )


def _public_signature(wrapper, driver):
    """Expose the supported parameters while keeping implementation hooks private."""
    parameters = []
    for parameter in inspect.signature(driver).parameters.values():
        if parameter.name in ("initial_variational_parameter_space", "_reuse_controller"):
            continue
        if parameter.name == "max_rom_training_history":
            parameter = parameter.replace(default=None)
        parameters.append(parameter)
    index = next((i for i, p in enumerate(parameters)
                  if p.kind == inspect.Parameter.KEYWORD_ONLY), len(parameters))
    parameters.insert(index, inspect.Parameter("initial_variational_parameter_space",
                      kind=inspect.Parameter.KEYWORD_ONLY))
    parameters.insert(index + 1, inspect.Parameter("max_covariance_log_step",
                      kind=inspect.Parameter.KEYWORD_ONLY, default=1.0, annotation=float))
    for name, default, annotation in (
            ("create_run_directories", True, bool),
            ("sample_reuse_config", None, VISampleReuseConfig)):
        parameters.append(inspect.Parameter(name, kind=inspect.Parameter.KEYWORD_ONLY,
                                           default=default, annotation=annotation))
    wrapper.__signature__ = inspect.signature(driver).replace(parameters=parameters)


_public_signature(run_vi, drivers.run_vi)
_public_signature(run_mf_vi, mf_drivers.run_mf_vi)
# Auto-ROM has a concrete signature; make the required initializer explicit there too.
_auto_signature = inspect.signature(mf_vi_with_auto_rom)
_auto_parameters = [p.replace(default=True) if p.name == "create_run_directories" else p
                    for p in _auto_signature.parameters.values()
                    if p.name not in ("initial_variational_parameter_space", "max_covariance_log_step")]
_auto_index = next(i for i, p in enumerate(_auto_parameters)
                   if p.kind == inspect.Parameter.KEYWORD_ONLY)
_auto_parameters.insert(_auto_index, inspect.Parameter("initial_variational_parameter_space",
                        kind=inspect.Parameter.KEYWORD_ONLY))
_auto_parameters.insert(_auto_index + 1, _auto_signature.parameters["max_covariance_log_step"])
mf_vi_with_auto_rom.__signature__ = _auto_signature.replace(parameters=_auto_parameters)
