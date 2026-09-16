"""Auto-ROM convenience wrapper for full-covariance MF-VI."""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np

from romtools.hpc.dispatchers import BaseDispatcher
from romtools.workflows.models import QoiModel
from romtools.workflows.inverse.mf_eki_drivers import (
    GaussianProcessQoiModelBuilderWithTrainingData,
)
from romtools.workflows.inverse.full_covariance_mf_vi_drivers import run_mf_vi


def mf_vi_with_auto_rom(
    model: QoiModel,
    prior_parameter_space,
    observations: np.ndarray,
    observations_covariance: np.ndarray,
    parameter_mins: np.ndarray = None,
    parameter_maxes: np.ndarray = None,
    initial_variational_parameter_space=None,
    restart_file: str = None,
    optimizer_method: str = "gradient",
    optimizer_config=None,
    line_search_method: str = "stochastic_nonmonotone",
    line_search_config=None,
    absolute_work_dir: str = None,
    fom_sample_size: int = 10,
    rom_extra_sample_size: int = 30,
    rom_tolerance: float = 0.005,
    max_rom_training_history: int = 1,
    random_seed: int = 1,
    sampling_method: str = "mc",
    fom_evaluation_concurrency=1,
    rom_evaluation_concurrency=1,
    covariance_regularization: float = 1e-8,
    restart_files_to_keep: int = 10,
    correlation_estimator: str = "in_sample",
    correlation_k_folds: int = 5,
    elbo_scaling_factor="diag_mean",
    elbo_relative_tolerance: float = None,
    baseline_method: str = None,
    use_mfmc_control_variate: bool = True,
    mfmc_control_variate_mode: str = "componentwise",
    rom_base_sampling_strategy: str = "coupled",
    bounded_parameter_handling: str = "transform",
    transform_interior_margin: float = 1e-8,
    transform_map: str = "sigmoid",
    min_physical_variational_std_fraction: float = 1e-8,
    rom_type: str = "gp",
    rom_args: Optional[dict] = None,
    dispatcher: Optional[BaseDispatcher] = None,
    *,
    absolute_vi_directory: str = None,
    score_function_entropy_strategy: str = "analytic",
    variational_distribution: str = "full_covariance",
    max_covariance_log_step: float = 1.0,
    create_run_directories: bool = True,
    sample_reuse_config=None,
):
    """Run full-covariance MF-VI with a built-in adaptive surrogate builder."""
    if str(variational_distribution).strip().lower().replace("-", "_") not in (
        "full",
        "full_covariance",
        "multivariate",
    ):
        raise ValueError(
            "This wrapper is only for variational_distribution='full_covariance'."
        )
    if sample_reuse_config is not None:
        raise NotImplementedError(
            "Sample reuse is not yet defined for the full-covariance variational family."
        )
    if absolute_vi_directory is not None:
        warnings.warn(
            "'absolute_vi_directory' is deprecated; use 'absolute_work_dir' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if absolute_work_dir is not None:
            raise TypeError(
                "Specify only 'absolute_work_dir', not both directory arguments."
            )
        absolute_work_dir = absolute_vi_directory
    if absolute_work_dir is None:
        absolute_work_dir = os.getcwd() + "/work/"

    rom_args = {} if rom_args is None else dict(rom_args)
    rom_type_normalized = rom_type.strip().lower()
    if rom_type_normalized != "gp":
        raise ValueError(f"Unsupported rom_type '{rom_type}'.")
    parameter_names = list(prior_parameter_space.get_names())
    rom_model_builder = GaussianProcessQoiModelBuilderWithTrainingData(
        parameter_names=parameter_names,
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
    return run_mf_vi(
        model=model,
        rom_model_builder=rom_model_builder,
        prior_parameter_space=prior_parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=parameter_mins,
        parameter_maxes=parameter_maxes,
        initial_variational_parameter_space=initial_variational_parameter_space,
        restart_file=restart_file,
        optimizer_method=optimizer_method,
        optimizer_config=optimizer_config,
        line_search_method=line_search_method,
        line_search_config=line_search_config,
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
        correlation_estimator=correlation_estimator,
        correlation_k_folds=correlation_k_folds,
        elbo_scaling_factor=elbo_scaling_factor,
        elbo_relative_tolerance=elbo_relative_tolerance,
        baseline_method=baseline_method,
        use_mfmc_control_variate=use_mfmc_control_variate,
        mfmc_control_variate_mode=mfmc_control_variate_mode,
        rom_base_sampling_strategy=rom_base_sampling_strategy,
        bounded_parameter_handling=bounded_parameter_handling,
        transform_interior_margin=transform_interior_margin,
        transform_map=transform_map,
        min_physical_variational_std_fraction=min_physical_variational_std_fraction,
        dispatcher=dispatcher,
        score_function_entropy_strategy=score_function_entropy_strategy,
        variational_distribution="full_covariance",
        max_covariance_log_step=max_covariance_log_step,
        create_run_directories=create_run_directories,
        sample_reuse_config=None,
    )
