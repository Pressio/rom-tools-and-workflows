r"""
Multifidelity ensemble Kalman inversion drivers.

This module extends the single-fidelity EKI workflow with an adaptive
low-fidelity surrogate, typically a ROM, to reduce the cost of the inverse
solve while preserving high-fidelity correction steps. The implementation uses
one high-fidelity ensemble and an additional ROM-only ensemble so that the
Kalman update benefits from a larger effective sample set.

MF-EKI supports the same adaptive or periodic ensemble rejuvenation as
:func:`romtools.workflows.inverse.eki_drivers.run_eki`. When rejuvenation moves
the ensemble, the FOM is reevaluated on the shared high-fidelity sample set and
the existing ROM validity check is applied. If the ROM error exceeds
``rom_tolerance``, the surrogate is rebuilt before the multifidelity Kalman
update is recomputed.
"""

import copy
import os
import time
from typing import Optional

import numpy as np

from romtools.hpc.dispatchers import (
    BaseDispatcher,
    resolve_dispatcher,
    resolve_local_dispatcher,
)
from romtools.rom.qoi_surrogates import (
    GaussianProcessKernel,
    GaussianProcessQoiModel,
    NeuralNetworkConfig,
)
from romtools.rom.neural_network_surrogate import (
    LipschitzConfig,
    NeuralNetworkQoiModelBuilderWithTrainingData,
)
from romtools.workflows.inverse._inverse_utils import *
from romtools.workflows.inverse.eki_drivers import (
    _compute_ensemble_covariance,
    _compute_ensemble_spread,
    _rejuvenate_parameter_samples,
    _should_rejuvenate,
    _validate_rejuvenation_settings,
    compute_eki_update,
)
from romtools.workflows.model_builders import QoiModelBuilderWithTrainingData
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace


class GaussianProcessQoiModelBuilderWithTrainingData:
    def __init__(self,
                 parameter_names,
                 pod_energy_fraction: float = 0.999999,
                 max_pod_modes: Optional[int] = None,
                 kernel: Optional[GaussianProcessKernel] = None,
                 noise_variance: Optional[float] = None,
                 auto_noise_variance: bool = False,
                 noise_variance_fraction: float = 1e-6,
                 tune_hyperparameters: bool = False,
                 length_scale_grid: Optional[list] = None,
                 signal_variance_grid: Optional[list] = None,
                 normalize_parameters: bool = False,
                 normalize_targets: bool = False) -> None:
        self.parameter_names = (
            list(parameter_names) if parameter_names is not None else None
        )
        self.pod_energy_fraction = pod_energy_fraction
        self.max_pod_modes = max_pod_modes
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.auto_noise_variance = auto_noise_variance
        self.noise_variance_fraction = noise_variance_fraction
        self.tune_hyperparameters = tune_hyperparameters
        self.length_scale_grid = length_scale_grid
        self.signal_variance_grid = signal_variance_grid
        self.normalize_parameters = normalize_parameters
        self.normalize_targets = normalize_targets

    def build_from_training_dirs(self,
                                 offline_data_dir: str,
                                 training_data_dirs,
                                 training_parameters: np.ndarray,
                                 training_qois: np.ndarray) -> QoiModel:
        return GaussianProcessQoiModel(
            parameters=training_parameters,
            qois=training_qois,
            parameter_names=self.parameter_names,
            pod_energy_fraction=self.pod_energy_fraction,
            max_pod_modes=self.max_pod_modes,
            kernel=self.kernel,
            noise_variance=self.noise_variance,
            auto_noise_variance=self.auto_noise_variance,
            noise_variance_fraction=self.noise_variance_fraction,
            tune_hyperparameters=self.tune_hyperparameters,
            length_scale_grid=self.length_scale_grid,
            signal_variance_grid=self.signal_variance_grid,
            normalize_parameters=self.normalize_parameters,
            normalize_targets=self.normalize_targets,
        )


def _validate_rom_substep_settings(rom_substep_start_iteration,
                                   rom_substep_end_iteration,
                                   num_rom_substeps):
    if num_rom_substeps < 0:
        raise ValueError("num_rom_substeps must be nonnegative")
    if rom_substep_start_iteration < 0:
        raise ValueError("rom_substep_start_iteration must be nonnegative")
    if (rom_substep_end_iteration is not None
            and rom_substep_end_iteration < rom_substep_start_iteration):
        raise ValueError(
            "rom_substep_end_iteration must be greater than or equal to "
            "rom_substep_start_iteration"
        )


def _rom_substeps_are_enabled(outer_iteration,
                              rom_substep_start_iteration,
                              rom_substep_end_iteration,
                              num_rom_substeps):
    if num_rom_substeps == 0:
        return False
    if outer_iteration < rom_substep_start_iteration:
        return False
    return (rom_substep_end_iteration is None
            or outer_iteration < rom_substep_end_iteration)


def _apply_rom_only_substeps(rom_model,
                             observations,
                             observations_covariance,
                             parameter_sample_sets,
                             parameter_names,
                             step_size,
                             regularization_parameter,
                             parameter_mins,
                             parameter_maxes,
                             absolute_eki_directory,
                             outer_iteration,
                             num_rom_substeps,
                             rom_evaluation_concurrency,
                             rom_dispatcher):
    """Advance both MF-EKI sample sets with ROM-only EKI steps."""
    first_set_size = parameter_sample_sets[0].shape[0]
    combined_parameter_samples = np.vstack(parameter_sample_sets).copy()

    for substep in range(num_rom_substeps):
        run_directory_base = (
            f'{absolute_eki_directory}/iteration_{outer_iteration}/'
            f'rom_substep_{substep}/run_rom_'
        )
        rom_results = run_eki_iteration(
            rom_model,
            observations,
            run_directory_base,
            parameter_names,
            combined_parameter_samples,
            rom_evaluation_concurrency,
            rom_dispatcher,
        )
        dp = compute_eki_update(
            combined_parameter_samples,
            rom_results['qois'],
            rom_results['mean-qoi'],
            rom_results['errors'],
            observations_covariance,
            regularization_parameter,
        )
        combined_parameter_samples = bound_samples(
            combined_parameter_samples + step_size * dp,
            parameter_mins,
            parameter_maxes,
        )
        print(
            f'  ROM-only substep {substep + 1}/{num_rom_substeps} after outer '
            f'iteration {outer_iteration}, Delta p: {np.linalg.norm(dp):.5f}'
        )

    return [
        combined_parameter_samples[:first_set_size].copy(),
        combined_parameter_samples[first_set_size:].copy(),
    ]


def _relative_rom_error(rom_results, fom_results):
    denominator = np.linalg.norm(fom_results['qois'])
    if denominator == 0.0:
        return np.linalg.norm(rom_results['qois'] - fom_results['qois'])
    return (
        np.linalg.norm(rom_results['qois'] - fom_results['qois'])
        / denominator
    )


def _save_mf_eki_restart(
        dispatcher,
        restart_path,
        sample_one_rom_results,
        sample_two_rom_results,
        sample_one_fom_results,
        parameter_sample_sets,
        iteration,
        step_size,
        training_dirs,
        rom_training_dirs,
        training_parameters,
        training_qois,
        rom_training_parameters,
        rom_training_qois,
        rom_substep_start_iteration,
        rom_substep_end_iteration,
        num_rom_substeps,
        rejuvenation_reference_covariance,
        rejuvenation_count):
    dispatcher.np_savez(
        restart_path,
        sample_one_rom_results=sample_one_rom_results,
        sample_two_rom_results=sample_two_rom_results,
        sample_one_fom_results=sample_one_fom_results,
        parameter_samples_one=parameter_sample_sets[0],
        parameter_samples_two=parameter_sample_sets[1],
        iteration=iteration,
        step_size=step_size,
        training_directories=training_dirs,
        rom_training_directories=rom_training_dirs,
        training_parameters=training_parameters,
        training_qois=training_qois,
        rom_training_parameters=rom_training_parameters,
        rom_training_qois=rom_training_qois,
        rom_substep_start_iteration=rom_substep_start_iteration,
        rom_substep_end_iteration=(
            -1 if rom_substep_end_iteration is None
            else rom_substep_end_iteration
        ),
        num_rom_substeps=num_rom_substeps,
        rejuvenation_reference_covariance=rejuvenation_reference_covariance,
        rejuvenation_count=rejuvenation_count,
    )


def _run_mf_eki_rejuvenation(
        model,
        rom_model_builder,
        rom_model,
        observations,
        observations_covariance,
        absolute_eki_directory,
        iteration,
        parameter_names,
        parameter_sample_sets,
        reference_covariance,
        rejuvenation_count,
        rejuvenation_inflation,
        rejuvenation_prior_weight,
        random_seed,
        parameter_mins,
        parameter_maxes,
        fom_evaluation_concurrency,
        rom_evaluation_concurrency,
        dispatcher,
        rom_dispatcher,
        rom_tolerance,
        max_rom_training_dirs,
        training_dirs,
        training_parameters,
        training_qois,
        rom_training_dirs,
        rom_training_parameters,
        rom_training_qois,
        regularization_parameter):
    """Rejuvenate MF-EKI and refresh the ROM when the new cloud requires it."""
    first_set_size = parameter_sample_sets[0].shape[0]
    combined_parameter_samples = np.vstack(parameter_sample_sets)
    event_seed = random_seed + rejuvenation_count + 1
    combined_parameter_samples = _rejuvenate_parameter_samples(
        combined_parameter_samples,
        reference_covariance,
        rejuvenation_inflation,
        rejuvenation_prior_weight,
        event_seed,
        parameter_mins,
        parameter_maxes,
    )
    parameter_sample_sets = [
        combined_parameter_samples[:first_set_size].copy(),
        combined_parameter_samples[first_set_size:].copy(),
    ]

    event_dir = (
        f'{absolute_eki_directory}/iteration_{iteration}/'
        f'rejuvenation_{rejuvenation_count}'
    )
    fom_run_base = f'{event_dir}/run_fom_sample_set_0_'
    sample_one_fom_results = run_eki_iteration(
        model,
        observations,
        fom_run_base,
        parameter_names,
        parameter_sample_sets[0],
        fom_evaluation_concurrency,
        dispatcher,
    )

    training_dirs = copy.deepcopy(training_dirs)
    for i in range(first_set_size):
        training_dirs.append(fom_run_base + str(i))
    training_dirs.append(fom_run_base + "mean")
    training_parameters = np.vstack([
        training_parameters,
        parameter_sample_sets[0],
        np.mean(parameter_sample_sets[0], axis=0)[None, :],
    ])
    training_qois = np.vstack([
        training_qois,
        sample_one_fom_results['qois'].T,
        sample_one_fom_results['mean-qoi'][None, :],
    ])

    rom_run_base = f'{event_dir}/run_rom_sample_set_0_'
    sample_one_rom_results = run_eki_iteration(
        rom_model,
        observations,
        rom_run_base,
        parameter_names,
        parameter_sample_sets[0],
        rom_evaluation_concurrency,
        rom_dispatcher,
    )
    rom_error = _relative_rom_error(
        sample_one_rom_results,
        sample_one_fom_results,
    )

    if rom_error >= rom_tolerance:
        print(
            f'  Rejuvenated ensemble ROM error = {rom_error} above tolerance, '
            're-building ROM'
        )
        rom_training_dirs = training_dirs[-max_rom_training_dirs:]
        rom_training_parameters = training_parameters[-max_rom_training_dirs:]
        rom_training_qois = training_qois[-max_rom_training_dirs:]
        rom_model = rom_model_builder.build_from_training_dirs(
            event_dir + '/',
            rom_training_dirs,
            rom_training_parameters,
            rom_training_qois,
        )
        sample_one_rom_results = run_eki_iteration(
            rom_model,
            observations,
            rom_run_base,
            parameter_names,
            parameter_sample_sets[0],
            rom_evaluation_concurrency,
            rom_dispatcher,
        )
        rom_error = _relative_rom_error(
            sample_one_rom_results,
            sample_one_fom_results,
        )
        print(f'  Updated rejuvenated-ensemble ROM error = {rom_error}')
    else:
        print(
            f'  Rejuvenated ensemble ROM error = {rom_error} below tolerance, '
            're-using ROM'
        )

    rom_run_base = f'{event_dir}/run_rom_sample_set_1_'
    sample_two_rom_results = run_eki_iteration(
        rom_model,
        observations,
        rom_run_base,
        parameter_names,
        parameter_sample_sets[1],
        rom_evaluation_concurrency,
        rom_dispatcher,
    )

    error_norm = np.mean(
        np.linalg.norm(sample_one_fom_results['errors'], axis=0)
    )
    dps = compute_mf_eki_update(
        parameter_sample_sets,
        [sample_one_fom_results],
        [sample_one_rom_results, sample_two_rom_results],
        observations_covariance,
        regularization_parameter,
    )
    dp_norm = np.linalg.norm(dps[0])
    rejuvenation_count += 1
    spread = _compute_ensemble_spread(combined_parameter_samples)
    print(
        f'  Rejuvenated MF-EKI ensemble {rejuvenation_count}: '
        f'Error 2-norm: {error_norm:.5f}, Delta p: {dp_norm:.5f}, '
        f'Ensemble spread: {spread:.5f}'
    )

    return (
        parameter_sample_sets,
        sample_one_fom_results,
        sample_one_rom_results,
        sample_two_rom_results,
        error_norm,
        dps,
        dp_norm,
        rejuvenation_count,
        rom_model,
        training_dirs,
        training_parameters,
        training_qois,
        rom_training_dirs,
        rom_training_parameters,
        rom_training_qois,
    )


def run_mf_eki(model: QoiModel,
               rom_model_builder: QoiModelBuilderWithTrainingData,
               parameter_space: ParameterSpace,
               observations: np.ndarray,
               observations_covariance: np.ndarray,
               parameter_mins: np.ndarray = None,
               parameter_maxes: np.ndarray = None,
               absolute_eki_directory: str = os.getcwd() + "/work/",
               fom_ensemble_size: int = 10,
               rom_extra_ensemble_size=30,
               rom_tolerance: float = 0.005,
               use_updated_rom_in_update_on_rebuild: bool = True,
               initial_step_size: float = 1e-1,
               regularization_parameter: float = 1e-4,
               step_size_growth_factor: float = 1.25,
               step_size_decay_factor: float = 2.0,
               max_step_size_decrease_trys: int = 5,
               relaxation_parameter: float = 1.05,
               error_norm_tolerance: float = 1e-5,
               delta_params_tolerance: float = 1e-4,
               rejuvenation_strategy: str = "none",
               rejuvenation_interval: int = 5,
               rejuvenation_inflation: float = 1.1,
               rejuvenation_prior_weight: float = 0.0025,
               max_rejuvenations: int = 3,
               max_rom_training_history: int = 1,
               rom_substep_start_iteration: int = 0,
               rom_substep_end_iteration: Optional[int] = None,
               num_rom_substeps: int = 0,
               max_iterations: int = 50,
               random_seed: int = 1,
               fom_evaluation_concurrency: int = 1,
               rom_evaluation_concurrency: int = 1,
               restart_file: str = None,
               dispatcher: Optional[BaseDispatcher] = None):
    """Run a multi-fidelity ensemble Kalman inversion workflow."""
    dispatcher = resolve_dispatcher(dispatcher)
    rom_dispatcher = resolve_local_dispatcher(dispatcher)
    max_rom_training_dirs = int(
        max_rom_training_history * (fom_ensemble_size + 1)
    )
    start_time = time.time()

    require_relative_or_absolute_path(dispatcher, absolute_eki_directory)
    dispatcher.require_supported_concurrency(fom_evaluation_concurrency)
    assert step_size_growth_factor > 1.0, (
        "step_size_growth_factor must be greater than 1.0"
    )
    assert step_size_decay_factor > 1.0, (
        "step_size_decay_factor must be greater than 1.0"
    )
    _validate_rom_substep_settings(
        rom_substep_start_iteration,
        rom_substep_end_iteration,
        num_rom_substeps,
    )
    _validate_rejuvenation_settings(
        rejuvenation_strategy,
        rejuvenation_interval,
        rejuvenation_inflation,
        rejuvenation_prior_weight,
        max_rejuvenations,
    )
    if parameter_mins is not None:
        assert np.size(parameter_mins) == parameter_space.get_dimensionality(), (
            f"parameter_mins of size {np.size(parameter_mins)} is inconsistent "
            f"with the parameter_space of size {parameter_space.get_dimensionality()}"
        )
    if parameter_maxes is not None:
        assert np.size(parameter_maxes) == parameter_space.get_dimensionality(), (
            f"parameter_maxes of size {np.size(parameter_maxes)} is inconsistent "
            f"with the parameter_space of size {parameter_space.get_dimensionality()}"
        )

    np.random.seed(random_seed)
    ensemble_size = fom_ensemble_size + rom_extra_ensemble_size

    if restart_file is None:
        iteration = 0
        parameter_samples = parameter_space.generate_samples(ensemble_size)
        parameter_samples = bound_samples(
            parameter_samples,
            parameter_mins,
            parameter_maxes,
        )
        rejuvenation_reference_covariance = _compute_ensemble_covariance(
            parameter_samples
        )
        rejuvenation_count = 0
        parameter_sample_sets = [
            parameter_samples[:fom_ensemble_size].copy(),
            parameter_samples[fom_ensemble_size:].copy(),
        ]
        parameter_names = parameter_space.get_names()

        run_directory_base = (
            f'{absolute_eki_directory}/iteration_0/run_fom_sample_set_0_'
        )
        sample_one_fom_results = run_eki_iteration(
            model,
            observations,
            run_directory_base,
            parameter_names,
            parameter_sample_sets[0],
            fom_evaluation_concurrency,
            dispatcher,
        )
        training_dirs = [
            run_directory_base + str(i) for i in range(fom_ensemble_size)
        ]
        training_dirs.append(run_directory_base + "mean")
        training_parameters = np.vstack([
            parameter_sample_sets[0],
            np.mean(parameter_sample_sets[0], axis=0)[None, :],
        ])
        training_qois = np.vstack([
            sample_one_fom_results['qois'].T,
            sample_one_fom_results['mean-qoi'][None, :],
        ])
        rom_training_dirs = copy.deepcopy(training_dirs)
        rom_training_parameters = training_parameters.copy()
        rom_training_qois = training_qois.copy()
        offline_dir = f'{absolute_eki_directory}/iteration_0/'
        rom_model = rom_model_builder.build_from_training_dirs(
            offline_dir,
            rom_training_dirs,
            rom_training_parameters,
            rom_training_qois,
        )

        run_directory_base = (
            f'{absolute_eki_directory}/iteration_0/run_rom_sample_set_0_'
        )
        sample_one_rom_results = run_eki_iteration(
            rom_model,
            observations,
            run_directory_base,
            parameter_names,
            parameter_sample_sets[0],
            rom_evaluation_concurrency,
            rom_dispatcher,
        )
        rom_error = _relative_rom_error(
            sample_one_rom_results,
            sample_one_fom_results,
        )
        print(f'  ROM error = {rom_error}')

        run_directory_base = (
            f'{absolute_eki_directory}/iteration_0/run_rom_sample_set_1_'
        )
        sample_two_rom_results = run_eki_iteration(
            rom_model,
            observations,
            run_directory_base,
            parameter_names,
            parameter_sample_sets[1],
            rom_evaluation_concurrency,
            rom_dispatcher,
        )
        error_norm = np.mean(
            np.linalg.norm(sample_one_fom_results['errors'], axis=0)
        )
        step_size = initial_step_size
    else:
        restart_data = np.load(restart_file, allow_pickle=True)
        parameter_sample_sets = [
            restart_data['parameter_samples_one'],
            restart_data['parameter_samples_two'],
        ]
        iteration = int(restart_data['iteration'])
        step_size = float(restart_data['step_size'])
        if 'num_rom_substeps' in restart_data:
            rom_substep_start_iteration = int(
                restart_data['rom_substep_start_iteration']
            )
            saved_end_iteration = int(
                restart_data['rom_substep_end_iteration']
            )
            rom_substep_end_iteration = (
                None if saved_end_iteration < 0 else saved_end_iteration
            )
            num_rom_substeps = int(restart_data['num_rom_substeps'])
            _validate_rom_substep_settings(
                rom_substep_start_iteration,
                rom_substep_end_iteration,
                num_rom_substeps,
            )
        if 'training_directories' in restart_data:
            training_dirs = restart_data['training_directories'].tolist()
        else:
            training_dirs = restart_data['training_dirs'].tolist()
        rom_training_dirs = restart_data[
            'rom_training_directories'
        ].tolist()
        if ('training_parameters' not in restart_data
                or 'training_qois' not in restart_data):
            raise ValueError(
                "Restart file missing training_parameters/training_qois."
            )
        training_parameters = restart_data['training_parameters']
        training_qois = restart_data['training_qois']
        rom_training_parameters = restart_data['rom_training_parameters']
        rom_training_qois = restart_data['rom_training_qois']
        parameter_names = parameter_space.get_names()
        sample_one_fom_results = restart_data[
            'sample_one_fom_results'
        ].item()

        offline_dir = f'{absolute_eki_directory}/iteration_{iteration}/'
        print("==================Building ROM=============")
        rom_model = rom_model_builder.build_from_training_dirs(
            offline_dir,
            rom_training_dirs,
            rom_training_parameters,
            rom_training_qois,
        )
        print("==================ROM built================")
        sample_one_rom_results = restart_data[
            'sample_one_rom_results'
        ].item()
        sample_two_rom_results = restart_data[
            'sample_two_rom_results'
        ].item()
        error_norm = np.mean(
            np.linalg.norm(sample_one_fom_results['errors'], axis=0)
        )
        combined_parameter_samples = np.vstack(parameter_sample_sets)
        if 'rejuvenation_reference_covariance' in restart_data:
            rejuvenation_reference_covariance = restart_data[
                'rejuvenation_reference_covariance'
            ]
        else:
            rejuvenation_reference_covariance = _compute_ensemble_covariance(
                combined_parameter_samples
            )
        rejuvenation_count = (
            int(restart_data['rejuvenation_count'])
            if 'rejuvenation_count' in restart_data
            else 0
        )

    dps = compute_mf_eki_update(
        parameter_sample_sets,
        [sample_one_fom_results],
        [sample_one_rom_results, sample_two_rom_results],
        observations_covariance,
        regularization_parameter,
    )
    dp_norm = np.linalg.norm(dps[0])
    spread = _compute_ensemble_spread(np.vstack(parameter_sample_sets))
    wall_time = time.time() - start_time
    print(
        f'Iteration: {iteration}, Error 2-norm: {error_norm:.5f}, '
        f'Step size: {step_size:.5f}, Delta p: {dp_norm:.5f}, '
        f'Ensemble spread: {spread:.5f}, Wall time: {wall_time:.5f}'
    )

    while _should_rejuvenate(
            rejuvenation_strategy,
            iteration,
            dp_norm,
            error_norm,
            delta_params_tolerance,
            error_norm_tolerance,
            rejuvenation_count,
            max_rejuvenations,
            rejuvenation_interval):
        if rejuvenation_strategy != "adaptive":
            break
        (
            parameter_sample_sets,
            sample_one_fom_results,
            sample_one_rom_results,
            sample_two_rom_results,
            error_norm,
            dps,
            dp_norm,
            rejuvenation_count,
            rom_model,
            training_dirs,
            training_parameters,
            training_qois,
            rom_training_dirs,
            rom_training_parameters,
            rom_training_qois,
        ) = _run_mf_eki_rejuvenation(
            model,
            rom_model_builder,
            rom_model,
            observations,
            observations_covariance,
            absolute_eki_directory,
            iteration,
            parameter_names,
            parameter_sample_sets,
            rejuvenation_reference_covariance,
            rejuvenation_count,
            rejuvenation_inflation,
            rejuvenation_prior_weight,
            random_seed,
            parameter_mins,
            parameter_maxes,
            fom_evaluation_concurrency,
            rom_evaluation_concurrency,
            dispatcher,
            rom_dispatcher,
            rom_tolerance,
            max_rom_training_dirs,
            training_dirs,
            training_parameters,
            training_qois,
            rom_training_dirs,
            rom_training_parameters,
            rom_training_qois,
            regularization_parameter,
        )

    _save_mf_eki_restart(
        dispatcher,
        f'{absolute_eki_directory}/iteration_{iteration}/restart.npz',
        sample_one_rom_results,
        sample_two_rom_results,
        sample_one_fom_results,
        parameter_sample_sets,
        iteration,
        step_size,
        training_dirs,
        rom_training_dirs,
        training_parameters,
        training_qois,
        rom_training_parameters,
        rom_training_qois,
        rom_substep_start_iteration,
        rom_substep_end_iteration,
        num_rom_substeps,
        rejuvenation_reference_covariance,
        rejuvenation_count,
    )

    iteration += 1
    step_failed_counter = 0
    while iteration < max_iterations and error_norm > error_norm_tolerance:
        if dp_norm <= delta_params_tolerance:
            break

        test_parameter_sample_sets = copy.deepcopy(parameter_sample_sets)
        for i in range(len(dps)):
            test_parameter_sample_sets[i] = (
                parameter_sample_sets[i] + step_size * dps[i]
            )
            test_parameter_sample_sets[i] = bound_samples(
                test_parameter_sample_sets[i],
                parameter_mins,
                parameter_maxes,
            )

        outer_iteration = iteration - 1
        if _rom_substeps_are_enabled(
                outer_iteration,
                rom_substep_start_iteration,
                rom_substep_end_iteration,
                num_rom_substeps):
            test_parameter_sample_sets = _apply_rom_only_substeps(
                rom_model=rom_model,
                observations=observations,
                observations_covariance=observations_covariance,
                parameter_sample_sets=test_parameter_sample_sets,
                parameter_names=parameter_names,
                step_size=step_size,
                regularization_parameter=regularization_parameter,
                parameter_mins=parameter_mins,
                parameter_maxes=parameter_maxes,
                absolute_eki_directory=absolute_eki_directory,
                outer_iteration=outer_iteration,
                num_rom_substeps=num_rom_substeps,
                rom_evaluation_concurrency=rom_evaluation_concurrency,
                rom_dispatcher=rom_dispatcher,
            )

        run_directory_base = (
            f'{absolute_eki_directory}/iteration_{iteration}/'
            'run_fom_sample_set_0_'
        )
        test_training_dirs = copy.deepcopy(training_dirs)
        for i in range(fom_ensemble_size):
            test_training_dirs.append(run_directory_base + str(i))
        test_training_dirs.append(run_directory_base + "mean")
        test_sample_one_fom_results = run_eki_iteration(
            model,
            observations,
            run_directory_base,
            parameter_names,
            test_parameter_sample_sets[0],
            fom_evaluation_concurrency,
            dispatcher,
        )
        test_error_norm = np.mean(
            np.linalg.norm(test_sample_one_fom_results['errors'], axis=0)
        )
        test_training_parameters = np.vstack([
            training_parameters,
            test_parameter_sample_sets[0],
            np.mean(test_parameter_sample_sets[0], axis=0)[None, :],
        ])
        test_training_qois = np.vstack([
            training_qois,
            test_sample_one_fom_results['qois'].T,
            test_sample_one_fom_results['mean-qoi'][None, :],
        ])

        run_directory_base = (
            f'{absolute_eki_directory}/iteration_{iteration}/'
            'run_rom_sample_set_0_'
        )
        test_sample_one_rom_results = run_eki_iteration(
            rom_model,
            observations,
            run_directory_base,
            parameter_names,
            test_parameter_sample_sets[0],
            rom_evaluation_concurrency,
            rom_dispatcher,
        )
        rom_error = _relative_rom_error(
            test_sample_one_rom_results,
            test_sample_one_fom_results,
        )

        run_directory_base = (
            f'{absolute_eki_directory}/iteration_{iteration}/'
            'run_rom_sample_set_1_'
        )
        old_sample_two_rom_results = run_eki_iteration(
            rom_model,
            observations,
            run_directory_base,
            parameter_names,
            test_parameter_sample_sets[1],
            rom_evaluation_concurrency,
            rom_dispatcher,
        )
        old_sample_one_rom_results = test_sample_one_rom_results

        test_rom_training_dirs = copy.deepcopy(rom_training_dirs)
        test_rom_training_parameters = rom_training_parameters.copy()
        test_rom_training_qois = rom_training_qois.copy()
        rom_rebuilt_this_iteration = False
        if rom_error >= rom_tolerance:
            print(
                f'  ROM error = {rom_error} above tolerance, re-building ROM'
            )
            offline_dir = f'{absolute_eki_directory}/iteration_{iteration}/'
            test_rom_training_dirs = test_training_dirs[-max_rom_training_dirs:]
            test_rom_training_parameters = test_training_parameters[
                -max_rom_training_dirs:
            ]
            test_rom_training_qois = test_training_qois[-max_rom_training_dirs:]
            rom_model = rom_model_builder.build_from_training_dirs(
                offline_dir,
                test_rom_training_dirs,
                test_rom_training_parameters,
                test_rom_training_qois,
            )
            rom_rebuilt_this_iteration = True
            run_directory_base = (
                f'{absolute_eki_directory}/iteration_{iteration}/'
                'run_rom_sample_set_0_'
            )
            test_sample_one_rom_results = run_eki_iteration(
                rom_model,
                observations,
                run_directory_base,
                parameter_names,
                test_parameter_sample_sets[0],
                rom_evaluation_concurrency,
                rom_dispatcher,
            )
            rom_error = _relative_rom_error(
                test_sample_one_rom_results,
                test_sample_one_fom_results,
            )
            print(f'  Updated ROM error = {rom_error}')
            run_directory_base = (
                f'{absolute_eki_directory}/iteration_{iteration}/'
                'run_rom_sample_set_1_'
            )
            test_sample_two_rom_results = run_eki_iteration(
                rom_model,
                observations,
                run_directory_base,
                parameter_names,
                test_parameter_sample_sets[1],
                rom_evaluation_concurrency,
                rom_dispatcher,
            )
        else:
            print(f'  ROM error = {rom_error} below tolerance, re-using ROM')
            test_sample_two_rom_results = old_sample_two_rom_results

        if test_error_norm < relaxation_parameter * error_norm:
            step_failed_counter = 0
            parameter_sample_sets = test_parameter_sample_sets.copy()
            sample_one_fom_results = test_sample_one_fom_results.copy()
            sample_one_rom_results = test_sample_one_rom_results.copy()
            sample_two_rom_results = test_sample_two_rom_results.copy()
            error_norm = float(test_error_norm)
            step_size = min(step_size * step_size_growth_factor, 1.0)

            if rom_rebuilt_this_iteration and not use_updated_rom_in_update_on_rebuild:
                rom_sample_results = [
                    old_sample_one_rom_results,
                    old_sample_two_rom_results,
                ]
            else:
                rom_sample_results = [
                    sample_one_rom_results,
                    sample_two_rom_results,
                ]
            dps = compute_mf_eki_update(
                parameter_sample_sets,
                [sample_one_fom_results],
                rom_sample_results,
                observations_covariance,
                regularization_parameter,
            )
            dp_norm = np.linalg.norm(dps[0])
            training_dirs = copy.deepcopy(test_training_dirs)
            training_parameters = test_training_parameters.copy()
            training_qois = test_training_qois.copy()
            rom_training_dirs = copy.deepcopy(test_rom_training_dirs)
            rom_training_parameters = test_rom_training_parameters.copy()
            rom_training_qois = test_rom_training_qois.copy()

            if _should_rejuvenate(
                    rejuvenation_strategy,
                    iteration,
                    dp_norm,
                    error_norm,
                    delta_params_tolerance,
                    error_norm_tolerance,
                    rejuvenation_count,
                    max_rejuvenations,
                    rejuvenation_interval):
                if rejuvenation_strategy == "periodic":
                    rejuvenations_this_iteration = 1
                else:
                    rejuvenations_this_iteration = max_rejuvenations
                while (
                    rejuvenations_this_iteration > 0
                    and _should_rejuvenate(
                        rejuvenation_strategy,
                        iteration,
                        dp_norm,
                        error_norm,
                        delta_params_tolerance,
                        error_norm_tolerance,
                        rejuvenation_count,
                        max_rejuvenations,
                        rejuvenation_interval,
                    )
                ):
                    (
                        parameter_sample_sets,
                        sample_one_fom_results,
                        sample_one_rom_results,
                        sample_two_rom_results,
                        error_norm,
                        dps,
                        dp_norm,
                        rejuvenation_count,
                        rom_model,
                        training_dirs,
                        training_parameters,
                        training_qois,
                        rom_training_dirs,
                        rom_training_parameters,
                        rom_training_qois,
                    ) = _run_mf_eki_rejuvenation(
                        model,
                        rom_model_builder,
                        rom_model,
                        observations,
                        observations_covariance,
                        absolute_eki_directory,
                        iteration,
                        parameter_names,
                        parameter_sample_sets,
                        rejuvenation_reference_covariance,
                        rejuvenation_count,
                        rejuvenation_inflation,
                        rejuvenation_prior_weight,
                        random_seed,
                        parameter_mins,
                        parameter_maxes,
                        fom_evaluation_concurrency,
                        rom_evaluation_concurrency,
                        dispatcher,
                        rom_dispatcher,
                        rom_tolerance,
                        max_rom_training_dirs,
                        training_dirs,
                        training_parameters,
                        training_qois,
                        rom_training_dirs,
                        rom_training_parameters,
                        rom_training_qois,
                        regularization_parameter,
                    )
                    rejuvenations_this_iteration -= 1

            wall_time = time.time() - start_time
            spread = _compute_ensemble_spread(np.vstack(parameter_sample_sets))
            print(
                f'Iteration: {iteration}, Error 2-norm: {error_norm:.5f}, '
                f'Step size: {step_size:.5f}, Delta p: {dp_norm:.5f}, '
                f'Ensemble spread: {spread:.5f}, Wall time: {wall_time:.5f}'
            )
            _save_mf_eki_restart(
                dispatcher,
                f'{absolute_eki_directory}/iteration_{iteration}/restart.npz',
                sample_one_rom_results,
                sample_two_rom_results,
                sample_one_fom_results,
                parameter_sample_sets,
                iteration,
                step_size,
                training_dirs,
                rom_training_dirs,
                training_parameters,
                training_qois,
                rom_training_parameters,
                rom_training_qois,
                rom_substep_start_iteration,
                rom_substep_end_iteration,
                num_rom_substeps,
                rejuvenation_reference_covariance,
                rejuvenation_count,
            )
            iteration += 1
        else:
            step_failed_counter += 1
            step_size /= step_size_decay_factor
            print(
                f'  Warning, lowering step size, Iteration: {iteration}, '
                f'Error 2-norm: {error_norm:.5f}, Step size: {step_size:.5f}, '
                f'Delta p: {dp_norm:.5f}'
            )
            if step_failed_counter > max_step_size_decrease_trys:
                print(
                    f'  Failed to advance after '
                    f'{max_step_size_decrease_trys}, exiting'
                )
                break

    if iteration >= max_iterations:
        print('Max iterations reached, terminating')
    elif error_norm <= error_norm_tolerance:
        print('Error norm dropped below tolerance!')
    elif dp_norm <= delta_params_tolerance:
        if rejuvenation_count >= max_rejuvenations:
            print(
                'MF-EKI parameter update stagnated above the residual '
                'tolerance and the maximum number of rejuvenations was reached.'
            )
        else:
            print('MF-EKI parameter update dropped below tolerance.')
    return parameter_sample_sets[0], sample_one_fom_results['qois']


def mf_eki_with_auto_rom(model: QoiModel,
                         parameter_space: ParameterSpace,
                         observations: np.ndarray,
                         observations_covariance: np.ndarray,
                         parameter_mins: np.ndarray = None,
                         parameter_maxes: np.ndarray = None,
                         absolute_eki_directory: str = os.getcwd() + "/work/",
                         fom_ensemble_size: int = 10,
                         rom_extra_ensemble_size=30,
                         rom_tolerance: float = 0.005,
                         use_updated_rom_in_update_on_rebuild: bool = False,
                         initial_step_size: float = 0.05,
                         regularization_parameter: float = 1e-4,
                         step_size_growth_factor: float = 1.25,
                         step_size_decay_factor: float = 2.0,
                         max_step_size_decrease_trys: int = 5,
                         relaxation_parameter: float = 1.05,
                         error_norm_tolerance: float = 1e-5,
                         delta_params_tolerance: float = 1e-4,
                         rejuvenation_strategy: str = "none",
                         rejuvenation_interval: int = 5,
                         rejuvenation_inflation: float = 1.1,
                         rejuvenation_prior_weight: float = 0.0025,
                         max_rejuvenations: int = 3,
                         max_rom_training_history: int = 1,
                         rom_substep_start_iteration: int = 0,
                         rom_substep_end_iteration: Optional[int] = None,
                         num_rom_substeps: int = 0,
                         max_iterations: int = 50,
                         random_seed: int = 1,
                         fom_evaluation_concurrency: int = 1,
                         rom_evaluation_concurrency: int = 1,
                         restart_file: str = None,
                         rom_type: str = "gp",
                         rom_args: Optional[dict] = None,
                         dispatcher: Optional[BaseDispatcher] = None):
    """Run MF-EKI with a built-in data-driven surrogate type.

    ``rom_type='gp'`` selects the Gaussian-process QoI surrogate.
    ``rom_type='nn'`` selects the PyTorch neural-network QoI surrogate.
    ``rom_args`` contains surrogate-specific builder options. For the neural
    network, ``network_config`` and ``lipschitz_config`` may be supplied as
    :class:`NeuralNetworkConfig` and :class:`LipschitzConfig` instances.
    """
    rom_args = {} if rom_args is None else dict(rom_args)
    rom_type_normalized = rom_type.strip().lower()
    if rom_type_normalized == "gp":
        rom_model_builder = GaussianProcessQoiModelBuilderWithTrainingData(
            parameter_names=parameter_space.get_names(),
            pod_energy_fraction=rom_args.get("pod_energy_fraction", 0.999999),
            max_pod_modes=rom_args.get("max_pod_modes"),
            kernel=rom_args.get("kernel"),
            noise_variance=rom_args.get("noise_variance"),
            auto_noise_variance=rom_args.get("auto_noise_variance", False),
            noise_variance_fraction=rom_args.get(
                "noise_variance_fraction", 1e-6
            ),
            tune_hyperparameters=rom_args.get("tune_hyperparameters", False),
            length_scale_grid=rom_args.get("length_scale_grid"),
            signal_variance_grid=rom_args.get("signal_variance_grid"),
            normalize_parameters=rom_args.get("normalize_parameters", False),
            normalize_targets=rom_args.get("normalize_targets", False),
        )
    elif rom_type_normalized in ("nn", "neural_network", "neural-network"):
        rom_model_builder = NeuralNetworkQoiModelBuilderWithTrainingData(
            parameter_names=parameter_space.get_names(),
            pod_energy_fraction=rom_args.get("pod_energy_fraction", 0.999999),
            max_pod_modes=rom_args.get("max_pod_modes"),
            network_config=rom_args.get("network_config"),
            lipschitz_config=rom_args.get("lipschitz_config"),
            normalize_parameters=rom_args.get("normalize_parameters", True),
            normalize_targets=rom_args.get("normalize_targets", True),
        )
    else:
        raise ValueError(
            f"Unsupported rom_type '{rom_type}'. Supported options are 'gp' and 'nn'."
        )
    return run_mf_eki(
        model=model,
        rom_model_builder=rom_model_builder,
        parameter_space=parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=parameter_mins,
        parameter_maxes=parameter_maxes,
        absolute_eki_directory=absolute_eki_directory,
        fom_ensemble_size=fom_ensemble_size,
        rom_extra_ensemble_size=rom_extra_ensemble_size,
        rom_tolerance=rom_tolerance,
        use_updated_rom_in_update_on_rebuild=use_updated_rom_in_update_on_rebuild,
        initial_step_size=initial_step_size,
        regularization_parameter=regularization_parameter,
        step_size_growth_factor=step_size_growth_factor,
        step_size_decay_factor=step_size_decay_factor,
        max_step_size_decrease_trys=max_step_size_decrease_trys,
        relaxation_parameter=relaxation_parameter,
        error_norm_tolerance=error_norm_tolerance,
        delta_params_tolerance=delta_params_tolerance,
        rejuvenation_strategy=rejuvenation_strategy,
        rejuvenation_interval=rejuvenation_interval,
        rejuvenation_inflation=rejuvenation_inflation,
        rejuvenation_prior_weight=rejuvenation_prior_weight,
        max_rejuvenations=max_rejuvenations,
        max_rom_training_history=max_rom_training_history,
        rom_substep_start_iteration=rom_substep_start_iteration,
        rom_substep_end_iteration=rom_substep_end_iteration,
        num_rom_substeps=num_rom_substeps,
        max_iterations=max_iterations,
        random_seed=random_seed,
        fom_evaluation_concurrency=fom_evaluation_concurrency,
        rom_evaluation_concurrency=rom_evaluation_concurrency,
        restart_file=restart_file,
        dispatcher=dispatcher,
    )


def compute_mf_eki_update(
        parameter_sample_sets,
        fom_results_for_sample_sets,
        rom_results_for_sample_sets,
        observations_covariance,
        regularization_parameter):
    """Compute the update matrices for the MF-EKI algorithm."""
    ensemble_sizes = [
        parameter_sample_sets[0].shape[1],
        parameter_sample_sets[1].shape[1],
    ]

    dys_fom = [
        fom_results_for_sample_sets[0]['qois']
        - fom_results_for_sample_sets[0]['mean-qoi'][:, None]
    ]
    dys_rom = [
        rom_results_for_sample_sets[0]['qois']
        - rom_results_for_sample_sets[0]['mean-qoi'][:, None],
        rom_results_for_sample_sets[1]['qois']
        - rom_results_for_sample_sets[1]['mean-qoi'][:, None],
    ]

    C = dys_fom[0] @ dys_fom[0].transpose()
    C += 0.25 * (dys_rom[0] @ dys_rom[0].transpose())
    C -= 0.5 * (dys_fom[0] @ dys_rom[0].transpose())
    C -= 0.5 * (dys_rom[0] @ dys_fom[0].transpose())
    C *= 1.0 / (ensemble_sizes[0] - 1)
    C += (
        0.25
        * (dys_rom[1] @ dys_rom[1].transpose())
        / (ensemble_sizes[1] - 1)
    )

    dws = [
        (
            parameter_sample_sets[0]
            - np.mean(parameter_sample_sets[0], axis=0)[None]
        ).transpose(),
        (
            parameter_sample_sets[1]
            - np.mean(parameter_sample_sets[1], axis=0)[None]
        ).transpose(),
    ]

    C_p = dws[0] @ dys_fom[0].transpose()
    C_p += 0.25 * dws[0] @ dys_rom[0].transpose()
    C_p -= 0.5 * dws[0] @ dys_rom[0].transpose()
    C_p -= 0.5 * dws[0] @ dys_fom[0].transpose()
    C_p *= 1.0 / (ensemble_sizes[0] - 1)
    C_p += (
        0.25
        * dws[1]
        @ dys_rom[1].transpose()
        / (ensemble_sizes[1] - 1)
    )

    I = np.eye(observations_covariance.shape[1])
    LHS = C + observations_covariance + regularization_parameter * I
    RHS = fom_results_for_sample_sets[0]['errors']
    dp = np.linalg.solve(LHS, RHS)
    dp = C_p @ dp

    RHS = rom_results_for_sample_sets[1]['errors']
    dpr = np.linalg.solve(LHS, RHS)
    dpr = C_p @ dpr

    return [dp.transpose(), dpr.transpose()]
