import numpy as np
import os
import time
from typing import Optional
from romtools.workflows.inverse.eki_utils import *
from romtools.workflows.models import QoiModel
from romtools.workflows.model_builders import QoiModelBuilderWithTrainingData
from romtools.rom.qoi_surrogates import GaussianProcessKernel, GaussianProcessQoiModel
import copy
from romtools.workflows.parameter_spaces import ParameterSpace
import concurrent.futures
import multiprocessing


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
        self.parameter_names = list(parameter_names) if parameter_names is not None else None
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

def run_mf_eki(model: QoiModel,
            rom_model_builder: QoiModelBuilderWithTrainingData,
            parameter_space: ParameterSpace,
            observations: np.ndarray,
            observations_covariance: np.ndarray,
            parameter_mins: np.ndarray = None,
            parameter_maxes: np.ndarray = None,
            absolute_eki_directory: str = os.getcwd() + "/work/",
            fom_ensemble_size: int = 10,
            rom_extra_ensemble_size = 30,
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
            max_rom_training_history: int = 1,
            max_iterations: int = 50,
            random_seed: int = 1,
            fom_evaluation_concurrency: int = 1,
            rom_evaluation_concurrency: int = 1,
            restart_file: str = None):  # Optional parameter for restart file
    '''
    Run a multi-fidelity ensemble Kalman inversion (MF-EKI) workflow.

    This routine couples a high-fidelity model with ROM evaluations built on
    the fly from FOM training data. It supports adaptive step sizing, ROM
    refresh based on QoI error, and concurrent evaluation of FOM/ROM samples.

    Args:
        model: High-fidelity QoiModel.
        rom_model_builder: Builder for ROM models from training data and directories.
        parameter_space: ParameterSpace used to draw the initial ensemble.
        observations: Observed QoI vector.
        observations_covariance: Observation covariance matrix.
        parameter_mins: Optional lower bounds on parameters.
        parameter_maxes: Optional upper bounds on parameters.
        absolute_eki_directory: Absolute path to the working directory for runs.
        fom_ensemble_size: Number of FOM samples per iteration.
        rom_extra_ensemble_size: Extra ROM-only samples per iteration.
        rom_tolerance: Relative QoI error threshold for ROM rebuilds.
        use_updated_rom_in_update_on_rebuild: When True, use the updated ROM
            in the update for that step. When False, use the previous ROM in
            the update for that step.
        initial_step_size: Initial step size for the update.
        regularization_parameter: Tikhonov regularization parameter.
        step_size_growth_factor: Growth factor when a step is accepted.
        step_size_decay_factor: Decay factor when a step is rejected.
        max_step_size_decrease_trys: Max consecutive step reductions before exit.
        relaxation_parameter: Error reduction factor for step acceptance.
        error_norm_tolerance: Stop when mean error norm falls below this value.
        delta_params_tolerance: Stop when parameter update norm falls below this value.
        max_rom_training_history: Max number of training iterations to keep.
        max_iterations: Maximum number of EKI iterations.
        random_seed: RNG seed for initial sampling.
        fom_evaluation_concurrency: Concurrent FOM evaluations per iteration.
        rom_evaluation_concurrency: Concurrent ROM evaluations per iteration.
        restart_file: Optional restart file path.

    Returns:
        Tuple of (fom_parameter_samples, fom_qois) from the final iteration.
    '''


    max_rom_training_dirs = int(max_rom_training_history*(fom_ensemble_size+1))
    start_time = time.time()
    
    ## Error checking======
    assert os.path.isabs(absolute_eki_directory), f"eki_directory is not an absolute path ({absolute_eki_directory})"
    assert step_size_growth_factor > 1.0, "step_size_growth_factor must be greater than 1.0"
    assert step_size_decay_factor > 1.0, "step_size_decay_factor must be greater than 1.0"
    if parameter_mins is not None:
      assert np.size(parameter_mins) == parameter_space.get_dimensionality(), f"parameter_mins of size {np.size(parameter_mins)} is inconsistent with the parameter_space of size {parameter_space.get_dimensionality()}" 
    if parameter_maxes is not None:
      assert np.size(parameter_maxes) == parameter_space.get_dimensionality(), f"parameter_maxes of size {np.size(parameter_maxes)} is inconsistent with the parameter_space of size {parameter_space.get_dimensionality()}" 
    ##====================

    np.random.seed(random_seed)

    # create initial samples
    ensemble_size = fom_ensemble_size + rom_extra_ensemble_size

    if restart_file is None:
        iteration = 0
        parameter_samples = parameter_space.generate_samples(ensemble_size)
        parameter_samples = bound_samples(parameter_samples,parameter_mins,parameter_maxes)
        parameter_samples_one = parameter_samples[0:fom_ensemble_size]
        parameter_samples_two = parameter_samples[fom_ensemble_size::]
        parameter_sample_sets = [parameter_samples_one,parameter_samples_two]
        parameter_names = parameter_space.get_names()
        #Run initial step and compute update
        run_directory_base = f'{absolute_eki_directory}/iteration_{0}/run_fom_sample_set_0_'
        sample_one_fom_results = {}
        sample_one_fom_results = run_eki_iteration(model, observations, run_directory_base, parameter_names, parameter_sample_sets[0], fom_evaluation_concurrency)

        # Build ROM
        training_dirs = []
        for i in range(0,fom_ensemble_size):
          training_dirs.append(run_directory_base + str(i))
        training_dirs.append(run_directory_base + "mean")

        offline_dir = f'{absolute_eki_directory}/iteration_{0}/'
        training_parameters = np.vstack([
            parameter_sample_sets[0],
            np.mean(parameter_sample_sets[0], axis=0)[None, :]
        ])
        training_qois = np.vstack([
            sample_one_fom_results['qois'].T,
            sample_one_fom_results['mean-qoi'][None, :]
        ])
        rom_training_dirs = copy.deepcopy(training_dirs)
        rom_training_parameters = training_parameters.copy()
        rom_training_qois = training_qois.copy()
        rom_model = rom_model_builder.build_from_training_dirs(
            offline_dir,
            rom_training_dirs,
            rom_training_parameters,
            rom_training_qois
        )
 
        run_directory_base = f'{absolute_eki_directory}/iteration_{0}/run_rom_sample_set_0_'
        sample_one_rom_results = run_eki_iteration(rom_model, observations, run_directory_base, parameter_names, parameter_sample_sets[0], rom_evaluation_concurrency)

        rom_errors = np.linalg.norm(sample_one_rom_results['qois'] - sample_one_fom_results['qois'])/ np.linalg.norm(sample_one_fom_results['qois'])
        print(f'  ROM error = {rom_errors}')

        run_directory_base = f'{absolute_eki_directory}/iteration_{0}/run_rom_sample_set_1_'
        sample_two_rom_results = run_eki_iteration(rom_model, observations, run_directory_base, parameter_names, parameter_sample_sets[1], rom_evaluation_concurrency)

        error_norm = np.mean(np.linalg.norm(sample_one_fom_results['errors'], axis=0))
        print(f'Initial error: {error_norm}')
        step_size = initial_step_size 

        np.savez(
            f'{absolute_eki_directory}/iteration_{iteration}/restart.npz',
            sample_one_rom_results=sample_one_rom_results,
            sample_two_rom_results=sample_two_rom_results,
            sample_one_fom_results=sample_one_fom_results,
            parameter_samples_one=parameter_sample_sets[0],
            parameter_samples_two=parameter_sample_sets[1],
            iteration=iteration,
            step_size=step_size,
            training_dirs=training_dirs,
            rom_training_directories=rom_training_dirs,
            training_parameters=training_parameters,
            training_qois=training_qois,
            rom_training_parameters=rom_training_parameters,
            rom_training_qois=rom_training_qois
        )
    else:
        restart_file = np.load(restart_file,allow_pickle=True)
        parameter_samples_one = restart_file['parameter_samples_one']
        parameter_samples_two = restart_file['parameter_samples_two']
        parameter_sample_sets = [parameter_samples_one,parameter_samples_two]
        iteration = restart_file['iteration']
        step_size = restart_file['step_size']
        training_dirs = restart_file['training_directories'].tolist()
        rom_training_dirs = restart_file['rom_training_directories'].tolist()
        if 'training_parameters' not in restart_file or 'training_qois' not in restart_file:
            raise ValueError("Restart file missing training_parameters/training_qois.")
        training_parameters = restart_file['training_parameters']
        training_qois = restart_file['training_qois']
        rom_training_parameters = restart_file['rom_training_parameters']
        rom_training_qois = restart_file['rom_training_qois']
        parameter_names = parameter_space.get_names()

        # Run initial step and compute update
        run_directory_base = f'{absolute_eki_directory}/iteration_{iteration}/run_fom_sample_set_0_'
        sample_one_fom_results = restart_file['sample_one_fom_results'].item()

        offline_dir = f'{absolute_eki_directory}/iteration_{iteration}/'
        print("==================Building ROM=============")
        rom_model = rom_model_builder.build_from_training_dirs(
            offline_dir,
            rom_training_dirs,
            rom_training_parameters,
            rom_training_qois
        )
        print("==================ROM built================")
        sample_one_rom_results = restart_file['sample_one_rom_results'].item() 

        sample_two_rom_results = restart_file['sample_two_rom_results'].item()
        error_norm = np.mean(np.linalg.norm(sample_one_fom_results['errors'], axis=0))

    # Compute ENKF update 
    fom_sample_results = [sample_one_fom_results]
    rom_sample_results = [sample_one_rom_results,sample_two_rom_results]
    dps = compute_mf_eki_update(parameter_sample_sets,fom_sample_results,rom_sample_results,observations_covariance, regularization_parameter)
    dp_norm = np.linalg.norm(dps[0])
    wall_time = time.time() - start_time
    print(f'Iteration: {iteration}, Error 2-norm: {error_norm:.5f}, Step size: {step_size:.5f}, Delta p: {dp_norm:.5f}, Wall time: {wall_time:.5f}')
    # Iterative optimization loop
    iteration += 1
    step_failed_counter = 0
    while iteration < max_iterations and error_norm > error_norm_tolerance and dp_norm > delta_params_tolerance:
        # Test the parameter update for the step size
        test_parameter_sample_sets = copy.deepcopy(parameter_sample_sets)
        for i in range(len(dps)):
          test_parameter_sample_sets[i] = parameter_sample_sets[i] + step_size * dps[i] 
          test_parameter_sample_sets[i] = bound_samples(test_parameter_sample_sets[i],parameter_mins,parameter_maxes)

        run_directory_base = f'{absolute_eki_directory}/iteration_{iteration}/run_fom_sample_set_0_'
        test_training_dirs = copy.deepcopy(training_dirs)
        for i in range(0,fom_ensemble_size):
          test_training_dirs.append(run_directory_base + str(i))
        test_training_dirs.append(run_directory_base + "mean")

        # Run the EKI iteration with test parameters
        test_sample_one_fom_results = run_eki_iteration(model, observations, run_directory_base, parameter_names, test_parameter_sample_sets[0], fom_evaluation_concurrency)
        test_error_norm = np.mean(np.linalg.norm(test_sample_one_fom_results['errors'], axis=0))
        test_training_parameters = np.vstack([
            training_parameters,
            test_parameter_sample_sets[0],
            np.mean(test_parameter_sample_sets[0], axis=0)[None, :]
        ])
        test_training_qois = np.vstack([
            training_qois,
            test_sample_one_fom_results['qois'].T,
            test_sample_one_fom_results['mean-qoi'][None, :]
        ])

        run_directory_base = f'{absolute_eki_directory}/iteration_{iteration}/run_rom_sample_set_0_'
        test_sample_one_rom_results = run_eki_iteration(rom_model, observations, run_directory_base, parameter_names, test_parameter_sample_sets[0], rom_evaluation_concurrency)
        rom_errors = np.linalg.norm(test_sample_one_rom_results['qois'] - test_sample_one_fom_results['qois'])/ np.linalg.norm(test_sample_one_fom_results['qois'])

        run_directory_base = f'{absolute_eki_directory}/iteration_{iteration}/run_rom_sample_set_1_'
        old_sample_two_rom_results = run_eki_iteration(rom_model, observations, run_directory_base, parameter_names, test_parameter_sample_sets[1], rom_evaluation_concurrency)
        old_sample_one_rom_results = test_sample_one_rom_results

        test_rom_training_dirs = copy.deepcopy(rom_training_dirs)
        test_rom_training_parameters = rom_training_parameters.copy()
        test_rom_training_qois = rom_training_qois.copy()
        rom_rebuilt_this_iteration = False
        if rom_errors >= rom_tolerance:
          # Build ROM
          print(f'  ROM error = {rom_errors} above tolerance, re-building ROM')
          offline_dir = f'{absolute_eki_directory}/iteration_{iteration}/'
          test_rom_training_dirs = test_training_dirs[-max_rom_training_dirs::]
          test_rom_training_parameters = test_training_parameters[-max_rom_training_dirs::]
          test_rom_training_qois = test_training_qois[-max_rom_training_dirs::]
          rom_model = rom_model_builder.build_from_training_dirs(
              offline_dir,
              test_rom_training_dirs,
              test_rom_training_parameters,
              test_rom_training_qois
          )
          rom_rebuilt_this_iteration = True
          run_directory_base = f'{absolute_eki_directory}/iteration_{iteration}/run_rom_sample_set_0_'
          test_sample_one_rom_results = run_eki_iteration(rom_model, observations, run_directory_base, parameter_names, test_parameter_sample_sets[0], rom_evaluation_concurrency)
          rom_errors = np.linalg.norm(test_sample_one_rom_results['qois'] - test_sample_one_fom_results['qois'])/ np.linalg.norm(test_sample_one_fom_results['qois'])
          print(f'  Updated ROM error = {rom_errors}')
          run_directory_base = f'{absolute_eki_directory}/iteration_{iteration}/run_rom_sample_set_1_'
          test_sample_two_rom_results = run_eki_iteration(rom_model, observations, run_directory_base, parameter_names, test_parameter_sample_sets[1], rom_evaluation_concurrency)
        else:
          print(f'  ROM error = {rom_errors} below tolerance, re-using ROM')
          test_sample_two_rom_results = old_sample_two_rom_results

        if test_error_norm < relaxation_parameter * error_norm:
            step_failed_counter = 0
            # If error norm drops, continue the iteration and grow the step size
            parameter_sample_sets = test_parameter_sample_sets.copy()
            sample_one_fom_results = test_sample_one_fom_results.copy()
            sample_one_rom_results = test_sample_one_rom_results.copy()
            sample_two_rom_results = test_sample_two_rom_results.copy()
            error_norm = test_error_norm
            step_size *= step_size_growth_factor
            wall_time = time.time() - start_time

            # Compute Kalman update
            fom_sample_results = [sample_one_fom_results]
            if rom_rebuilt_this_iteration and not use_updated_rom_in_update_on_rebuild:
              rom_sample_results = [old_sample_one_rom_results, old_sample_two_rom_results]
            else:
              rom_sample_results = [sample_one_rom_results, sample_two_rom_results]
            dps = compute_mf_eki_update(parameter_sample_sets, fom_sample_results, rom_sample_results, observations_covariance, regularization_parameter)
            dp_norm = np.linalg.norm(dps[0])
            training_dirs = copy.deepcopy(test_training_dirs)
            training_parameters = test_training_parameters.copy()
            training_qois = test_training_qois.copy()
            rom_training_dirs = copy.deepcopy(test_rom_training_dirs)
            rom_training_parameters = test_rom_training_parameters.copy()
            rom_training_qois = test_rom_training_qois.copy()
            print(f'Iteration: {iteration}, Error 2-norm: {error_norm:.5f}, Step size: {step_size:.5f}, Delta p: {dp_norm:.5f}, Wall time: {wall_time:.5f}')
            # Save the current state to the restart file
            np.savez(
                f'{absolute_eki_directory}/iteration_{iteration}/restart.npz',
                sample_one_rom_results=sample_one_rom_results,
                sample_two_rom_results=sample_two_rom_results,
                sample_one_fom_results=sample_one_fom_results,
                parameter_samples_one=parameter_sample_sets[0],
                parameter_samples_two=parameter_sample_sets[1],
                iteration=iteration,
                step_size=step_size,
                rom_training_directories=rom_training_dirs,
                training_directories=training_dirs,
                training_parameters=training_parameters,
                training_qois=training_qois,
                rom_training_parameters=rom_training_parameters,
                rom_training_qois=rom_training_qois
            )
            iteration += 1
          
        else:
            # Else, drop the step size 
            step_failed_counter += 1
            step_size /= step_size_decay_factor 
            print(f'  Warning, lowering step size, Iteration: {iteration}, Error 2-norm: {error_norm:.5f}, Step size: {step_size:.5f}, Delta p: {dp_norm:.5f}')
            if step_failed_counter > max_step_size_decrease_trys:
              print(f'  Failed to advance after {max_step_size_decrease_trys}, exiting')
              break 

    if iteration >= max_iterations:
        print(f'Max iterations reached, terminating')
    elif error_norm <= error_norm_tolerance:
        print(f'Error norm dropped below tolerance!')
    elif dp_norm <= delta_params_tolerance:
        print(f'Changed to parameter update dropped below tolerance!')
    return parameter_sample_sets[0],sample_one_fom_results['qois']


def mf_eki_with_auto_rom(model: QoiModel,
                         parameter_space: ParameterSpace,
                         observations: np.ndarray,
                         observations_covariance: np.ndarray,
                         parameter_mins: np.ndarray = None,
                         parameter_maxes: np.ndarray = None,
                         absolute_eki_directory: str = os.getcwd() + "/work/",
                         fom_ensemble_size: int = 10,
                         rom_extra_ensemble_size = 30,
                         rom_tolerance: float = 0.005,
                         use_updated_rom_in_update_on_rebuild: bool = False,
                         initial_step_size: float = 1e-1,
                         regularization_parameter: float = 1e-4,
                         step_size_growth_factor: float = 1.25,
                         step_size_decay_factor: float = 2.0,
                         max_step_size_decrease_trys: int = 5,
                         relaxation_parameter: float = 1.05,
                         error_norm_tolerance: float = 1e-5,
                         delta_params_tolerance: float = 1e-4,
                         max_rom_training_history: int = 1,
                         max_iterations: int = 50,
                         random_seed: int = 1,
                         fom_evaluation_concurrency: int = 1,
                         rom_evaluation_concurrency: int = 1,
                         restart_file: str = None,
                         rom_type: str = "gp",
                         rom_args: Optional[dict] = None):
    """
    Wrapper around run_mf_eki that selects a default ROM surrogate by rom_type.
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
            noise_variance_fraction=rom_args.get("noise_variance_fraction", 1e-6),
            tune_hyperparameters=rom_args.get("tune_hyperparameters", False),
            length_scale_grid=rom_args.get("length_scale_grid"),
            signal_variance_grid=rom_args.get("signal_variance_grid"),
            normalize_parameters=rom_args.get("normalize_parameters", False),
            normalize_targets=rom_args.get("normalize_targets", False),
        )
    else:
        raise ValueError(f"Unsupported rom_type '{rom_type}'.")
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
        max_rom_training_history=max_rom_training_history,
        max_iterations=max_iterations,
        random_seed=random_seed,
        fom_evaluation_concurrency=fom_evaluation_concurrency,
        rom_evaluation_concurrency=rom_evaluation_concurrency,
        restart_file=restart_file,
    )

def compute_mf_eki_update(parameter_sample_sets, fom_results_for_sample_sets, rom_results_for_sample_sets,observations_covariance, regularization_parameter):
    """Compute the update matrices for the MF-EKI algorithm."""
    ensemble_sizes = [parameter_sample_sets[0].shape[1],parameter_sample_sets[1].shape[1]]

    dys_fom = [ fom_results_for_sample_sets[0]['qois'] - fom_results_for_sample_sets[0]['mean-qoi'][:, None] ]
    dys_rom = [ rom_results_for_sample_sets[0]['qois'] - rom_results_for_sample_sets[0]['mean-qoi'][:, None] ,  rom_results_for_sample_sets[1]['qois'] - rom_results_for_sample_sets[1]['mean-qoi'][:, None] ] 

    # Covariance of total variate
 
    C = dys_fom[0] @ dys_fom[0].transpose() 
    C += 0.25 * (dys_rom[0] @ dys_rom[0].transpose() )
    C -= 0.5*( dys_fom[0] @ dys_rom[0].transpose())
    C -= 0.5*( dys_rom[0] @ dys_fom[0].transpose())
    C = C * ( 1. / (ensemble_sizes[0] - 1) )
    C += 0.25* (dys_rom[1] @ dys_rom[1].transpose()) * 1. / (ensemble_sizes[1] - 1) 

    dws = [  (parameter_sample_sets[0] - np.mean(parameter_sample_sets[0], axis=0)[None]).transpose() , (parameter_sample_sets[1] - np.mean(parameter_sample_sets[1], axis=0)[None]).transpose() ]

    # Compute parameter covariance
    C_p =  dws[0] @ dys_fom[0].transpose() 
    C_p += 0.25*dws[0] @ dys_rom[0].transpose() 
    C_p -= 0.5*dws[0] @ dys_rom[0].transpose() 
    C_p -= 0.5*dws[0] @ dys_fom[0].transpose() 
    C_p = C_p * ( 1. / (ensemble_sizes[0] - 1) )
    C_p += 0.25 * dws[1] @ dys_rom[1].transpose()  * 1. / (ensemble_sizes[1] - 1)

    # Compute Kalman gain
    I = np.eye(observations_covariance.shape[1])
    LHS = C + observations_covariance + regularization_parameter * I
    RHS = fom_results_for_sample_sets[0]['errors']
    dp = np.linalg.solve(LHS, RHS)
    dp = C_p @ dp

    # Compute update for secondary parameter set
    RHS = rom_results_for_sample_sets[1]['errors']
    dpr = np.linalg.solve(LHS, RHS)
    dpr = C_p @ dpr
   
    dps = [dp.transpose(),dpr.transpose()] 
    return dps
