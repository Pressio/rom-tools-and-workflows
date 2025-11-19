import numpy as np
import os
import time
from romtools.workflows.inverse.eki_utils import *

from romtools.workflows.models import QoiModel
from romtools.workflows.model_builders import QoiModelBuilder
import copy
from romtools.workflows.parameter_spaces import BoundedParameterSpace
import concurrent.futures
import multiprocessing

def run_mf_eki(model: QoiModel,
            rom_model_builder: QoiModelBuilder,
            parameter_space: BoundedParameterSpace,
            absolute_enkf_directory: str,
            observations: np.ndarray,
            observations_covariance: np.ndarray,
            fom_ensemble_size: int = 10,
            rom_extra_ensemble_size = 30,
            rom_tolerance: float = 0.01,
            initial_step_size: float = 1e-1,
            regularization_parameter: float = 1e-4,
            step_size_growth_factor: float = 1.25,
            step_size_decay_factor: float = 2.0,
            relaxation_parameter: float = 1.05,
            error_norm_tolerance: float = 1e-5,
            delta_params_tolerance: float = 1e-4,
            max_rom_training_history: int = 1,
            max_iterations: int = 50,
            random_seed: int = 1,
            fom_evaluation_concurrency: int = 1,
            rom_evaluation_concurrency: int = 1,
            restart_file: str = None):  # Optional parameter for restart file


    max_rom_training_dirs = int(max_rom_training_history*(fom_ensemble_size+1))
    start_time = time.time()
    
    # Validate input directory
    assert os.path.isabs(absolute_enkf_directory), f"enkf_directory is not an absolute path ({absolute_enkf_directory})"
    assert step_size_growth_factor > 1.0, "step_size_growth_factor must be greater than 1.0"
    assert step_size_decay_factor > 1.0, "step_size_decay_factor must be greater than 1.0"

    np.random.seed(random_seed)

    # create initial samples
    ensemble_size = fom_ensemble_size + rom_extra_ensemble_size

    if restart_file is None:
        iteration = 0
        parameter_samples = parameter_space.generate_samples(ensemble_size)
        parameter_samples = parameter_space.bound_samples(parameter_samples)
        parameter_samples_one = parameter_samples[0:fom_ensemble_size]
        parameter_samples_two = parameter_samples[fom_ensemble_size::]
        parameter_sample_sets = [parameter_samples_one,parameter_samples_two]
        parameter_names = parameter_space.get_names()
        #Run initial step and compute update
        run_directory_base = f'{absolute_enkf_directory}/iteration_{0}/run_fom_sample_set_0_'
        sample_one_fom_results = {}
        sample_one_fom_results = run_eki_iteration(model, observations, run_directory_base, parameter_names, parameter_sample_sets[0], fom_evaluation_concurrency)

        # Build ROM
        training_dirs = []
        for i in range(0,fom_ensemble_size):
          training_dirs.append(run_directory_base + str(i))
        training_dirs.append(run_directory_base + "mean")

        offline_dir = f'{absolute_enkf_directory}/iteration_{0}/'
        rom_model = rom_model_builder.build_from_training_dirs(offline_dir,training_dirs)
 
        run_directory_base = f'{absolute_enkf_directory}/iteration_{0}/run_rom_sample_set_0_'
        sample_one_rom_results = run_eki_iteration(rom_model, observations, run_directory_base, parameter_names, parameter_sample_sets[0], rom_evaluation_concurrency)

        run_directory_base = f'{absolute_enkf_directory}/iteration_{0}/run_rom_sample_set_1_'
        sample_two_rom_results = run_eki_iteration(rom_model, observations, run_directory_base, parameter_names, parameter_sample_sets[1], rom_evaluation_concurrency)

        error_norm = np.mean(np.linalg.norm(sample_one_fom_results['errors'], axis=0))
        print(f'Initial error: {error_norm}')
        step_size = initial_step_size 

        np.savez(f'{absolute_enkf_directory}/iteration_{iteration}/restart.npz', sample_one_rom_results=sample_one_rom_results,sample_two_rom_results=sample_two_rom_results,sample_one_fom_results=sample_one_fom_results,parameter_samples_one=parameter_sample_sets[0],parameter_samples_two=parameter_sample_sets[1], iteration=iteration, step_size=step_size,rom_training_directories=training_dirs)
    else:
        restart_file = np.load(restart_file,allow_pickle=True)
        parameter_samples_one = restart_file['parameter_samples_one']
        parameter_samples_two = restart_file['parameter_samples_two']
        parameter_sample_sets = [parameter_samples_one,parameter_samples_two]
        iteration = restart_file['iteration']
        step_size = restart_file['step_size']
        training_dirs = restart_file['rom_training_directories'].tolist()
        parameter_names = parameter_space.get_names()

        # Run initial step and compute update
        run_directory_base = f'{absolute_enkf_directory}/iteration_{iteration}/run_fom_sample_set_0_'
        sample_one_fom_results = restart_file['sample_one_fom_results'].item()

        offline_dir = f'{absolute_enkf_directory}/iteration_{iteration}/'
        print("==================Building ROM=============")
        rom_model = rom_model_builder.build_from_training_dirs(offline_dir,training_dirs[-max_rom_training_dirs::])
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
    while iteration < max_iterations and error_norm > error_norm_tolerance and dp_norm > delta_params_tolerance:
        # Test the parameter update for the step size
        test_parameter_sample_sets = copy.deepcopy(parameter_sample_sets)
        for i in range(len(dps)):
          test_parameter_sample_sets[i] = parameter_sample_sets[i] + step_size * dps[i] 
          test_parameter_sample_sets[i] = parameter_space.bound_samples(test_parameter_sample_sets[i])

        run_directory_base = f'{absolute_enkf_directory}/iteration_{iteration}/run_fom_sample_set_0_'
        test_training_dirs = copy.deepcopy(training_dirs)
        for i in range(0,fom_ensemble_size):
          test_training_dirs.append(run_directory_base + str(i))
        test_training_dirs.append(run_directory_base + "mean")

        # Run the EKI iteration with test parameters
        test_sample_one_fom_results = run_eki_iteration(model, observations, run_directory_base, parameter_names, test_parameter_sample_sets[0], fom_evaluation_concurrency)
        test_error_norm = np.mean(np.linalg.norm(test_sample_one_fom_results['errors'], axis=0))

        run_directory_base = f'{absolute_enkf_directory}/iteration_{iteration}/run_rom_sample_set_0_'
        test_sample_one_rom_results = run_eki_iteration(rom_model, observations, run_directory_base, parameter_names, test_parameter_sample_sets[0], rom_evaluation_concurrency)
        rom_errors = np.linalg.norm(test_sample_one_rom_results['qois'] - test_sample_one_fom_results['qois'])/ np.linalg.norm(test_sample_one_fom_results['qois'])

        if rom_errors >= rom_tolerance:
          # Build ROM
          print(f'  ROM error = {rom_errors} above tolerance, re-building ROM')
          offline_dir = f'{absolute_enkf_directory}/iteration_{iteration}/'
          rom_model = rom_model_builder.build_from_training_dirs(offline_dir,test_training_dirs[-max_rom_training_dirs::])
          test_sample_one_rom_results = run_eki_iteration(rom_model, observations, run_directory_base, parameter_names, test_parameter_sample_sets[0], rom_evaluation_concurrency)
          rom_errors = np.linalg.norm(test_sample_one_rom_results['qois'] - test_sample_one_fom_results['qois'])/ np.linalg.norm(test_sample_one_fom_results['qois'])
          print(f'  Updated ROM error = {rom_errors}')
        else:
          print(f'  ROM error = {rom_errors} below tolerance, re-using ROM')
    
        run_directory_base = f'{absolute_enkf_directory}/iteration_{iteration}/run_rom_sample_set_1_'
        test_sample_two_rom_results = run_eki_iteration(rom_model, observations, run_directory_base, parameter_names, test_parameter_sample_sets[1], rom_evaluation_concurrency)

        if test_error_norm < relaxation_parameter * error_norm:
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
            rom_sample_results = [sample_one_rom_results,sample_two_rom_results]
            dps = compute_mf_eki_update(parameter_sample_sets,fom_sample_results,rom_sample_results,observations_covariance, regularization_parameter)
            dp_norm = np.linalg.norm(dps[0])
            training_dirs = copy.deepcopy(test_training_dirs)
            print(f'Iteration: {iteration}, Error 2-norm: {error_norm:.5f}, Step size: {step_size:.5f}, Delta p: {dp_norm:.5f}, Wall time: {wall_time:.5f}')
            # Save the current state to the restart file
            np.savez(f'{absolute_enkf_directory}/iteration_{iteration}/restart.npz', sample_one_rom_results=sample_one_rom_results,sample_two_rom_results=sample_two_rom_results,sample_one_fom_results=sample_one_fom_results,parameter_samples_one=parameter_sample_sets[0],parameter_samples_two=parameter_sample_sets[1], iteration=iteration, step_size=step_size,rom_training_directories=training_dirs)
            iteration += 1
          
        else:
            # Else, drop the step size 
            step_size /= step_size_decay_factor 
            print(f'  Warning, lowering step size, Iteration: {iteration}, Error 2-norm: {error_norm:.5f}, Step size: {step_size:.5f}, Delta p: {dp_norm:.5f}')


    if iteration >= max_iterations:
        print(f'Max iterations reached, terminating')
    elif error_norm <= error_norm_tolerance:
        print(f'Error norm dropped below tolerance!')
    elif dp_norm <= delta_params_tolerance:
        print(f'Changed to parameter update dropped below tolerance!')


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


