import numpy as np
import os
import time
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import BoundedParameterSpace
from romtools.workflows.inverse.eki_utils import *
import copy

def _create_parameter_dict(parameter_names, parameter_values):
    return dict(zip(parameter_names, parameter_values))

def run_eki(model: QoiModel,
                 parameter_space: BoundedParameterSpace,
                 absolute_enkf_directory: str,
                 observations: np.ndarray,
                 observations_covariance: np.ndarray,
                 ensemble_size: int = 30,
                 initial_step_size: float = 1e-1,
                 regularization_parameter: float = 1e-4,
                 step_size_growth_factor: float = 1.25,
                 step_size_decay_factor: float = 2.0,
                 relaxation_parameter: float = 1.05,
                 error_norm_tolerance: float = 1e-5,
                 delta_params_tolerance:  float = 1e-6,
                 max_iterations: int = 50,
                 random_seed: int = 1,
                 evaluation_concurrency = 1,
                 restart_file = None):


    start_time = time.time()
    assert os.path.isabs(absolute_enkf_directory), f"enkf_directory is not an absolute path ({absolute_enkf_directory})"
    assert step_size_growth_factor > 1.0 , "step_size_growth_factor must be greater than 1.0"
    assert step_size_decay_factor > 1.0 , "step_size_decay_factor must be greater than 1.0"

    np.random.seed(random_seed)


    # create initial samples
    if restart_file is None:
        iteration = 0
        parameter_samples = parameter_space.generate_samples(ensemble_size)
        parameter_samples = parameter_space.bound_samples(parameter_samples)
        parameter_names = parameter_space.get_names()
        #Run initial step and compute update
        run_directory_base = f'{absolute_enkf_directory}/iteration_{0}/run_'
        results = run_eki_iteration(model,observations,run_directory_base,parameter_names,parameter_samples,evaluation_concurrency)
        qois,mean_qoi,errors = results['qois'],results['mean-qoi'],results['errors']
        np.savez(f'{absolute_enkf_directory}/iteration_{0}/restart.npz',parameter_samples=parameter_samples,iteration=iteration,step_size=initial_step_size)
        error_norm = np.mean(np.linalg.norm(errors,axis=0))
        step_size = initial_step_size 
        iteration = 1

    else:
        restart_file = np.load(restart_file)
        parameter_samples = restart_file['parameter_samples']
        iteration = restart_file['iteration']
        step_size = restart_file['step_size']
        parameter_means = np.mean(parameter_samples,axis=0) 
        parameter_names = parameter_space.get_names()
        run_directory_base = f'{absolute_enkf_directory}/iteration_{iteration}/run_'
        results = run_eki_iteration(model,observations,run_directory_base,parameter_names,parameter_samples,evaluation_concurrency)
        qois,mean_qoi,errors = results['qois'],results['mean-qoi'],results['errors']
        error_norm = np.mean(np.linalg.norm(errors,axis=0))
        
    #Compute ENKF update 
    dp = compute_eki_update(parameter_samples,qois,mean_qoi,errors,observations_covariance,regularization_parameter)
    dp_norm = np.linalg.norm(dp)
    
    print(f'Initial error: {error_norm}')
    while iteration < max_iterations and error_norm > error_norm_tolerance and dp_norm > delta_params_tolerance:
        # Test the parameter update for the step size
        test_parameter_samples = parameter_samples + step_size*dp 
        test_parameter_samples = parameter_space.bound_samples(test_parameter_samples)
        run_directory_base = f'{absolute_enkf_directory}/iteration_{iteration}/run_'
        test_results = run_eki_iteration(model,observations,run_directory_base,parameter_names,test_parameter_samples,evaluation_concurrency)
        test_qois,test_mean_qoi,test_errors = test_results['qois'],test_results['mean-qoi'],test_results['errors']
        test_error_norm = np.mean(np.linalg.norm(test_errors,axis=0))
        if test_error_norm < relaxation_parameter*error_norm:
          # If error norm drops, continue the iteration and grow the step size
          parameter_samples = test_parameter_samples*1.0
          qois = test_qois*1.0
          mean_qoi = test_mean_qoi*1.0
          errors = test_errors*1.0
          error_norm = test_error_norm*1.0
          step_size = step_size*step_size_growth_factor
          wall_time = time.time() - start_time
          print(f'Iteration: {iteration}, Error 2-norm: {error_norm:.5f}, Step size: {step_size:.5f}, Delta p: {dp_norm:.5f}, Wall time: {wall_time:.5f}')
          iteration += 1
          if os.path.isdir(f'{absolute_enkf_directory}/iteration_{iteration}'):
              pass
          else:
              os.makedirs(f'{absolute_enkf_directory}/iteration_{iteration}')
          np.savez(f'{absolute_enkf_directory}/iteration_{iteration}/restart.npz',parameter_samples=parameter_samples,iteration=iteration,step_size=step_size)
          # Compute Kalman update
          dp = compute_eki_update(parameter_samples,qois,mean_qoi,errors,observations_covariance,regularization_parameter)
          dp_norm = np.linalg.norm(dp)
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


def compute_eki_update(parameter_samples,qois,mean_qoi,errors,observations_covariance,regularization_parameter):
    #Compute update matrices
    ensemble_size = parameter_samples.shape[0]
    dw = ( parameter_samples - np.mean(parameter_samples,axis=0)[None] ).transpose()
    Sw = 1./np.sqrt(ensemble_size - 1)*dw 
    dy =  qois - mean_qoi[:,None]
    Sy = 1./np.sqrt(ensemble_size - 1)*dy
    # Compute Kalmann gain
    SySyT = Sy @ Sy.transpose()
    I = np.eye(mean_qoi.size)
    LHS = SySyT + observations_covariance + regularization_parameter*I
    RHS = errors
    dp = np.linalg.solve(LHS,RHS)
    dp = Sw @ (Sy.transpose() @ dp)
    return dp.transpose()


