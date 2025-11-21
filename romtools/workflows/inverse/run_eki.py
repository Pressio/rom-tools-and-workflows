import numpy as np
import os
import time
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows.inverse.eki_utils import *
import copy

def _create_parameter_dict(parameter_names, parameter_values):
    return dict(zip(parameter_names, parameter_values))



def run_eki(model: QoiModel,
                 parameter_space: ParameterSpace,
                 observations: np.ndarray,
                 observations_covariance: np.ndarray,
                 parameter_mins: np.ndarray = None,
                 parameter_maxes: np.ndarray = None,
                 absolute_eki_directory: str = os.getcwd() + "/work/",
                 ensemble_size: int = 30,
                 initial_step_size: float = 1e-1,
                 regularization_parameter: float = 1e-4,
                 step_size_growth_factor: float = 1.25,
                 step_size_decay_factor: float = 2.0,
                 max_step_size_decrease_trys: int = 5,
                 relaxation_parameter: float = 1.05,
                 error_norm_tolerance: float = 1e-5,
                 delta_params_tolerance:  float = 1e-6,
                 max_iterations: int = 50,
                 random_seed: int = 1,
                 evaluation_concurrency = 1,
                 restart_file = None):



    start_time = time.time()
    ## Error checking======
    assert os.path.isabs(absolute_eki_directory), f"enkf_directory is not an absolute path ({absolute_eki_directory})"
    assert step_size_growth_factor > 1.0 , "step_size_growth_factor must be greater than 1.0"
    assert step_size_decay_factor > 1.0 , "step_size_decay_factor must be greater than 1.0"
    if parameter_mins is not None:
      assert np.size(parameter_mins) == parameter_space.get_dimensionality(), f"parameter_mins of size {np.size(parameter_mins)} is inconsistent with the parameter_space of size {parameter_space.get_dimensionality()}" 
    if parameter_maxes is not None:
      assert np.size(parameter_maxes) == parameter_space.get_dimensionality(), f"parameter_maxes of size {np.size(parameter_maxes)} is inconsistent with the parameter_space of size {parameter_space.get_dimensionality()}" 
    ##====================
    np.random.seed(random_seed)


    # create initial samples
    if restart_file is None:
        iteration = 0
        parameter_samples = parameter_space.generate_samples(ensemble_size)
        parameter_samples = bound_samples(parameter_samples,parameter_mins,parameter_maxes)
        parameter_names = parameter_space.get_names()
        #Run initial step and compute update
        run_directory_base = f'{absolute_eki_directory}/iteration_{0}/run_'
        results = run_eki_iteration(model,observations,run_directory_base,parameter_names,parameter_samples,evaluation_concurrency)
        qois,mean_qoi,errors = results['qois'],results['mean-qoi'],results['errors']
        np.savez(f'{absolute_eki_directory}/iteration_{0}/restart.npz',qois=qois,mean_qoi=mean_qoi,errors=errors,parameter_samples=parameter_samples,iteration=iteration,step_size=initial_step_size)
        error_norm = np.mean(np.linalg.norm(errors,axis=0))
        step_size = initial_step_size 

    else:
        restart_file = np.load(restart_file)
        parameter_samples = restart_file['parameter_samples']
        iteration = restart_file['iteration']
        step_size = restart_file['step_size']
        parameter_names = parameter_space.get_names()
        run_directory_base = f'{absolute_eki_directory}/iteration_{iteration}/run_'
        qois =restart_file['qois']
        mean_qoi = restart_file['mean_qoi']
        errors = restart_file['errors']
        error_norm = np.mean(np.linalg.norm(errors,axis=0))
        
    #Compute ENKF update 
    dp = compute_eki_update(parameter_samples,qois,mean_qoi,errors,observations_covariance,regularization_parameter)
    dp_norm = np.linalg.norm(dp)
    wall_time = time.time() - start_time
    print(f'Iteration: {iteration}, Error 2-norm: {error_norm:.5f}, Step size: {step_size:.5f}, Delta p: {dp_norm:.5f}, Wall time: {wall_time:.5f}')
    iteration += 1
    step_failed_counter = 0
    while iteration < max_iterations and error_norm > error_norm_tolerance and dp_norm > delta_params_tolerance:
        # Test the parameter update for the step size
        test_parameter_samples = parameter_samples + step_size*dp 
        test_parameter_samples = bound_samples(test_parameter_samples,parameter_mins,parameter_maxes)
        run_directory_base = f'{absolute_eki_directory}/iteration_{iteration}/run_'
        test_results = run_eki_iteration(model,observations,run_directory_base,parameter_names,test_parameter_samples,evaluation_concurrency)
        test_qois,test_mean_qoi,test_errors = test_results['qois'],test_results['mean-qoi'],test_results['errors']
        test_error_norm = np.mean(np.linalg.norm(test_errors,axis=0))
        if test_error_norm < relaxation_parameter*error_norm:
          step_failed_counter = 0
          # If error norm drops, continue the iteration and grow the step size
          parameter_samples = test_parameter_samples*1.0
          qois = test_qois*1.0
          mean_qoi = test_mean_qoi*1.0
          errors = test_errors*1.0
          error_norm = test_error_norm*1.0
          step_size = step_size*step_size_growth_factor
          wall_time = time.time() - start_time
          # Compute Kalman update
          dp = compute_eki_update(parameter_samples,qois,mean_qoi,errors,observations_covariance,regularization_parameter)
          dp_norm = np.linalg.norm(dp)
          print(f'Iteration: {iteration}, Error 2-norm: {error_norm:.5f}, Step size: {step_size:.5f}, Delta p: {dp_norm:.5f}, Wall time: {wall_time:.5f}')
          np.savez(f'{absolute_eki_directory}/iteration_{iteration}/restart.npz',qois=qois,mean_qoi=mean_qoi,errors=errors,parameter_samples=parameter_samples,iteration=iteration,step_size=step_size)
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
    return parameter_samples,qois


def compute_eki_update2(parameter_samples,qois,mean_qoi,errors,observations_covariance,regularization_parameter):
    #Compute update matrices
    ensemble_size = parameter_samples.shape[0]
    Pyy = 0.0
    for i in range(0,ensemble_size):
        Pyy += ( qois[:,i] - mean_qoi)[:,None] @ ((qois[:,i] - mean_qoi)[:,None]).transpose()

    Pyy *= 1./(ensemble_size - 1)

    Pxy = 0.0
    for i in range(0,ensemble_size):
        Pxy += ( parameter_samples[i] - np.mean(parameter_samples,axis=0))[:,None] @ ((qois[:,i] - mean_qoi)[:,None]).transpose()
    Pxy =  1./(ensemble_size - 1)*Pxy 

    I = np.eye(mean_qoi.size)
    LHS = Pyy + observations_covariance + regularization_parameter*I
    RHS = errors
    dp = np.linalg.solve(LHS,RHS)
    dp = Pxy @ dp 
    return dp.transpose()


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


