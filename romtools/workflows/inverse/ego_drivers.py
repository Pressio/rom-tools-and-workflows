"""
Single-fidelity efficient global optimization drivers.

This module provides an efficient global optimization (EGO)
workflow for black-box forward models. The algorithm fits model outputs
with a Gaussian Processes, then uses an 'expected improvement' metric
to determine the next point to sample. 'Expected improvement' balances
exploration of a design space with exploitation of know minima of a function. 
"""

import numpy as np
import os
import concurrent.futures
import multiprocessing
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows.inverse._inverse_utils import *
from romtools.workflows.inverse.ego_optimization_methods import *
from romtools.rom.qoi_surrogates import *


def run_ego(model: QoiModel,
                 parameter_space: ParameterSpace,
                 observations: np.ndarray,
                 number_of_iterations: int,
                 parameter_mins: np.ndarray = None,
                 parameter_maxes: np.ndarray = None,
                 absolute_ego_directory: str = os.getcwd() + "/work/",
                 number_initial_samples: int=4,
                 random_seed: int = None,
                 use_relative_error: bool = True,
                 restart_file = None,
                 expected_improvement_epsilon: float=0.0):
    """
    Run a single-fidelity efficient global optimization

    Args:
        model: QoiModel to evaluate at ensemble samples.
        parameter_space: ParameterSpace used to draw the initial ensemble when
            ``restart_file`` is not provided.
        observations: Observed QoI vector :math:`y`.
        parameter_mins: Optional lower bounds applied to sampled and updated
            parameters.
        parameter_maxes: Optional upper bounds applied to sampled and updated
            parameters.
        absolute_ego_directory: Absolute path to the working directory. Each
            accepted or tested iteration writes into
            ``iteration_<k>/run_*`` subdirectories under this path.
        number_initial_samples: Optional number of model samples to train the 
            initial Gaussian process. Default is 4. 
        random_seed: Optional seed to fix random sampling. Default is None.
        use_relative_error: Optional boolean to use relative error with respect
            to observations as the objective runtion. Default is None.
        restart_file: Optional ``.npz`` restart file produced by a prior EGO
            run. When set, the saved samples and QoIs are restored instead of 
            drawing a new sample.
        expected_improvement_epsilon: Optional parameter for expected improvement.
            Values greater than zero will promote design space exploration

    Returns:
        Tuple ``(parameter_sample_min, obj_min, qoi_min)`` containing the final input parameters and
        the corresponding minimum objective function and QoI as of the last iteration.
    """

    start_time = time.time()
    # check that relative error is well-posed:
    if use_relative_error:
        assert(np.linalg.norm(observations) > 0)

    # Initial design point(s)
    if restart_file is None:
        parameter_samples = parameter_space.generate_samples(number_initial_samples, seed=random_seed)
        parameter_samples = bound_samples(parameter_samples,parameter_mins,parameter_maxes)
        parameter_names = parameter_space.get_names()

        qois = []
        errors = []
        objs = []
        # run model at samples
        iteration = 0
        run_directory_base = f'{absolute_ego_directory}/iteration_{0}/run_'
        for initial_sample in range(number_initial_samples):
            run_directory = f'{run_directory_base}{initial_sample}'
            qoi, error, _ = prepare_and_run(model, observations, run_directory, parameter_names, parameter_samples[initial_sample])
            obj = objective_function(qoi,observations,relative=use_relative_error)
            qois.append(qoi)
            errors.append(error)
            objs.append(obj)
        qois = np.array(qois)
        errors = np.array(errors)
        objs = np.array(objs)
    else:
        restart_file = np.load(restart_file)
        parameter_samples = restart_file['parameter_samples']
        iteration = restart_file['iteration']
        parameter_names = parameter_space.get_names()
        run_directory_base = f'{absolute_ego_directory}/iteration_{iteration}/run_'
        qois = restart_file['qois']
        errors = restart_file['errors']
        objs = restart_file['objs']

    # Optimization loop
    wall_time = time.time() - start_time
    obj_min = np.min(objs)
    parameter_sample_min = parameter_samples[np.argmin(objs)]
    print(f'Iteration: {iteration}, Minimum Normalized L2 Error: {obj_min:.5f}, Wall time: {wall_time:.5f}')

    iteration += 1
    while iteration < number_of_iterations:

        # fit GP to samples
        gp_regressor = GaussianProcessQoiModel(parameter_samples,objs,tune_hyperparameters=True)

        # determine design point that maximizes expected improvement
        parameter_sample_new = argmax_expected_improvement(gp_regressor,
                                                            obj_min,
                                                            parameter_space,
                                                            parameter_mins,
                                                            parameter_maxes,
                                                            random_seed=random_seed,
                                                            epsilon=expected_improvement_epsilon)

        # evaluate function at new design point
        run_directory = f'{absolute_ego_directory}/iteration_{iteration}/run'
        qoi_new, error_new, _ = prepare_and_run(model, observations, run_directory, parameter_names, parameter_sample_new)
        obj_new = np.array([objective_function(qoi_new, observations, relative=use_relative_error),])

        # update sample vectors
        parameter_samples = np.vstack([parameter_samples,parameter_sample_new])
        qois = np.vstack([qois,qoi_new])
        errors = np.vstack([errors,error_new])
        objs = np.concatenate([objs,obj_new])

        wall_time = time.time() - start_time
        i_min = np.argmin(objs)
        obj_min = objs[i_min]
        parameter_sample_min = parameter_samples[i_min]
        qoi_min = qois[i_min]
        print(f'Iteration: {iteration}, Minimum Normalized L2 Error: {obj_min:.5f}, Wall time: {wall_time:.5f}')
        np.savez(f'{absolute_ego_directory}/iteration_{iteration}/restart.npz',qois=qois,errors=errors,objs=objs,parameter_samples=parameter_samples,iteration=iteration)

        iteration += 1

    # return final parameter sample and qoi
    return parameter_sample_min, obj_min, qoi_min

def run_batch_ego(model: QoiModel,
                    parameter_space: ParameterSpace,
                    observations: np.ndarray,
                    number_of_iterations: int,
                    batch_size: int,
                    parameter_mins: np.ndarray = None,
                    parameter_maxes: np.ndarray = None,
                    absolute_ego_directory: str = os.getcwd() + "/work/",
                    number_initial_samples: int=4,
                    random_seed: int = None,
                    evaluation_concurrency: int=-1,
                    use_relative_error: bool = True,
                    restart_file = None,
                    expected_improvement_epsilon: float=0.0,
                    constant_liar_type: str='pessimistic'):
    """
    Run a single-fidelity batch efficient global optimization

    Args:
        model: QoiModel to evaluate at ensemble samples.
        parameter_space: ParameterSpace used to draw the initial ensemble when
            ``restart_file`` is not provided.
        observations: Observed QoI vector :math:`y`.
        number_of_iterations: Number of EGO iterations. 
        batch_size: number of function evaluations per iteration.
        parameter_mins: Optional lower bounds applied to sampled and updated
            parameters.
        parameter_maxes: Optional upper bounds applied to sampled and updated
            parameters.
        absolute_ego_directory: Absolute path to the working directory. Each
            accepted or tested iteration writes into
            ``iteration_<k>/run_*`` subdirectories under this path.
        number_initial_samples: Optional number of model samples to train the 
            initial Gaussian process. Default is 4. 
        random_seed: Optional seed to fix random sampling. Default is None.
        evaluation_concurrency: Number of concurrent model evaluations used by
            each batch EGO iteration. Default is the batch_size
        use_relative_error: Optional boolean to use relative error with respect
            to observations as the objective runtion. Default is None.
        restart_file: Optional ``.npz`` restart file produced by a prior EGO
            run. When set, the saved samples and QoIs are restored instead of 
            drawing a new sample.
        expected_improvement_epsilon: Optional parameter for expected improvement.
            Values greater than zero will promote design space exploration
        constant_liar_type: Optional string for type of constant liar aquisition
            function. Valid options are "pessimistic", "optimistic", and "average".
        

    Returns:
        Tuple ``(parameter_sample_min, obj_min, qoi_min)`` containing the final input parameters and
        the corresponding minimum objective function and QoI as of the last iteration.
    """

    start_time = time.time()
    mp_cntxt = multiprocessing.get_context("fork")

    # check that relative error is well-posed:
    if use_relative_error:
        assert(np.linalg.norm(observations) > 0)

    # if evaluation concurrency is not explicitly set, make it equal to batch_size
    if evaluation_concurrency < 0:
        evaluation_concurrency = batch_size

    # Initial design point(s)
    if restart_file is None:
        parameter_samples = parameter_space.generate_samples(number_initial_samples, seed=random_seed)
        parameter_samples = bound_samples(parameter_samples,parameter_mins,parameter_maxes)
        parameter_names = parameter_space.get_names()

        qois = []
        errors = []
        objs = []
        # run model at samples
        iteration = 0
        run_directory_base = f'{absolute_ego_directory}/iteration_0/run_'

        if evaluation_concurrency == 1:
            for initial_sample in range(number_initial_samples):
                run_directory = f'{run_directory_base}{initial_sample}'
                qoi, error, _ = prepare_and_run(model, observations, run_directory, parameter_names, parameter_samples[initial_sample])
                obj = objective_function(qoi,observations,relative=use_relative_error)
                qois.append(qoi)
                errors.append(error)
                objs.append(obj)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=evaluation_concurrency, mp_context=mp_cntxt) as executor:
                these_futures = [executor.submit(prepare_and_run, model, observations, f'{run_directory_base}{initial_sample}', parameter_names, parameter_samples[initial_sample]) for initial_sample in range(number_initial_samples)]
                concurrent.futures.wait(these_futures)
            for future in these_futures:
                qoi, error, _ = future.result()
                obj = objective_function(qoi,observations,relative=use_relative_error)
                qois.append(qoi)
                errors.append(error)
                objs.append(obj)

        qois = np.array(qois)
        errors = np.array(errors)
        objs = np.array(objs)
    else:
        restart_file = np.load(restart_file)
        parameter_samples = restart_file['parameter_samples']
        iteration = restart_file['iteration']
        parameter_names = parameter_space.get_names()
        run_directory_base = f'{absolute_ego_directory}/iteration_{iteration}/run_'
        qois = restart_file['qois']
        errors = restart_file['errors']
        objs = restart_file['objs']

    # Optimization loop
    wall_time = time.time() - start_time
    obj_min = np.min(objs)
    parameter_sample_min = parameter_samples[np.argmin(objs)]
    print(f'Iteration: {iteration}, Minimum Normalized L2 Error: {obj_min:.5f}, Wall time: {wall_time:.5f}')

    iteration += 1
    while iteration < number_of_iterations:

        # fit GP to samples
        gp_regressor = GaussianProcessQoiModel(parameter_samples,objs,tune_hyperparameters=True)

        # determine design point that maximizes expected improvement
        parameter_samples_new = q_point_expected_improvement_constant_liar(gp_regressor,
                                                                            obj_min,
                                                                            parameter_samples,
                                                                            objs,
                                                                            batch_size,
                                                                            parameter_space,
                                                                            parameter_mins,
                                                                            parameter_maxes,
                                                                            random_seed=random_seed,
                                                                            epsilon=expected_improvement_epsilon,
                                                                            liar_type=constant_liar_type)

        objs_new = np.zeros((batch_size))
        qois_new = np.zeros((batch_size,1))
        errors_new = np.zeros((batch_size,1))
        if evaluation_concurrency == 1:
            # evaluate function at new design point
            for i,parameter_sample_new in enumerate(parameter_samples_new):
                run_directory = f'{run_directory_base}{i}'
                qoi_new, error_new, _ = prepare_and_run(model, observations, run_directory, parameter_names, parameter_sample_new)
                objs_new[i] = objective_function(qoi_new, observations, relative=use_relative_error)
                qois_new[i] = qoi_new
                errors_new[i] = error_new
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=evaluation_concurrency, mp_context=mp_cntxt) as executor:
                these_futures = [executor.submit(prepare_and_run, model, observations, f'{run_directory_base}{i}', parameter_names, parameter_samples_new[i]) for i in range(batch_size)]
                concurrent.futures.wait(these_futures)
            for i,future in enumerate(these_futures):
                qoi_new, error_new, _ = future.result()
                objs_new[i] = objective_function(qoi_new,observations,relative=use_relative_error)
                qois_new[i] = qoi_new
                errors_new[i] = error_new

        # update sample vectors
        parameter_samples = np.vstack([parameter_samples,parameter_samples_new])
        qois = np.vstack([qois,qois_new])
        errors = np.vstack([errors,errors_new])
        objs = np.concatenate([objs,objs_new])

        wall_time = time.time() - start_time
        i_min = np.argmin(objs)
        obj_min = objs[i_min]
        parameter_sample_min = parameter_samples[i_min]
        qoi_min = qois[i_min]
        print(f'Iteration: {iteration}, Minimum Normalized L2 Error: {obj_min:.5f}, Wall time: {wall_time:.5f}')
        np.savez(f'{absolute_ego_directory}/iteration_{iteration}/restart.npz',qois=qois,errors=errors,objs=objs,parameter_samples=parameter_samples,iteration=iteration)

        iteration += 1

    # return final parameter sample and qoi
    return parameter_sample_min, obj_min, qoi_min