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
                 evaluation_concurrency = 1,
                 use_relative_error: bool = True,
                 restart_file = None):
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
        absolute_eki_directory: Absolute path to the working directory. Each
            accepted or tested iteration writes into
            ``iteration_<k>/run_*`` subdirectories under this path.
        evaluation_concurrency: Number of concurrent model evaluations used by
            the initial sampling. EGO is sequential and does not support concurrency
            during iterations. 
        restart_file: Optional ``.npz`` restart file produced by a prior EGO
            run. When set, the saved samples and QoIs are restored instead of 
            drawing a new sample.

    Returns:
        Tuple ``(parameters, qois)`` containing the final input parameters and
        the corresponding QoI matrix from the last iteration.
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
                                                            random_seed=random_seed)

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

