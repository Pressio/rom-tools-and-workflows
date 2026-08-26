r"""
Single-fidelity ensemble Kalman inversion drivers.

This module provides a derivative-free ensemble Kalman inversion (EKI)
workflow for black-box forward models. The algorithm evolves an ensemble of
parameter realizations so that the corresponding model outputs become
consistent with observed quantities of interest.

.. rubric:: Theory

``run_eki`` solves a deterministic inverse problem by repeatedly updating an
ensemble :math:`\{\theta^{(j)}\}_{j=1}^{J}`. At each iteration the forward
model is evaluated on every ensemble member, producing QoIs
:math:`g(\theta^{(j)})`. The ensemble mean prediction is compared with the
observations :math:`y`, and the parameter ensemble is corrected with a
Kalman-style affine update.

With parameter anomalies

.. math::

   S_{\theta}
   =
   \frac{1}{\sqrt{J-1}}
   \left[
   \theta^{(1)} - \bar{\theta}, \ldots, \theta^{(J)} - \bar{\theta}
   \right],

and QoI anomalies

.. math::

   S_y
   =
   \frac{1}{\sqrt{J-1}}
   \left[
   g(\theta^{(1)}) - \bar{g}, \ldots, g(\theta^{(J)}) - \bar{g}
   \right],

the implementation forms the update directions by solving

.. math::

   \Delta \Theta
   =
   S_{\theta} S_y^{\top}
   \left(
   S_y S_y^{\top} + \Gamma_y + \lambda I
   \right)^{-1}
   \left[
   y - g(\theta^{(1)}), \ldots, y - g(\theta^{(J)})
   \right],

where :math:`\Gamma_y` is the observation covariance and :math:`\lambda` is
the Tikhonov regularization parameter. Each ensemble member is then updated by
adding a scaled column of :math:`\Delta \Theta`.

.. rubric:: Step Acceptance

The routine uses a simple trust-region-like acceptance rule on top of the EKI
update. A trial step is accepted only if the mean observation-space error norm
decreases by at least the factor set by ``relaxation_parameter``. Accepted
steps grow the step size by ``step_size_growth_factor``; rejected steps shrink
it by ``step_size_decay_factor`` until either a step is accepted or the
maximum number of retries is reached.

.. rubric:: Practical Notes

- The forward model is treated as derivative-free; only QoI evaluations are
  required.
- Parameter bounds are enforced by clipping sampled and updated parameters to
  ``parameter_mins`` and ``parameter_maxes`` when provided.
- Restart files store the ensemble, QoIs, mean QoI, and current step size so
  long runs can be resumed from an iteration directory.

.. rubric:: Relation to ``run_mf_eki``

``run_eki`` is the single-fidelity baseline. The multifidelity driver
:func:`romtools.workflows.inverse.mf_eki_drivers.run_mf_eki` augments this
workflow with ROM control variates and adaptive surrogate rebuilding, but the
high-fidelity correction and step-acceptance logic remain closely related.
"""

import numpy as np
import os
import time
from typing import Optional

from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows.inverse._inverse_utils import *

from romtools.hpc.dispatchers import BaseDispatcher, LocalDispatcher, resolve_dispatcher

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
                 restart_file = None,
                 dispatcher: Optional[BaseDispatcher] = None):
    """
    Run a single-fidelity ensemble Kalman inversion (EKI) workflow.

    The routine draws or restores a parameter ensemble, evaluates the forward
    model for every ensemble member, forms the ensemble Kalman correction, and
    accepts or rejects trial updates based on the reduction in mean QoI error.
    The workflow is fully derivative-free with respect to ``model``.

    Args:
        model: QoiModel to evaluate at ensemble samples.
        parameter_space: ParameterSpace used to draw the initial ensemble when
            ``restart_file`` is not provided.
        observations: Observed QoI vector :math:`y`.
        observations_covariance: Observation covariance matrix
            :math:`\Gamma_y` used in the Kalman solve.
        parameter_mins: Optional lower bounds applied to sampled and updated
            parameters.
        parameter_maxes: Optional upper bounds applied to sampled and updated
            parameters.
        absolute_eki_directory: Absolute path to the working directory. Each
            accepted or tested iteration writes into
            ``iteration_<k>/run_*`` subdirectories under this path. When a
            RemoteDispatcher is supplied, a relative path is also accepted and
            is resolved against the dispatcher's configured remote root.
        ensemble_size: Number of ensemble members used in the EKI update.
        initial_step_size: Initial multiplier applied to the computed Kalman
            update directions.
        regularization_parameter: Tikhonov regularization added to the QoI
            covariance solve for numerical stability.
        step_size_growth_factor: Factor used to increase the step size after
            an accepted iteration.
        step_size_decay_factor: Factor used to decrease the step size after a
            rejected trial iteration.
        max_step_size_decrease_trys: Maximum number of consecutive rejected
            trial steps before the routine exits.
        relaxation_parameter: Acceptance threshold on the mean error norm. A
            trial step is accepted when the new norm is below
            ``relaxation_parameter * current_error_norm``.
        error_norm_tolerance: Stop when the mean observation-space error norm
            falls below this value.
        delta_params_tolerance: Stop when the norm of the proposed ensemble
            update falls below this value.
        max_iterations: Maximum number of EKI iterations.
        random_seed: RNG seed used for the initial ensemble draw.
        evaluation_concurrency: Number of concurrent model evaluations used by
            each EKI iteration.
        restart_file: Optional ``.npz`` restart file produced by a prior EKI
            run. When set, the saved ensemble, QoIs, error state, and step
            size are restored instead of drawing a new ensemble.
        dispatcher: Optional (defaults to None, which instantiates a LocalDispatcher).
            Provide a RemoteDispatcher to send work to a remote HPC cluster.

    Returns:
        Tuple ``(parameter_samples, qois)`` containing the final ensemble and
        the corresponding QoI matrix from the last accepted iteration.
    """
    # Fall back to LocalDispatcher if none is provided
    dispatcher = resolve_dispatcher(dispatcher)

    start_time = time.time()
    ## Error checking======
    # Remote run directories are resolved against the dispatcher's remote root, so only local runs require an absolute path
    if isinstance(dispatcher, LocalDispatcher):
        assert os.path.isabs(absolute_eki_directory), f"enkf_directory is not an absolute path ({absolute_eki_directory})"
    assert step_size_growth_factor > 1.0 , "step_size_growth_factor must be greater than 1.0"
    assert step_size_decay_factor > 1.0 , "step_size_decay_factor must be greater than 1.0"
    if parameter_mins is not None:
      assert np.size(parameter_mins) == parameter_space.get_dimensionality(), f"parameter_mins of size {np.size(parameter_mins)} is inconsistent with the parameter_space of size {parameter_space.get_dimensionality()}"
    if parameter_maxes is not None:
      assert np.size(parameter_maxes) == parameter_space.get_dimensionality(), f"parameter_maxes of size {np.size(parameter_maxes)} is inconsistent with the parameter_space of size {parameter_space.get_dimensionality()}"
    ##====================
    np.random.seed(random_seed)

    # Create folder if it doesn't exist
    dispatcher.create_empty_dir(absolute_eki_directory)

    # create initial samples
    if restart_file is None:
        iteration = 0
        parameter_samples = parameter_space.generate_samples(ensemble_size)
        parameter_samples = bound_samples(parameter_samples,parameter_mins,parameter_maxes)
        parameter_names = parameter_space.get_names()
        #Run initial step and compute update
        run_directory_base = f'{absolute_eki_directory}/iteration_{0}/run_'
        results = run_eki_iteration(model,observations,run_directory_base,parameter_names,parameter_samples,evaluation_concurrency,dispatcher)
        qois,mean_qoi,errors = results['qois'],results['mean-qoi'],results['errors']
        dispatcher.np_savez(f'{absolute_eki_directory}/iteration_{0}/restart.npz',qois=qois,mean_qoi=mean_qoi,errors=errors,parameter_samples=parameter_samples,iteration=iteration,step_size=initial_step_size)
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
        test_results = run_eki_iteration(model,observations,run_directory_base,parameter_names,test_parameter_samples,evaluation_concurrency, dispatcher)
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
          dispatcher.np_savez(f'{absolute_eki_directory}/iteration_{iteration}/restart.npz',qois=qois,mean_qoi=mean_qoi,errors=errors,parameter_samples=parameter_samples,iteration=iteration,step_size=step_size)
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
