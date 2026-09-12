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

.. rubric:: Ensemble Rejuvenation

EKI can stagnate when the empirical parameter-observation cross-covariance
collapses before the observation residual is small. ``run_eki`` can rejuvenate
the ensemble by resampling around the current mean with an inflated covariance.
Adaptive rejuvenation is triggered when the proposed update is below
``delta_params_tolerance`` while the residual remains above
``error_norm_tolerance``. Periodic rejuvenation is also available.

The rejuvenation covariance is

.. math::

   C_{\mathrm{rejuv}}
   = \alpha C_{\theta,k} + \beta C_{\mathrm{ref},k},

with a parameter-scaled diagonal reference covariance

.. math::

   C_{\mathrm{ref},k}
   = \operatorname{diag}(s_1^2, \ldots, s_d^2),
   \qquad
   s_i = |\bar{\theta}_{k,i}|.

When a current parameter mean is numerically zero and finite lower and upper
bounds are available, its admissible range is used as the reference scale.
For an unbounded zero-mean parameter, the initial ensemble covariance provides
a fallback scale. The default :math:`\beta=0.0025=0.05^2` therefore produces a
5 percent standard-deviation perturbation about a nonzero current parameter
mean when :math:`C_{\theta,k}` has collapsed. Rejuvenation uses a deterministic
seed based on the rejuvenation count so restart behavior is reproducible.

.. rubric:: Step Acceptance

The routine uses a simple trust-region-like acceptance rule on top of the EKI
update. A trial step is accepted when the mean observation-space error norm is
below ``relaxation_parameter`` times the current value. Accepted steps grow the
step size by ``step_size_growth_factor``; rejected steps shrink it by
``step_size_decay_factor`` until either a step is accepted or the maximum
number of retries is reached.

.. rubric:: Practical Notes

- The forward model is treated as derivative-free; only QoI evaluations are
  required.
- Parameter bounds are enforced by clipping sampled, rejuvenated, and updated
  parameters to ``parameter_mins`` and ``parameter_maxes`` when provided.
- Restart files store the ensemble, QoIs, current step size, rejuvenation count,
  and initial covariance used only as a zero-mean fallback scale.

.. rubric:: Relation to ``run_mf_eki``

``run_eki`` is the single-fidelity baseline. The multifidelity driver
:func:`romtools.workflows.inverse.mf_eki_drivers.run_mf_eki` augments this
workflow with ROM control variates and adaptive surrogate rebuilding, but the
high-fidelity correction and step-acceptance logic remain closely related.
"""

import os
import time
from typing import Optional

import numpy as np

from romtools.hpc.dispatchers import BaseDispatcher, resolve_dispatcher
from romtools.workflows.inverse._inverse_utils import *
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace


_REJUVENATION_STRATEGIES = {"none", "adaptive", "periodic"}


def _validate_rejuvenation_settings(
        rejuvenation_strategy,
        rejuvenation_interval,
        rejuvenation_inflation,
        rejuvenation_prior_weight,
        max_rejuvenations):
    if rejuvenation_strategy not in _REJUVENATION_STRATEGIES:
        raise ValueError(
            "rejuvenation_strategy must be one of "
            f"{sorted(_REJUVENATION_STRATEGIES)}"
        )
    if rejuvenation_interval <= 0:
        raise ValueError("rejuvenation_interval must be positive")
    if rejuvenation_inflation < 1.0:
        raise ValueError("rejuvenation_inflation must be at least 1.0")
    if rejuvenation_prior_weight < 0.0:
        raise ValueError("rejuvenation_prior_weight must be nonnegative")
    if max_rejuvenations < 0:
        raise ValueError("max_rejuvenations must be nonnegative")


def _compute_ensemble_covariance(parameter_samples):
    """Return the empirical covariance of an ensemble with samples in rows."""
    parameter_samples = np.asarray(parameter_samples, dtype=float)
    if parameter_samples.ndim != 2:
        raise ValueError("parameter_samples must be a two-dimensional array")
    ensemble_size = parameter_samples.shape[0]
    if ensemble_size < 2:
        raise ValueError("At least two ensemble members are required")
    anomalies = parameter_samples - np.mean(parameter_samples, axis=0)[None, :]
    return anomalies.T @ anomalies / (ensemble_size - 1)


def _compute_ensemble_spread(parameter_samples):
    """Return the root-sum variance of an ensemble."""
    covariance = _compute_ensemble_covariance(parameter_samples)
    return float(np.sqrt(max(np.trace(covariance), 0.0)))


def _compute_parameter_scaled_reference_covariance(
        parameter_samples,
        parameter_mins=None,
        parameter_maxes=None,
        fallback_covariance=None):
    """Build a diagonal reference covariance from current parameter scales."""
    parameter_samples = np.asarray(parameter_samples, dtype=float)
    mean_magnitudes = np.abs(np.mean(parameter_samples, axis=0))
    scales = mean_magnitudes.copy()

    if parameter_mins is not None and parameter_maxes is not None:
        parameter_mins = np.asarray(parameter_mins, dtype=float)
        parameter_maxes = np.asarray(parameter_maxes, dtype=float)
        parameter_ranges = np.maximum(parameter_maxes - parameter_mins, 0.0)
        zero_tolerance = (
            np.sqrt(np.finfo(float).eps)
            * np.maximum(parameter_ranges, np.finfo(float).tiny)
        )
        use_range = scales <= zero_tolerance
        scales[use_range] = parameter_ranges[use_range]

    if fallback_covariance is not None:
        fallback_covariance = np.asarray(fallback_covariance, dtype=float)
        if fallback_covariance.shape != (scales.size, scales.size):
            raise ValueError(
                "fallback_covariance shape must match the parameter dimension"
            )
        fallback_scales = np.sqrt(
            np.maximum(np.diag(fallback_covariance), 0.0)
        )
        zero_tolerance = (
            np.sqrt(np.finfo(float).eps)
            * np.maximum(fallback_scales, np.finfo(float).tiny)
        )
        use_fallback = scales <= zero_tolerance
        scales[use_fallback] = fallback_scales[use_fallback]

    return np.diag(scales**2)


def _build_rejuvenation_covariance(
        parameter_samples,
        reference_covariance,
        rejuvenation_inflation,
        rejuvenation_prior_weight,
        parameter_mins=None,
        parameter_maxes=None):
    current_covariance = _compute_ensemble_covariance(parameter_samples)
    parameter_scaled_reference_covariance = (
        _compute_parameter_scaled_reference_covariance(
            parameter_samples,
            parameter_mins,
            parameter_maxes,
            fallback_covariance=reference_covariance,
        )
    )
    covariance = (
        rejuvenation_inflation * current_covariance
        + rejuvenation_prior_weight * parameter_scaled_reference_covariance
    )
    return 0.5 * (covariance + covariance.T)


def _rejuvenate_parameter_samples(
        parameter_samples,
        reference_covariance,
        rejuvenation_inflation,
        rejuvenation_prior_weight,
        random_seed,
        parameter_mins=None,
        parameter_maxes=None):
    """Resample around the current mean using an inflated covariance."""
    parameter_samples = np.asarray(parameter_samples, dtype=float)
    mean = np.mean(parameter_samples, axis=0)
    covariance = _build_rejuvenation_covariance(
        parameter_samples,
        reference_covariance,
        rejuvenation_inflation,
        rejuvenation_prior_weight,
        parameter_mins,
        parameter_maxes,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    covariance_sqrt = (
        eigenvectors
        @ np.diag(np.sqrt(eigenvalues))
        @ eigenvectors.T
    )

    rng = np.random.default_rng(random_seed)
    standardized = rng.standard_normal(parameter_samples.shape)
    standardized -= np.mean(standardized, axis=0)[None, :]
    rejuvenated_samples = mean[None, :] + standardized @ covariance_sqrt.T
    return bound_samples(
        rejuvenated_samples,
        parameter_mins,
        parameter_maxes,
    )


def _should_rejuvenate(
        rejuvenation_strategy,
        iteration,
        dp_norm,
        error_norm,
        delta_params_tolerance,
        error_norm_tolerance,
        rejuvenation_count,
        max_rejuvenations,
        rejuvenation_interval):
    if rejuvenation_strategy == "none" or rejuvenation_count >= max_rejuvenations:
        return False
    if error_norm <= error_norm_tolerance:
        return False
    if rejuvenation_strategy == "adaptive":
        return dp_norm <= delta_params_tolerance
    return iteration > 0 and iteration % rejuvenation_interval == 0


def _run_eki_rejuvenation(
        model,
        observations,
        observations_covariance,
        absolute_eki_directory,
        iteration,
        parameter_names,
        parameter_samples,
        reference_covariance,
        rejuvenation_count,
        rejuvenation_inflation,
        rejuvenation_prior_weight,
        random_seed,
        parameter_mins,
        parameter_maxes,
        evaluation_concurrency,
        regularization_parameter,
        dispatcher):
    """Rejuvenate, reevaluate, and recompute the EKI update."""
    event_seed = random_seed + rejuvenation_count + 1
    parameter_samples = _rejuvenate_parameter_samples(
        parameter_samples,
        reference_covariance,
        rejuvenation_inflation,
        rejuvenation_prior_weight,
        event_seed,
        parameter_mins,
        parameter_maxes,
    )
    run_directory_base = (
        f'{absolute_eki_directory}/iteration_{iteration}/'
        f'rejuvenation_{rejuvenation_count}/run_'
    )
    results = run_eki_iteration(
        model,
        observations,
        run_directory_base,
        parameter_names,
        parameter_samples,
        evaluation_concurrency,
        dispatcher,
    )
    qois = results['qois']
    mean_qoi = results['mean-qoi']
    errors = results['errors']
    error_norm = np.mean(np.linalg.norm(errors, axis=0))
    dp = compute_eki_update(
        parameter_samples,
        qois,
        mean_qoi,
        errors,
        observations_covariance,
        regularization_parameter,
    )
    dp_norm = np.linalg.norm(dp)
    rejuvenation_count += 1
    spread = _compute_ensemble_spread(parameter_samples)
    print(
        f'  Rejuvenated ensemble {rejuvenation_count}: '
        f'Error 2-norm: {error_norm:.5f}, Delta p: {dp_norm:.5f}, '
        f'Ensemble spread: {spread:.5f}'
    )
    return (
        parameter_samples,
        qois,
        mean_qoi,
        errors,
        error_norm,
        dp,
        dp_norm,
        rejuvenation_count,
    )


def _save_eki_restart(
        dispatcher,
        restart_path,
        qois,
        mean_qoi,
        errors,
        parameter_samples,
        iteration,
        step_size,
        rejuvenation_reference_covariance,
        rejuvenation_count):
    dispatcher.np_savez(
        restart_path,
        qois=qois,
        mean_qoi=mean_qoi,
        errors=errors,
        parameter_samples=parameter_samples,
        iteration=iteration,
        step_size=step_size,
        rejuvenation_reference_covariance=rejuvenation_reference_covariance,
        rejuvenation_count=rejuvenation_count,
    )


def run_eki(model: QoiModel,
            parameter_space: ParameterSpace,
            observations: np.ndarray,
            observations_covariance: np.ndarray,
            parameter_mins: np.ndarray = None,
            parameter_maxes: np.ndarray = None,
            absolute_eki_directory: str = os.getcwd() + "/work/",
            ensemble_size: int = 30,
            initial_step_size: float = 0.05,
            regularization_parameter: float = 1e-4,
            step_size_growth_factor: float = 1.25,
            step_size_decay_factor: float = 2.0,
            max_step_size_decrease_trys: int = 5,
            relaxation_parameter: float = 1.05,
            error_norm_tolerance: float = 1e-5,
            delta_params_tolerance: float = 1e-6,
            rejuvenation_strategy: str = "none",
            rejuvenation_interval: int = 5,
            rejuvenation_inflation: float = 1.1,
            rejuvenation_prior_weight: float = 0.0025,
            max_rejuvenations: int = 3,
            max_iterations: int = 50,
            random_seed: int = 1,
            evaluation_concurrency=1,
            restart_file=None,
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
            ``iteration_<k>/run_*`` subdirectories under this path.
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
        delta_params_tolerance: Parameter-update norm below which EKI is
            considered stagnant. With adaptive rejuvenation, a large residual
            triggers rejuvenation instead of immediate termination.
        rejuvenation_strategy: ``"none"`` (default), ``"adaptive"``, or
            ``"periodic"``. Adaptive rejuvenation is triggered by a small
            update and non-converged residual.
        rejuvenation_interval: Number of accepted iterations between periodic
            rejuvenations.
        rejuvenation_inflation: Multiplicative inflation applied to the current
            ensemble covariance during rejuvenation. Must be at least 1.
        rejuvenation_prior_weight: Weight :math:`\beta` applied to the
            parameter-scaled reference covariance. The default ``0.0025`` is
            ``0.05**2``, giving a 5 percent standard-deviation perturbation
            relative to a nonzero current parameter mean when the empirical
            covariance has collapsed.
        max_rejuvenations: Maximum number of rejuvenation events.
        max_iterations: Maximum number of EKI iterations.
        random_seed: RNG seed used for the initial ensemble draw and as the
            base for deterministic rejuvenation-event seeds.
        evaluation_concurrency: Number of concurrent model evaluations used by
            each EKI iteration. Must be 1 when a RemoteDispatcher is supplied.
        restart_file: Optional ``.npz`` restart file produced by a prior EKI
            run. Rejuvenation count and fallback covariance are restored when
            available; older restart files remain supported.
        dispatcher: Optional dispatcher. ``None`` uses a LocalDispatcher.

    Returns:
        Tuple ``(parameter_samples, qois)`` containing the final ensemble and
        corresponding QoI matrix from the last accepted or rejuvenated state.
    """
    dispatcher = resolve_dispatcher(dispatcher)
    start_time = time.time()

    dispatcher.require_absolute_path(absolute_eki_directory)
    dispatcher.require_supported_concurrency(evaluation_concurrency)
    assert step_size_growth_factor > 1.0, (
        "step_size_growth_factor must be greater than 1.0"
    )
    assert step_size_decay_factor > 1.0, (
        "step_size_decay_factor must be greater than 1.0"
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
    dispatcher.create_empty_dir(absolute_eki_directory)

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
        parameter_names = parameter_space.get_names()
        run_directory_base = f'{absolute_eki_directory}/iteration_0/run_'
        results = run_eki_iteration(
            model,
            observations,
            run_directory_base,
            parameter_names,
            parameter_samples,
            evaluation_concurrency,
            dispatcher,
        )
        qois = results['qois']
        mean_qoi = results['mean-qoi']
        errors = results['errors']
        error_norm = np.mean(np.linalg.norm(errors, axis=0))
        step_size = initial_step_size
    else:
        restart_data = np.load(restart_file)
        parameter_samples = restart_data['parameter_samples']
        iteration = int(restart_data['iteration'])
        step_size = float(restart_data['step_size'])
        parameter_names = parameter_space.get_names()
        qois = restart_data['qois']
        mean_qoi = restart_data['mean_qoi']
        errors = restart_data['errors']
        error_norm = np.mean(np.linalg.norm(errors, axis=0))
        if 'rejuvenation_reference_covariance' in restart_data:
            rejuvenation_reference_covariance = restart_data[
                'rejuvenation_reference_covariance'
            ]
        else:
            rejuvenation_reference_covariance = _compute_ensemble_covariance(
                parameter_samples
            )
        rejuvenation_count = (
            int(restart_data['rejuvenation_count'])
            if 'rejuvenation_count' in restart_data
            else 0
        )

    dp = compute_eki_update(
        parameter_samples,
        qois,
        mean_qoi,
        errors,
        observations_covariance,
        regularization_parameter,
    )
    dp_norm = np.linalg.norm(dp)
    wall_time = time.time() - start_time
    spread = _compute_ensemble_spread(parameter_samples)
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
            parameter_samples,
            qois,
            mean_qoi,
            errors,
            error_norm,
            dp,
            dp_norm,
            rejuvenation_count,
        ) = _run_eki_rejuvenation(
            model,
            observations,
            observations_covariance,
            absolute_eki_directory,
            iteration,
            parameter_names,
            parameter_samples,
            rejuvenation_reference_covariance,
            rejuvenation_count,
            rejuvenation_inflation,
            rejuvenation_prior_weight,
            random_seed,
            parameter_mins,
            parameter_maxes,
            evaluation_concurrency,
            regularization_parameter,
            dispatcher,
        )

    _save_eki_restart(
        dispatcher,
        f'{absolute_eki_directory}/iteration_{iteration}/restart.npz',
        qois,
        mean_qoi,
        errors,
        parameter_samples,
        iteration,
        step_size,
        rejuvenation_reference_covariance,
        rejuvenation_count,
    )

    iteration += 1
    step_failed_counter = 0
    while iteration < max_iterations and error_norm > error_norm_tolerance:
        if dp_norm <= delta_params_tolerance:
            break

        test_parameter_samples = parameter_samples + step_size * dp
        test_parameter_samples = bound_samples(
            test_parameter_samples,
            parameter_mins,
            parameter_maxes,
        )
        run_directory_base = f'{absolute_eki_directory}/iteration_{iteration}/run_'
        test_results = run_eki_iteration(
            model,
            observations,
            run_directory_base,
            parameter_names,
            test_parameter_samples,
            evaluation_concurrency,
            dispatcher,
        )
        test_qois = test_results['qois']
        test_mean_qoi = test_results['mean-qoi']
        test_errors = test_results['errors']
        test_error_norm = np.mean(np.linalg.norm(test_errors, axis=0))

        if test_error_norm < relaxation_parameter * error_norm:
            step_failed_counter = 0
            parameter_samples = test_parameter_samples.copy()
            qois = test_qois.copy()
            mean_qoi = test_mean_qoi.copy()
            errors = test_errors.copy()
            error_norm = float(test_error_norm)
            step_size = min(step_size * step_size_growth_factor, 1.0)

            dp = compute_eki_update(
                parameter_samples,
                qois,
                mean_qoi,
                errors,
                observations_covariance,
                regularization_parameter,
            )
            dp_norm = np.linalg.norm(dp)

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
                    (
                        parameter_samples,
                        qois,
                        mean_qoi,
                        errors,
                        error_norm,
                        dp,
                        dp_norm,
                        rejuvenation_count,
                    ) = _run_eki_rejuvenation(
                        model,
                        observations,
                        observations_covariance,
                        absolute_eki_directory,
                        iteration,
                        parameter_names,
                        parameter_samples,
                        rejuvenation_reference_covariance,
                        rejuvenation_count,
                        rejuvenation_inflation,
                        rejuvenation_prior_weight,
                        random_seed,
                        parameter_mins,
                        parameter_maxes,
                        evaluation_concurrency,
                        regularization_parameter,
                        dispatcher,
                    )
                else:
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
                        (
                            parameter_samples,
                            qois,
                            mean_qoi,
                            errors,
                            error_norm,
                            dp,
                            dp_norm,
                            rejuvenation_count,
                        ) = _run_eki_rejuvenation(
                            model,
                            observations,
                            observations_covariance,
                            absolute_eki_directory,
                            iteration,
                            parameter_names,
                            parameter_samples,
                            rejuvenation_reference_covariance,
                            rejuvenation_count,
                            rejuvenation_inflation,
                            rejuvenation_prior_weight,
                            random_seed,
                            parameter_mins,
                            parameter_maxes,
                            evaluation_concurrency,
                            regularization_parameter,
                            dispatcher,
                        )

            wall_time = time.time() - start_time
            spread = _compute_ensemble_spread(parameter_samples)
            print(
                f'Iteration: {iteration}, Error 2-norm: {error_norm:.5f}, '
                f'Step size: {step_size:.5f}, Delta p: {dp_norm:.5f}, '
                f'Ensemble spread: {spread:.5f}, Wall time: {wall_time:.5f}'
            )
            _save_eki_restart(
                dispatcher,
                f'{absolute_eki_directory}/iteration_{iteration}/restart.npz',
                qois,
                mean_qoi,
                errors,
                parameter_samples,
                iteration,
                step_size,
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
                'Parameter update stagnated above the residual tolerance and '
                'the maximum number of rejuvenations was reached.'
            )
        else:
            print('Parameter update dropped below tolerance.')
    return parameter_samples, qois


def compute_eki_update2(
        parameter_samples,
        qois,
        mean_qoi,
        errors,
        observations_covariance,
        regularization_parameter):
    ensemble_size = parameter_samples.shape[0]
    Pyy = 0.0
    for i in range(ensemble_size):
        Pyy += (
            (qois[:, i] - mean_qoi)[:, None]
            @ ((qois[:, i] - mean_qoi)[:, None]).transpose()
        )
    Pyy *= 1.0 / (ensemble_size - 1)

    Pxy = 0.0
    for i in range(ensemble_size):
        Pxy += (
            (parameter_samples[i] - np.mean(parameter_samples, axis=0))[:, None]
            @ ((qois[:, i] - mean_qoi)[:, None]).transpose()
        )
    Pxy *= 1.0 / (ensemble_size - 1)

    I = np.eye(mean_qoi.size)
    LHS = Pyy + observations_covariance + regularization_parameter * I
    RHS = errors
    dp = np.linalg.solve(LHS, RHS)
    dp = Pxy @ dp
    return dp.transpose()


def compute_eki_update(
        parameter_samples,
        qois,
        mean_qoi,
        errors,
        observations_covariance,
        regularization_parameter):
    ensemble_size = parameter_samples.shape[0]
    dw = (
        parameter_samples - np.mean(parameter_samples, axis=0)[None]
    ).transpose()
    Sw = dw / np.sqrt(ensemble_size - 1)
    dy = qois - mean_qoi[:, None]
    Sy = dy / np.sqrt(ensemble_size - 1)

    SySyT = Sy @ Sy.transpose()
    I = np.eye(mean_qoi.size)
    LHS = SySyT + observations_covariance + regularization_parameter * I
    RHS = errors
    dp = np.linalg.solve(LHS, RHS)
    dp = Sw @ (Sy.transpose() @ dp)
    return dp.transpose()
