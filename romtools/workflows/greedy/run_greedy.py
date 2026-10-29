#
# ************************************************************************
#
#                         ROM Tools and Workflows
# Copyright 2019 National Technology & Engineering Solutions of Sandia,LLC
#                              (NTESS)
#
# Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
#
# ROM Tools and Workflows is licensed under BSD-3-Clause terms of use:
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Questions? Contact Eric Parish (ejparis@sandia.gov)
#
# ************************************************************************
#

'''
The greedy procedure iteratively constructs a reduced basis ROM until it reaches a desired tolerance.
The algorithm is as follows:
 1. We generate a parameter training set, :math:`\\mathcal{D}_{\\mathrm{train}}`,
    :math:`|\\mathcal{D}_{\\mathrm{train}}| = N_{\\mathrm{train}}`
 2. We select an initial sample, :math:`\\mu_1 \\in \\mathcal{D}_{\\mathrm{train}}` and set
    :math:`\\mathcal{D}_{\\mathrm{train}}=\\mathcal{D}_{\\mathrm{train}} - \\{\\mu_1\\}`
 3. We then solve the FOM to obtain the solution, :math:`\\mathbf{u}(\\mu_1)`.
 4. We select a second sample, :math:`\\mu_2 \\in \\mathcal{D}_{\\mathrm{train}}` and set
    :math:`\\mathcal{D}_{\\mathrm{train}}=\\mathcal{D}_{\\mathrm{train}} - \\{\\mu_2\\}`
 4. We then solve the FOM to obtain the solution, :math:`\\mathbf{u}(\\mu_2)`.
 5. We employ the first two solutions to compute the trial space, e.g.,

    .. math::

       \\boldsymbol \\Phi = \\mathrm{orthogonalize}(\\mathbf{u}(\\mu_2)) - \\mathbf{u}_{\\mathrm{shift}}, \\\\
       \\mathbf{u}_{\\mathrm{shift}} = \\mathbf{u}(\\mu_1)

 6. We then solve the resulting ROM for the remaining parameter samples :math:`\\mathcal{D}_{\\mathrm{train}}` to
    generate approximate solutions :math:`\\mathbf{u}(\\mu), \\mu \\in \\mathcal{D}_{\\mathrm{train}}`
 7. For each ROM solution :math:`\\mathbf{u}(\\mu), \\mu \\in \\mathcal{D}_{\\mathrm{train}}` we compute an error
    estimate, :math:`e(\\mu)`
 8. If the maximum error estimate is less than some tolerance, we exit the algorithm. If not, we:
   - Set :math:`\\mu^* = \\underset{ \\mu \\in \\mathcal{D}_{\\mathrm{train}} }{ \\mathrm{arg\\; max} } \\; e(\\mu)`
   - Remove :math:`\\mu^{*}` from the training set,
     :math:`\\mathcal{D}_{\\mathrm{train}}=\\mathcal{D}_{\\mathrm{train}} - \\{\\mu^{*}\\}`
   - Solve the FOM for :math:`\\mathbf{u}(\\mu^{*})`
   - Set the reduced basis to be

     .. math::

        \\boldsymbol \\Phi = \\mathrm{orthogonalize}(\\boldsymbol \\Phi, \\mathbf{u}(\\mu^{*})) - \\mathbf{u}_{\\mathrm{shift}}
   - Go back to step 6 and continue until convergence.

This function implements the basic greedy workflow. In addition, we enable an adaptive
error estimate based on a QoI. This is based on the fact that, throughout the greedy algorithm, we have a
history of error indicators as well as a set of FOM and ROM runs for the same parameter instance. We leverage this data
to improve the error estimate. Specifically, for the max error estimates over the first :math:`j` iterations,
:math:`e_1,\\ldots,e_j`, we additionally can access QoI errors :math:`e^q_1,\\ldots,e^q_j` from our FOM-ROM runs, and we
define a scaling based on

.. math::

   C = \\frac{\\sum_{i=1}^j | e^q_j| }{ \\sum_{i=1}^j | e_j | }.

The error at the :math:`j+1` iteration can be approximated by :math:`C e(\\mu)`.
This will adaptively scale the error estimate to match the QoI error as
closely as possible, which can be helpful for defining exit criterion.

Alternatively, if `calibrated_error` is set to `False`, then the scaling factor :math:`C` will always be :math:`1`.
'''

import time
import warnings
from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Tuple
import numpy as np

import romtools.linalg.linalg as la
from romtools.workflows.models import QoiModel, QoiModelWithErrorEstimate
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows.workflow_utils import create_empty_dir
from romtools.workflows.model_builders import QoiModelBuilderWithTrainingData


__all__ = ["GreedyResult", "run_greedy", "QoIvsErrorIndicatorRegressor"]


@dataclass
class GreedyResult:
    """Final ROM, aligned training data, and completed enrichment histories.

    Error histories describe the ROMs before their corresponding enrichments,
    rather than independent validation of the final ROM. ``sample_counts``
    includes the initial build and every subsequent rebuild. Array fields are
    independent copies of workflow state; the final ROM is returned directly.
    """

    rom_model: QoiModelWithErrorEstimate
    training_parameters: np.ndarray
    training_qois: np.ndarray
    training_dirs: Tuple[str, ...]
    training_sample_indices: np.ndarray
    max_error_indicators: np.ndarray
    qoi_errors: np.ndarray
    predicted_qoi_errors: np.ndarray
    sample_counts: np.ndarray
    converged: bool
    termination_reason: str

    @property
    def iterations(self) -> int:
        """Number of completed enrichments after the initial build."""
        return len(self.qoi_errors)


def _create_parameter_dict(parameter_names, parameter_values):
    return dict(zip(parameter_names, parameter_values))


def _compute_qoi(model, run_directory, parameter_dict):
    """Keep one independent, flattened QoI row per completed FOM run."""
    return np.asarray(
        model.compute_qoi(run_directory, parameter_dict),
    ).reshape(-1).copy()


def _exclude_training_samples(parameters, candidates, training):
    """Exclude exact matches to explicitly supplied/accumulated training rows."""
    return np.asarray([
        index for index in candidates
        if not any(np.array_equal(parameters[index], parameters[i]) for i in training)
    ], dtype=int)


def run_greedy(fom_model: QoiModel,
               rom_model_builder: QoiModelBuilderWithTrainingData,
               parameter_space: ParameterSpace,
               absolute_work_dir: str = None,
               tolerance: float = None,
               testing_sample_size: int = 10,
               random_seed: int = 1,
               calibrated_error: bool=True,
               *,
               initial_sample_size: int = 2,
               max_iterations: int = 100,
               initial_samples: Optional[np.ndarray] = None,
               create_rom_run_directories: bool = True,
               absolute_greedy_work_directory: str = None) -> GreedyResult:
    '''
    Train a ROM using adaptive error-indicator-based sample selection.

    The builder receives ``offline_data_dir``, ``training_data_dirs``,
    ``training_parameters`` of shape (N, parameter_dimension), and
    ``training_qois`` of shape (N, qoi_dimension), all in matching training
    order. Every rebuild includes all completed training FOM runs. Scalars
    become one-component QoI rows; array QoIs are flattened. Copies of the
    training arrays are passed so builders cannot modify the sampling pool
    or accumulated QoI history.

    The returned ROM must implement ``compute_error_estimate`` in addition
    to the QoiModel methods. Directory-only builders must be adapted to the
    four-argument QoiModelBuilderWithTrainingData interface.

    ``initial_sample_size`` selects the first training rows from the initial
    candidate pool (default two). ``max_iterations`` bounds the number of
    subsequent greedy FOM enrichments and defaults to 100; zero builds only
    the initial ROM. Without explicit initial samples, ``testing_sample_size``
    must exceed ``initial_sample_size`` for enrichment so there is an
    untrained candidate.
    Calibration and convergence checks retain their existing behavior: the
    first convergence check follows one enrichment. Run directories are
    populated once, immediately before their model evaluation.

    ``initial_samples``, if supplied, is a nonempty (N, parameter_dimension)
    array of parameter rows to simulate before adaptation. Its row count
    overrides ``initial_sample_size``; ``testing_sample_size`` then controls
    a separate adaptation candidate pool. Exact matches to accumulated
    training rows are excluded from this pool. No domain projection or reuse
    of existing FOM outputs is performed.

    ``create_rom_run_directories=False`` skips only per-evaluation ROM run
    directory creation. Model methods still receive their nominal paths and
    must support in-memory execution. FOM, offline-data, log, and statistics
    directories remain available. Returns the final ROM and training history
    in a ``GreedyResult``, including for initial-only and limit exits.
    '''
    if absolute_greedy_work_directory is not None:
        warnings.warn(
            "'absolute_greedy_work_directory' is deprecated; use 'absolute_work_dir' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if absolute_work_dir is not None:
            raise TypeError("Specify only 'absolute_work_dir', not both directory arguments.")
        absolute_work_dir = absolute_greedy_work_directory
    if absolute_work_dir is None:
        raise TypeError("'absolute_work_dir' is required")
    if tolerance is None:
        raise TypeError("'tolerance' is required")
    if not isinstance(create_rom_run_directories, bool):
        raise TypeError("create_rom_run_directories must be a bool")
    parameter_names = list(parameter_space.get_names())
    supplied_initial_samples = initial_samples is not None
    if supplied_initial_samples:
        initial_samples = np.asarray(initial_samples).copy()
        if (initial_samples.ndim != 2 or initial_samples.shape[0] == 0
                or initial_samples.shape[1] != len(parameter_names)):
            raise ValueError("initial_samples must be nonempty and have shape (N, parameter_dimension)")
        if initial_samples.dtype.kind in "biufc" and not np.all(np.isfinite(initial_samples)):
            raise ValueError("initial_samples must contain finite numeric values")
        initial_sample_size = len(initial_samples)
    for name, value, minimum in (
        ("initial_sample_size", initial_sample_size, 1),
        ("testing_sample_size", testing_sample_size, 1),
        ("max_iterations", max_iterations, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            qualifier = "nonnegative" if minimum == 0 else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer")
    if not supplied_initial_samples and initial_sample_size > testing_sample_size:
        raise ValueError("initial_sample_size must not exceed testing_sample_size")
    if not supplied_initial_samples and max_iterations > 0 and initial_sample_size == testing_sample_size:
        raise ValueError("testing_sample_size must exceed initial_sample_size for greedy enrichment")

    np.random.seed(random_seed)
    if supplied_initial_samples and max_iterations == 0:
        parameter_samples = initial_samples
    else:
        parameter_samples = np.asarray(parameter_space.generate_samples(testing_sample_size))
        if parameter_samples.ndim != 2 or parameter_samples.shape[1] != len(parameter_names):
            raise ValueError("generated samples must have shape (N, parameter_dimension)")
        if parameter_samples.shape[0] != testing_sample_size:
            raise ValueError("parameter_space must generate testing_sample_size candidate rows")
        if supplied_initial_samples:
            parameter_samples = np.concatenate((initial_samples, parameter_samples), axis=0)
    training_samples = np.arange(initial_sample_size)
    samples_left = np.arange(initial_sample_size, len(parameter_samples))
    if supplied_initial_samples:
        samples_left = _exclude_training_samples(parameter_samples, samples_left, training_samples)

    greedy_directory = absolute_work_dir
    create_empty_dir(greedy_directory)
    offline_directory_prefix = 'offline_data'

    run_directory_prefix = "run_"
    greedy_file = open(f"{greedy_directory}/greedy_status.log", "w", encoding="utf-8") # pylint: disable=consider-using-with
    greedy_file.write("Greedy reduced basis status \n")
    fom_time = 0.
    rom_time = 0.
    basis_time = 0.

    # Run FOM training cases
    t0 = time.time()
    training_dirs = []
    initial_fom_qois = []

    for i in training_samples:
        greedy_file.write(f"Running FOM sample {i} \n")
        sample_index = i
        parameter_dict = _create_parameter_dict(parameter_names,parameter_samples[sample_index])
        fom_run_directory = f'{greedy_directory}/fom/{run_directory_prefix}{sample_index}'
        create_empty_dir(fom_run_directory)
        fom_model.populate_run_directory(fom_run_directory,parameter_dict)
        fom_model.run_model(fom_run_directory,parameter_dict)
        fom_qoi = _compute_qoi(fom_model, fom_run_directory, parameter_dict)
        initial_fom_qois.append(fom_qoi)
        training_dirs.append(fom_run_directory)
    fom_qois = np.stack(initial_fom_qois)
    fom_time += time.time() - t0

    # Create ROM bases
    t0 = time.time()
    greedy_file.write("Creating ROM bases \n")

    outer_loop_counter = 0
    updated_offline_data_dir = f'{greedy_directory}/rom_iteration_{outer_loop_counter}/{offline_directory_prefix}/'

    create_empty_dir(updated_offline_data_dir)
    rom_model = rom_model_builder.build_from_training_dirs(
        updated_offline_data_dir, list(training_dirs),
        parameter_samples[training_samples], fom_qois.copy(),
    )

    basis_time += time.time() - t0

    converged = False
    max_error_indicators = np.zeros(0)
    reg = QoIvsErrorIndicatorRegressor(calibrated_error)
    qoi_errors = np.zeros(0)
    predicted_qoi_errors = np.zeros(0)
    greedy_file.flush()

    def save_stats(termination_reason):
        np.savez(f'{greedy_directory}/greedy_stats',
                 max_error_indicators=max_error_indicators,
                 qoi_errors=qoi_errors,
                 predicted_qoi_errors=predicted_qoi_errors,
                 training_samples=training_samples,
                 training_parameters=parameter_samples[training_samples],
                 training_qois=fom_qois,
                 training_dirs=np.asarray(training_dirs),
                 sample_counts=np.arange(initial_sample_size, training_samples.size + 1),
                 fom_time=fom_time,
                 rom_time=rom_time,
                 basis_time=basis_time,
                 initial_sample_size=initial_sample_size,
                 create_rom_run_directories=create_rom_run_directories,
                 max_iterations=max_iterations,
                 iterations=training_samples.size - initial_sample_size,
                 converged=converged,
                 termination_reason=termination_reason)

    save_stats("running")
    termination_reason = "max_iterations"
    while converged is False and outer_loop_counter < max_iterations:
        if samples_left.size == 0:
            termination_reason = "candidate_pool_exhausted"
            break
        print(f'Greedy iteration # {outer_loop_counter}')
        error_indicators = np.zeros(samples_left.size)

        t0 = time.time()
        greedy_file.write(f"Greedy iteration # {outer_loop_counter}\n" +
                          f"Parameter samples: \n {parameter_samples}\n")
        greedy_file.flush()
        for counter, sample_index in enumerate(samples_left):
            greedy_file.write(f"    Running ROM sample {sample_index}\n")
            greedy_file.flush()
            parameter_dict = _create_parameter_dict(parameter_names,parameter_samples[sample_index])
            rom_run_directory = f'{greedy_directory}/rom_iteration_{outer_loop_counter}/{run_directory_prefix}{sample_index}'
            if create_rom_run_directories:
                create_empty_dir(rom_run_directory)
            rom_model.populate_run_directory(rom_run_directory,parameter_dict)
            rom_model.run_model(rom_run_directory,parameter_dict)
            error_indicators[counter] = rom_model.compute_error_estimate(rom_run_directory,parameter_dict)

        rom_time += time.time() - t0

        sample_with_highest_error_indicator = samples_left[la.argmax(error_indicators)]
        max_error_indicator = np.amax(error_indicators)
        greedy_file.write(f"Sample {sample_with_highest_error_indicator}"
                          " had the highest error indicator of"
                          f" {max_error_indicator} \n")

        outer_loop_counter += 1
        if outer_loop_counter > 1:
            predicted_max_qoi_error = reg.predict(error_indicators[la.argmax(error_indicators)])
            greedy_file.write("Our MLEM error estimate is "
                              f"{predicted_max_qoi_error}\n")
            greedy_file.flush()
            if np.amax(predicted_max_qoi_error < tolerance):
                converged = True
                print('Run converged, max approximate qoi error'
                      f' = {predicted_max_qoi_error}')
                break
            predicted_qoi_errors = np.append(predicted_qoi_errors,
                                             predicted_max_qoi_error)

        # Histories describe completed enrichments, excluding the final
        # tolerance check that does not select another FOM training sample.
        max_error_indicators = np.append(max_error_indicators, max_error_indicator)

        t0 = time.time()
        greedy_file.write("Running FOM sample"
                          f" {sample_with_highest_error_indicator}\n")
        greedy_file.flush()

        ## Identify the sample with the highest error and run FOM
        fom_run_directory = f'{greedy_directory}/fom/{run_directory_prefix}{sample_with_highest_error_indicator}'
        parameter_dict = _create_parameter_dict(parameter_names,parameter_samples[sample_with_highest_error_indicator])
        create_empty_dir(fom_run_directory)
        fom_model.populate_run_directory(fom_run_directory, parameter_dict)
        fom_model.run_model(fom_run_directory,parameter_dict)
        fom_qoi = _compute_qoi(fom_model, fom_run_directory, parameter_dict)

        fom_time += time.time() - t0

        # Get ROM QoI to calibrate our error estimator
        outer_loop_counter_m_1 = outer_loop_counter - 1
        rom_run_directory = (
            f'{greedy_directory}/rom_iteration_{outer_loop_counter_m_1}/'
            f'{run_directory_prefix}{sample_with_highest_error_indicator}'
        )
        rom_qoi = _compute_qoi(rom_model, rom_run_directory, parameter_dict)
        qoi_error = np.linalg.norm(rom_qoi - fom_qoi) / np.linalg.norm(fom_qoi)
        greedy_file.write(f"Sample {sample_with_highest_error_indicator} had "
                          f"an error of {qoi_error}\n")
        qoi_errors = np.append(qoi_errors, qoi_error)
        reg.fit(max_error_indicators, qoi_errors)

        ## Update our samples
        training_samples = np.append(training_samples,
                                     sample_with_highest_error_indicator)
        fom_qois = np.vstack((fom_qois, fom_qoi))
        samples_left = np.delete(samples_left, la.argmax(error_indicators))

        # Update ROM basis
        t0 = time.time()
        greedy_file.write("Computing ROM bases \n")
        greedy_file.flush()
        training_dirs = [f'{greedy_directory}/fom/run_{i}' for i in training_samples]

        updated_offline_data_dir = f'{greedy_directory}/rom_iteration_{outer_loop_counter}/{offline_directory_prefix}/'
        create_empty_dir(updated_offline_data_dir)
        rom_model = rom_model_builder.build_from_training_dirs(
            updated_offline_data_dir, list(training_dirs),
            parameter_samples[training_samples], fom_qois.copy(),
        )

        basis_time += time.time() - t0

        # Replenish only if another iteration can run. Directory preparation
        # is deferred until a candidate actually needs a FOM or ROM evaluation.
        if outer_loop_counter < max_iterations:
            new_parameter_sample = parameter_space.generate_samples(1)
            new_sample_number = len(parameter_samples)
            parameter_samples = np.append(parameter_samples,
                                          new_parameter_sample, axis=0)
            samples_left = np.append(samples_left, new_sample_number)
            if supplied_initial_samples:
                samples_left = _exclude_training_samples(parameter_samples, samples_left, training_samples)
        greedy_file.flush()
        save_stats("running")

    if converged:
        termination_reason = "tolerance"
    elif termination_reason == "max_iterations":
        print(f'Maximum greedy iterations ({max_iterations}) reached')
    else:
        print('No untrained samples remain in the greedy candidate pool')
    save_stats(termination_reason)
    greedy_file.write(f"Stopped: {termination_reason}\n")
    greedy_file.close()
    return GreedyResult(
        rom_model=rom_model,
        training_parameters=parameter_samples[training_samples].copy(),
        training_qois=fom_qois.copy(),
        training_dirs=tuple(training_dirs),
        training_sample_indices=training_samples.copy(),
        max_error_indicators=max_error_indicators.copy(),
        qoi_errors=qoi_errors.copy(),
        predicted_qoi_errors=predicted_qoi_errors.copy(),
        sample_counts=np.arange(initial_sample_size, training_samples.size + 1),
        converged=converged,
        termination_reason=termination_reason,
    )


class QoIvsErrorIndicatorRegressor:
    '''
    Regressor for modeling the relationship between QoI error and error
    indicator.

    This class provides a simple linear regression model for estimating the
    scaling factor (c) between QoI error and the error indicator.
    '''
    def __init__(self, calibrated_error: bool=True):
        '''
        Initializes the regressor with a default scaling factor of 1.0.
        '''
        self.__c = 1.
        self.__calibrated_error = calibrated_error

    def fit(self, x, y):
        '''
        If calibrated_error was specified, this function fits the regression
        model to the provided data points (x, y) to estimate the scaling factor.

        Args:
            x: Error indicator values.
            y: Corresponding QoI error values.
        '''
        if self.__calibrated_error:
            self.__c = np.mean(y) / np.mean(x)

    def predict(self, x):
        '''
        Predicts QoI error based on the error indicator using the estimated
        scaling factor.

        Args:
            x: Error indicator value(s) for prediction.
        '''
        return self.__c*x
