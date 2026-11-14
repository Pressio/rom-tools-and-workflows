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

from functools import partial
import os
import warnings
import numpy as np
from typing import Optional

from romtools.workflows.model_execution import (
    ModelEvaluation, ModelRunError, execute_models, require_supported_concurrency,
)
from romtools.workflows.models import Model
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.hpc.dispatchers import BaseDispatcher, resolve_dispatcher


def _get_run_id_from_run_dir(run_dir):
    return int(run_dir.split('_')[-1])


def _create_parameter_dict(parameter_names, parameter_values):
    return dict(zip(parameter_names, parameter_values))


def _model_has_compute_qoi(model: Model) -> bool:
    return callable(getattr(model, "compute_qoi", None))


def _compute_qoi_for_sample(model: Model, run_directory: str, parameter_sample: dict) -> np.ndarray:
    qoi = np.asarray(model.compute_qoi(run_directory, parameter_sample))
    if qoi.ndim == 0:
        qoi = qoi.reshape(1)
    else:
        qoi = qoi.reshape(-1)
    return qoi.copy()


def _compute_qoi_statistics(qoi_samples):
    qoi_array = np.vstack(qoi_samples)
    return {
        "qoi_values": qoi_array,
        "qoi_mean": np.mean(qoi_array, axis=0),
        "qoi_std": np.std(qoi_array, axis=0),
        "qoi_min": np.min(qoi_array, axis=0),
        "qoi_max": np.max(qoi_array, axis=0),
        "qoi_num_samples": np.array([qoi_array.shape[0]], dtype=int),
    }


def run_sampling(model: Model,
                 parameter_space: ParameterSpace,
                 absolute_work_dir: Optional[str] = None,
                 evaluation_concurrency=1,
                 number_of_samples: int = 10,
                 random_seed: int = 1,
                 dry_run: bool = False,
                 overwrite: bool = False,
                 dispatcher: Optional[BaseDispatcher] = None,
                 *,
                 absolute_sampling_directory: Optional[str] = None):
    '''
    Core algorithm
    '''
    if absolute_sampling_directory is not None:
        warnings.warn(
            "'absolute_sampling_directory' is deprecated; use 'absolute_work_dir' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if absolute_work_dir is not None:
            raise TypeError("Specify only 'absolute_work_dir', not both directory arguments.")
        absolute_work_dir = absolute_sampling_directory
    if absolute_work_dir is None:
        raise TypeError("'absolute_work_dir' is required")

    dispatcher = resolve_dispatcher(dispatcher)

    require_supported_concurrency(model, dispatcher, evaluation_concurrency)

    np.random.seed(random_seed)

    # Create folder if it doesn't exist
    dispatcher.create_empty_dir(absolute_work_dir)

    # create parameter samples
    parameter_samples = parameter_space.generate_samples(number_of_samples)
    parameter_names = parameter_space.get_names()

    # Save parameter samples
    samples_file = os.path.join(absolute_work_dir, 'sample_parameters.txt')
    fmt = "%s "*parameter_space.get_dimensionality()
    dispatcher.np_savetxt(samples_file, parameter_samples, fmt)

    model_has_qoi = _model_has_compute_qoi(model)
    run_directories = []
    samples_to_run = []
    qois = [None] * len(parameter_samples)
    for sample_index, sample in enumerate(parameter_samples):
        run_directory = os.path.join(absolute_work_dir, f"run_{sample_index}")
        run_directories.append(run_directory)
        dispatcher.create_empty_dir(run_directory)
        parameter_dict = _create_parameter_dict(parameter_names, sample)
        passed_file = os.path.join(run_directory, "passed.txt")
        if dispatcher.path_exists(passed_file):
            if not overwrite:
                print(f"Skipping sample {sample_index} (Sample has already run successfully)")
                if model_has_qoi and not dry_run:
                    qois[sample_index] = _compute_qoi_for_sample(model, run_directory, parameter_dict)
                continue
            dispatcher.remove(passed_file)
        model.populate_run_directory(run_directory, parameter_dict)
        samples_to_run.append((sample_index, ModelEvaluation(run_directory, parameter_dict)))

    if dry_run:
        return run_directories

    print("""
    Warning: If you are using your model with MPI via a direct call to `mpirun -n ...`,
    be aware that this may or may not work for issues that are purely related to MPI.
    """)
    stats_path = os.path.join(absolute_work_dir, "sampling_stats")
    postprocess = partial(_finish_sample, compute_qoi=model_has_qoi, dispatcher=dispatcher)
    serial = evaluation_concurrency == 1
    if serial:
        # Yield one result at a time so we checkpoint before running the next sample.
        completed = (
            execute_models(
                model, [evaluation], evaluation_concurrency, postprocess=postprocess,
                return_exceptions=True,
            )[0]
            for _, evaluation in samples_to_run
        )
    else:
        completed = execute_models(
            model, (evaluation for _, evaluation in samples_to_run), evaluation_concurrency,
            postprocess=postprocess, return_exceptions=True,
        )
    run_times = np.zeros(len(parameter_samples) if serial else len(samples_to_run))
    for result_index, ((sample_index, _), result) in enumerate(zip(samples_to_run, completed)):
        run_time, qoi = _sample_result(sample_index, result)
        run_times[sample_index if serial else result_index] = run_time
        qois[sample_index] = qoi
        if serial:
            dispatcher.np_savez(stats_path, run_times=run_times)

    qoi_samples = [qoi for qoi in qois if qoi is not None]
    qoi_stats = _compute_qoi_statistics(qoi_samples) if qoi_samples else {}
    if qoi_stats or not serial or not samples_to_run:
        dispatcher.np_savez(stats_path, run_times=run_times, **qoi_stats)
    if qoi_stats:
        print("QoI statistics:")
        print(f"  count: {qoi_stats['qoi_num_samples'][0]}")
        print(f"  mean: {qoi_stats['qoi_mean']}")
        print(f"  std: {qoi_stats['qoi_std']}")
        print(f"  min: {qoi_stats['qoi_min']}")
        print(f"  max: {qoi_stats['qoi_max']}")

    return run_directories


def _finish_sample(model, evaluation, compute_qoi, dispatcher):
    run_directory = evaluation.run_directory
    qoi = None
    if compute_qoi:
        qoi = _compute_qoi_for_sample(model, run_directory, evaluation.parameter_sample)
    dispatcher.np_savetxt(os.path.join(run_directory, "passed.txt"), np.array([0]), "%i")
    return qoi


def _sample_result(run_id, result):
    if isinstance(result, ModelRunError):
        print(f"Sample {run_id} failed, run time = {result.run_time}")
        return result.run_time, None
    if isinstance(result, Exception):
        raise result
    print(f"Sample {run_id} is complete, run time = {result.run_time}")
    if result.output is not None:
        print(f"Sample {run_id} QoI = {result.output}")
    return result.run_time, result.output


def run_sample(run_directory: str, model: Model, parameter_sample: dict, compute_qoi: bool = False,
               dispatcher: Optional[BaseDispatcher] = None):
    dispatcher = resolve_dispatcher(dispatcher)
    result = execute_models(
        model, [ModelEvaluation(run_directory, parameter_sample)],
        postprocess=partial(_finish_sample, compute_qoi=compute_qoi, dispatcher=dispatcher),
        return_exceptions=True,
    )[0]
    run_time, qoi = _sample_result(_get_run_id_from_run_dir(run_directory), result)
    if compute_qoi:
        return run_time, qoi
    return run_time
