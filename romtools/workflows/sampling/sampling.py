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

import os
import time
import numpy as np
import concurrent.futures
import multiprocessing

from romtools.hpc.dispatcher_base import DispatcherBase
from romtools.hpc.local_dispatcher import LocalDispatcher

from romtools.workflows.models import Model
from romtools.workflows.parameter_spaces import ParameterSpace


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
    return qoi


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
                 absolute_sampling_directory: str,
                 evaluation_concurrency = 1,
                 number_of_samples: int = 10,
                 random_seed: int = 1,
                 dry_run: bool = False,
                 overwrite: bool = False,
                 dispatcher: DispatcherBase = None):
    '''
    Core algorithm
    '''

    # we use here spawn because the default fork causes issues with mpich,
    # see here: https://github.com/Pressio/rom-tools-and-workflows/pull/206
    #
    # to read more about fork/spawn:
    #   https://docs.python.org/3/library/multiprocessing.html#multiprocessing-start-methods
    #
    # and
    #   https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor
    #

    if dispatcher is None:
        dispatcher = LocalDispatcher()
    mp_cntxt=multiprocessing.get_context("spawn")

    np.random.seed(random_seed)

    # Create folder if it doesn't exist
    dispatcher.create_empty_dir(absolute_sampling_directory)

    # create parameter samples
    parameter_samples = parameter_space.generate_samples(number_of_samples)
    parameter_names = parameter_space.get_names()

    # Save parameter samples
    samples_file = os.path.join(absolute_sampling_directory, 'sample_parameters.txt')
    fmt = "%s "*parameter_space.get_dimensionality()
    dispatcher.np_savetxt(samples_file, parameter_samples, fmt)

    # Set up model directories
    run_directory_base = f'{absolute_sampling_directory}/run_'
    run_directories = []
    starting_sample_index = 0
    end_sample_index = starting_sample_index + parameter_samples.shape[0]
    for sample_index in range(starting_sample_index, end_sample_index):
        run_directory = f'{run_directory_base}{sample_index}'
        dispatcher.create_empty_dir(run_directory)
        parameter_dict = _create_parameter_dict(parameter_names, parameter_samples[sample_index - starting_sample_index])
        model.populate_run_directory(run_directory, parameter_dict, dispatcher)
        run_directories.append(run_directory)

    # Print MPI warnings
    print("""
    Warning: If you are using your model with MPI via a direct call to `mpirun -n ...`,
    be aware that this may or may not work for issues that are purely related to MPI.
    """)
    model_has_qoi = _model_has_compute_qoi(model)
    qoi_samples = []

    if not dry_run:
        # Run cases
        if evaluation_concurrency == 1:
            run_times = np.zeros(number_of_samples)
            for sample_index in range(0, number_of_samples):
                print("=======  Sample " + str(sample_index) + " ============")
                run_directory = f'{run_directory_base}{sample_index}'
                passed_file = os.path.join(run_directory, 'passed.txt')
                if dispatcher.path_exists(passed_file) and not overwrite:
                    print("Skipping (Sample has already run successfully)")
                    if model_has_qoi:
                        parameter_dict = _create_parameter_dict(parameter_names, parameter_samples[sample_index])
                        qoi = _compute_qoi_for_sample(model, run_directory, parameter_dict)
                        print(f"Sample {sample_index} QoI = {qoi}")
                        qoi_samples.append(qoi)
                else:
                    print("Running")
                    parameter_dict = _create_parameter_dict(parameter_names, parameter_samples[sample_index])
                    sample_result = run_sample(run_directory, model, parameter_dict, compute_qoi=model_has_qoi, dispatcher=dispatcher)
                    if model_has_qoi:
                        run_times[sample_index], qoi = sample_result
                        if qoi is not None:
                            print(f"Sample {sample_index} QoI = {qoi}")
                            qoi_samples.append(qoi)
                    else:
                        run_times[sample_index] = sample_result

                    sample_stats_save_directory = f'{run_directory_base}{sample_index}/../'
                    dispatcher.np_savez(f'{sample_stats_save_directory}/sampling_stats', run_times=run_times)
        else:
            #Identify samples to run
            samples_to_run = []
            for sample_index in range(0, number_of_samples):
                run_directory = f'{run_directory_base}{sample_index}'
                passed_file = os.path.join(run_directory, 'passed.txt')
                if dispatcher.path_exists(passed_file) and not overwrite:
                    print(f"Skipping sample {sample_index} (Sample has already run successfully)")
                    if model_has_qoi:
                        parameter_dict = _create_parameter_dict(parameter_names, parameter_samples[sample_index])
                        qoi = _compute_qoi_for_sample(model, run_directory, parameter_dict)
                        print(f"Sample {sample_index} QoI = {qoi}")
                        qoi_samples.append(qoi)
                else:
                    samples_to_run.append(sample_index)
            with concurrent.futures.ProcessPoolExecutor(max_workers = evaluation_concurrency, mp_context=mp_cntxt) as executor:
                these_futures = {
                    executor.submit(
                        run_sample,
                        f'{run_directory_base}{sample_id}',
                        model,
                        _create_parameter_dict(parameter_names, parameter_samples[sample_id]),
                        model_has_qoi,
                        dispatcher,
                    ): sample_id for sample_id in samples_to_run
                }

                # Wait for all processes to finish
                concurrent.futures.wait(these_futures.keys())

            run_times = []
            for future, sample_id in these_futures.items():
                sample_result = future.result()
                if model_has_qoi:
                    run_time, qoi = sample_result
                    if qoi is not None:
                        print(f"Sample {sample_id} QoI = {qoi}")
                        qoi_samples.append(qoi)
                else:
                    run_time = sample_result
                run_times.append(run_time)

            sample_stats_save_directory = f'{run_directory_base}{sample_index}/../'
            dispatcher.np_savez(f'{sample_stats_save_directory}/sampling_stats', run_times=run_times)

        if model_has_qoi and qoi_samples:
            qoi_stats = _compute_qoi_statistics(qoi_samples)
            sample_stats_save_directory = f'{run_directory_base}0/../'
            dispatcher.np_savez(f'{sample_stats_save_directory}/sampling_stats', run_times=run_times, **qoi_stats)
            print("QoI statistics:")
            print(f"  count: {qoi_stats['qoi_num_samples'][0]}")
            print(f"  mean: {qoi_stats['qoi_mean']}")
            print(f"  std: {qoi_stats['qoi_std']}")
            print(f"  min: {qoi_stats['qoi_min']}")
            print(f"  max: {qoi_stats['qoi_max']}")

    return run_directories


def run_sample(run_directory: str, model: Model, parameter_sample: dict, compute_qoi: bool = False, dispatcher: DispatcherBase = None):
    run_id = _get_run_id_from_run_dir(run_directory)
    ts = time.time()
    flag = model.run_model(run_directory, parameter_sample, dispatcher)
    tf = time.time()
    run_time = tf - ts
    qoi = None

    if flag == 0:
        print(f"Sample {run_id} is complete, run time = {run_time}")
        passed_file = os.path.join(run_directory, 'passed.txt')
        dispatcher.np_savetxt(passed_file, np.array([0]), '%i')
        if compute_qoi:
            qoi = _compute_qoi_for_sample(model, run_directory, parameter_sample)
    else:
        print(f"Sample {run_id} failed, run time = {run_time}")
        print(" ")
    if compute_qoi:
        return run_time, qoi
    return run_time
