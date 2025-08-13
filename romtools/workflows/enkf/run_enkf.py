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

# TODO: could define observation as function for an "online" sensor
# TODO: Assuming Gaussian IID noise, could be general noise matrix
# TODO: prior really should be an Iterable[callable] to permit different sampling spaces
# TODO: discussion of what should be returned
# TODO: add timings

import os
import time
from typing import Iterable
import multiprocessing

import numpy as np

from romtools.workflows.models import QoiModel
from romtools.workflows.workflow_utils import create_empty_dir
from romtools.workflows.enkf.enkf_utils import Transformer
from romtools.workflows.enkf.enkf_utils import create_minmax_transformer, multi_transform, process_model_qois, run_model_at_ensemble

def run_enkf(
    fom_model: QoiModel,
    observation_data: np.ndarray,
    prior: callable,
    parameter_names: Iterable[str],
    parameter_mins: Iterable[float],
    parameter_maxs: Iterable[float],
    obs_transformers: Iterable[Transformer],
    obs_noise: Iterable[float],
    enkf_directory: str,
    fixed_parameters: dict = {},
    n_ensemble: int = 10,
    n_enkf_iter: int = 5,
    random_seed: int = 1,
    evaluation_concurrency = 1,
):
    '''
    Main implementation of the enkf algorithm.
    '''

    # some input checking
    assert os.path.isabs(enkf_directory), f"enkf_directory is not an absolute path ({enkf_directory})"
    n_params = len(parameter_names)
    assert len(parameter_mins) == n_params, f"Length of parameter_mins is not same as parameter_names ({n_params})"
    assert len(parameter_maxs) == n_params, f"Length of parameter_maxs is not same as parameter_names ({n_params})"
    assert observation_data.ndim == 2, "Currently assumes that observation_data is 2D"
    n_observations, n_observers = observation_data.shape
    assert len(obs_transformers) == n_observers, "Unequal observers and observation transformers"
    assert len(obs_noise) == n_observers, "Unequal number of observers and noise parameters"
    assert n_ensemble > 1
    assert n_enkf_iter > 0

    # prep outputs
    create_empty_dir(enkf_directory)
    run_directory_prefix = "run_"
    enkf_file = open(f"{enkf_directory}/enkf_status.log", "w", encoding="utf-8")

    # Init multiprocessing env
    mp_cntxt = multiprocessing.get_context("spawn")

    # Generate prior guesses of parameters
    # NOTE: currently assumes that prior generates [n_ensemble, n_params] array
    parameter_ensemble_phys = prior(n_ensemble, random_seed)
    assert parameter_ensemble_phys.shape == (n_ensemble, n_params)

    # generate parameter non-dimensionalization transformers
    param_transformers = []
    for param_min, param_max in zip(parameter_mins, parameter_maxs):
        param_transformers.append(create_minmax_transformer(param_min, param_max))
    parameter_ensemble = multi_transform(parameter_ensemble_phys, param_transformers)

    # compute mean parameter(s) from prior samples
    mean_input = np.mean(parameter_ensemble, axis=0)
    mean_input_phys = multi_transform(mean_input, param_transformers, inverse=True)

    # normalize observations
    observation_data = multi_transform(observation_data, obs_transformers)
    observation_data = observation_data.flatten(order="C")

    # set output covariance
    output_cov = np.concatenate([noise * np.ones(n_observations, dtype=np.float64) for noise in obs_noise])
    output_cov = np.diag(output_cov)

    # initialize data to collect
    input_mean_phys = [np.mean(parameter_ensemble_phys, axis=0)]
    input_norm_phys = [np.linalg.norm(parameter_ensemble_phys, axis=0)]
    input_variance_phys = [np.var(parameter_ensemble_phys, axis=0)]
    output_diff_L2 = []
    output_from_mean_input_diff_L2 = []
    timer_runtime = []
    timer_enkf    = []

    for iiter in range(n_enkf_iter):
        enkf_file.write(f"ENKF iteration {iiter}\n")

        t1 = time.time()

        # run FOM at mean input
        fom_run_directory_mean = f"{enkf_directory}/enkf_iter_{iiter}/{run_directory_prefix}mean"
        output_from_mean_input_phys = process_model_qois(
            mean_input_phys,
            parameter_names,
            fom_model,
            fom_run_directory_mean,
            fixed_parameters=fixed_parameters,
        )

        # normalize output from mean
        output_from_mean_input = multi_transform(output_from_mean_input_phys, obs_transformers)
        output_from_mean_input = output_from_mean_input.flatten(order="C")

        # run FOM ensemble member
        log_str = "Iter {iiter}: Running FOM sample "
        fom_run_dir = f"{enkf_directory}/enkf_iter_{iiter}/{run_directory_prefix}"
        ensemble_outputs, _ = run_model_at_ensemble(
            n_ensemble,
            fom_model,
            parameter_ensemble_phys,
            parameter_names,
            obs_transformers,
            fom_run_dir,
            enkf_file,
            log_str,
            evaluation_concurrency,
            mp_cntxt,
            fixed_parameters=fixed_parameters,
        )

        timer_runtime.append(time.time() - t1)

        t1 = time.time()

        # compute square root matrices
        Sin = (parameter_ensemble.T - mean_input[:, np.newaxis]) / np.sqrt(n_ensemble - 1)
        Sout = (ensemble_outputs - output_from_mean_input[:, np.newaxis]) / np.sqrt(n_ensemble - 1)

        # first and second terms of Kalman gain calculation
        K1 = Sout @ Sout.T + output_cov
        K2 = Sin @ Sout.T

        # calculate parameter update
        output_differences = observation_data[:, np.newaxis] - ensemble_outputs
        update = K2 @ np.linalg.solve(K1, output_differences)
        parameter_ensemble += update.T

        timer_enkf.append(time.time() - t1)

        parameter_ensemble_phys = multi_transform(parameter_ensemble, param_transformers, inverse=True)
        enkf_file.write(f"{parameter_ensemble_phys}\n")

        # compute mean parameter set for next iteration
        mean_input = np.mean(parameter_ensemble, axis=0)
        mean_input_phys = np.mean(parameter_ensemble_phys, axis=0)
        var_input_phys = np.var(parameter_ensemble_phys, axis=0)

        # compute output for logging
        output_from_mean_input_diff = observation_data - output_from_mean_input
        input_mean_phys.append(mean_input_phys.copy())
        input_norm_phys.append(np.linalg.norm(parameter_ensemble_phys, axis=0))
        input_variance_phys.append(var_input_phys)
        output_diff_L2.append(np.linalg.norm(output_differences, axis=0) / np.linalg.norm(observation_data, axis=0))
        output_from_mean_input_diff_L2.append(np.linalg.norm(output_from_mean_input_diff, axis=0) / np.linalg.norm(observation_data, axis=0))

        # save stats online
        np.savez(
            f"{enkf_directory}/enkf_stats",
            input_mean_phys=input_mean_phys,
            input_norm_phys=input_norm_phys,
            input_variance_phys=input_variance_phys,
            output_diff_L2=output_diff_L2,
            output_from_mean_input_diff_L2=output_from_mean_input_diff_L2,
            timer_runtime=timer_runtime,
            timer_enkf=timer_enkf,
        )

    return mean_input_phys, var_input_phys