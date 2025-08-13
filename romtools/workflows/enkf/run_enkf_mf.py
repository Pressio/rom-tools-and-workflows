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
from typing import Iterable
import multiprocessing

import numpy as np

from romtools.workflows.models import QoiModel
from romtools.workflows.model_builders import QoiModelBuilder
from romtools.workflows.workflow_utils import create_empty_dir
from romtools.workflows.enkf.enkf_utils import Transformer
from romtools.workflows.enkf.enkf_utils import create_minmax_transformer, multi_transform, process_model_qois, run_model_at_ensemble


def run_enkf_mf(
    fom_model: QoiModel,
    rom_model_builder: QoiModelBuilder,
    observation_data: np.array,
    prior: callable,
    parameter_names: Iterable[str],
    parameter_mins: Iterable[float],
    parameter_maxs: Iterable[float],
    obs_transformers: Iterable[Transformer],
    obs_noise: Iterable[float],
    enkf_directory: str,
    rom_tol: float,
    fixed_parameters_fom: dict = {},
    fixed_parameters_rom: dict = {},
    n_ensemble_fom: int = 5,
    n_ensemble_rom: int = 15,
    n_enkf_iter: int = 5,
    random_seed: int = 1,
    evaluation_concurrency = 1,
):
    '''
    Main implementation of the MF enkf algorithm.
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
    assert rom_tol > 0.0
    assert n_ensemble_fom > 1
    assert n_ensemble_rom > 1
    assert n_enkf_iter > 0

    # Init multiprocessing env
    mp_cntxt = multiprocessing.get_context("spawn")

    # prep outputs
    create_empty_dir(enkf_directory)
    run_directory_prefix = "run_"
    offline_directory_prefix = "data_for_rom"
    enkf_file = open(f"{enkf_directory}/enkf_status.log", "w", encoding="utf-8")

    # Generate prior guess of parameters
    # NOTE: currently assumes that prior generates [n_ensemble, n_params] array
    parameter_ensemble_fom_phys = prior(n_ensemble_fom, random_seed)
    parameter_ensemble_rom_phys = prior(n_ensemble_rom, random_seed)
    assert parameter_ensemble_fom_phys.shape == (n_ensemble_fom, n_params)
    assert parameter_ensemble_rom_phys.shape == (n_ensemble_rom, n_params)

    # generate parameter non-dimensionalization transformers
    param_transformers = []
    for param_min, param_max in zip(parameter_mins, parameter_maxs):
        param_transformers.append(create_minmax_transformer(param_min, param_max))
    parameter_ensemble_fom = multi_transform(parameter_ensemble_fom_phys, param_transformers)
    parameter_ensemble_rom = multi_transform(parameter_ensemble_rom_phys, param_transformers)

    # compute mean parameter(s) from prior samples
    mean_input_fom = np.mean(parameter_ensemble_fom, axis=0)
    mean_input_rom = np.mean(parameter_ensemble_rom, axis=0)
    mean_input_fom_phys = multi_transform(mean_input_fom, param_transformers, inverse=True)
    mean_input_rom_phys = multi_transform(mean_input_rom, param_transformers, inverse=True)

    # normalize observations
    observation_data = multi_transform(observation_data, obs_transformers)
    observation_data = observation_data.flatten(order="C")

    # set output covariance
    output_cov = np.concatenate([noise * np.ones(n_observations, dtype=np.float64) for noise in obs_noise])
    output_cov = np.diag(output_cov)

    # initialize data to collect
    rom_data_indicator = []
    input_mean_phys_fom = [mean_input_fom_phys.copy()]
    input_mean_phys_rom = [mean_input_rom_phys.copy()]
    input_norm_phys_fom = [np.linalg.norm(parameter_ensemble_fom_phys, axis=0)]
    input_norm_phys_rom = [np.linalg.norm(parameter_ensemble_rom_phys, axis=0)]
    input_variance_phys_fom = [np.linalg.norm(np.var(parameter_ensemble_fom_phys, axis=0))]
    input_variance_phys_rom = [np.linalg.norm(np.var(parameter_ensemble_rom_phys, axis=0))]
    output_diff_L2_fom = []
    output_diff_L2_rom = []
    output_diff_L2_rom_fom = []
    output_from_mean_input_diff_L2_fom = []
    output_from_mean_input_diff_L2_rom_fom = []
    output_from_mean_input_diff_L2_rom = []
    timer_runtime_fom = []
    timer_runtime_rom = []
    timer_enkf        = []
    timer_training    = []

    training_dirs = []
    train_rom = True
    for iiter in range(n_enkf_iter):
        enkf_file.write(f"ENKF iteration {iiter}\n")

        ######## RUN FOM AT FOM PARAMETER INSTANCES ########

        t1 = time.time()

        # run FOM at mean input
        fom_run_directory_mean = f"{enkf_directory}/enkf_iter_{iiter}/fom/{run_directory_prefix}mean"
        output_from_mean_input_phys = process_model_qois(
            mean_input_fom_phys,
            parameter_names,
            fom_model,
            fom_run_directory_mean,
            fixed_parameters=fixed_parameters_fom,
        )

        # normalize output from mean
        output_from_mean_input_fom = multi_transform(output_from_mean_input_phys, obs_transformers)
        output_from_mean_input_fom = output_from_mean_input_fom.flatten(order="C")

        log_str_fom = "Iter {iiter}: Running FOM on FOM sample "
        fom_run_dir = f"{enkf_directory}/enkf_iter_{iiter}/fom/{run_directory_prefix}"

        ensemble_outputs_fom, training_dirs = run_model_at_ensemble(
            n_ensemble_fom,
            fom_model,
            parameter_ensemble_fom_phys,
            parameter_names,
            obs_transformers,
            fom_run_dir,
            enkf_file,
            log_str_fom,
            evaluation_concurrency,
            mp_cntxt,
            fixed_parameters=fixed_parameters_fom,
        )

        timer_runtime_fom.append(time.time() - t1)

        # Train ROM
        if train_rom:
            t1 = time.time()
            updated_offline_data_dir = f"{enkf_directory}/enkf_iter_{iiter}/rom/{offline_directory_prefix}/"
            create_empty_dir(updated_offline_data_dir)
            rom_model = rom_model_builder.build_from_training_dirs(updated_offline_data_dir, training_dirs)
            timer_training.append(time.time() - t1)

            # TODO: there has got to be a more general way to do this
            if hasattr(rom_model, "sample_meshfile"):
                if rom_model.sample_meshfile is not None:
                    fixed_parameters_rom["meshfile"] = rom_model.sample_meshfile

        ######## RUN ROM AT FOM PARAMETER INSTANCES ########

        t1 = time.time()

        # run ROM at FOM mean input
        rom_run_directory_mean_fom =  f"{enkf_directory}/enkf_iter_{iiter}/rom/rom_fom/{run_directory_prefix}mean"
        output_from_mean_input_phys = process_model_qois(
            mean_input_fom,
            parameter_names,
            rom_model,
            rom_run_directory_mean_fom,
            fixed_parameters=fixed_parameters_rom,
        )

        # normalize ROM output from FOM mean
        output_from_mean_input_rom_fom = multi_transform(output_from_mean_input_phys, obs_transformers)
        output_from_mean_input_rom_fom = output_from_mean_input_rom_fom.flatten(order="C")

        log_str_rom_fom = "Iter {iiter}: Running ROM on FOM sample "
        rom_fom_run_dir = f"{enkf_directory}/enkf_iter_{iiter}/rom/rom_fom/{run_directory_prefix}"

        ensemble_outputs_rom_fom, _ = run_model_at_ensemble(
            n_ensemble_fom,
            rom_model,
            parameter_ensemble_fom_phys,
            parameter_names,
            obs_transformers,
            rom_fom_run_dir,
            enkf_file,
            log_str_rom_fom,
            evaluation_concurrency,
            mp_cntxt,
            fixed_parameters=fixed_parameters_rom,
        )

        ######## RUN ROM AT ROM PARAMETER INSTANCES ########

        # run ROM at mean ROM input
        rom_run_directory_mean_rom = f"{enkf_directory}/enkf_iter_{iiter}/rom/rom_rom/{run_directory_prefix}mean"
        output_from_mean_input_phys = process_model_qois(
            mean_input_rom,
            parameter_names,
            rom_model,
            rom_run_directory_mean_rom,
            fixed_parameters=fixed_parameters_rom,
        )

        # normalize ROM output from ROM mean
        output_from_mean_input_rom_rom = multi_transform(output_from_mean_input_phys, obs_transformers)
        output_from_mean_input_rom_rom = output_from_mean_input_rom_rom.flatten(order="C")

        log_str_rom_rom = "Iter {iiter}: Running ROM on ROM sample "
        rom_rom_run_dir = f"{enkf_directory}/enkf_iter_{iiter}/rom/rom_rom/{run_directory_prefix}"

        ensemble_outputs_rom_rom, _ = run_model_at_ensemble(
            n_ensemble_rom,
            rom_model,
            parameter_ensemble_rom_phys,
            parameter_names,
            obs_transformers,
            rom_rom_run_dir,
            enkf_file,
            log_str_rom_rom,
            evaluation_concurrency,
            mp_cntxt,
            fixed_parameters=fixed_parameters_rom,
        )

        timer_runtime_rom.append(time.time() - t1)

        t1 = time.time()

        # compute square root matrices
        Sin_fom = (parameter_ensemble_fom.T - mean_input_fom[:, np.newaxis]) / np.sqrt(n_ensemble_fom - 1)
        Sin_rom = (parameter_ensemble_rom.T - mean_input_rom[:, np.newaxis]) / np.sqrt(n_ensemble_rom - 1)

        Sout_fom     = (ensemble_outputs_fom     - output_from_mean_input_fom[:, np.newaxis])     / np.sqrt(n_ensemble_fom - 1)
        Sout_rom_fom = (ensemble_outputs_rom_fom - output_from_mean_input_rom_fom[:, np.newaxis]) / np.sqrt(n_ensemble_fom - 1)
        Sout_rom_rom = (ensemble_outputs_rom_rom - output_from_mean_input_rom_rom[:, np.newaxis]) / np.sqrt(n_ensemble_rom - 1)

        output_differences_fom     = observation_data[:, np.newaxis] - ensemble_outputs_fom
        output_differences_rom_fom = observation_data[:, np.newaxis] - ensemble_outputs_rom_fom
        output_differences_rom_rom = observation_data[:, np.newaxis] - ensemble_outputs_rom_rom

        # compute Kalman gains
        K1 = Sout_fom @ Sout_fom.T \
            + 0.25 * Sout_rom_fom @ Sout_rom_fom.T \
            - 0.5 * Sout_fom @ Sout_rom_fom.T \
            - 0.5 * Sout_rom_fom @ Sout_fom.T \
            + 0.25 * Sout_rom_rom @ Sout_rom_rom.T \
            + output_cov
        K2 = Sin_fom @ Sout_fom.T \
            + 0.25 * Sin_fom @ Sout_rom_fom.T \
            - 0.5 * Sin_fom @ Sout_rom_fom.T \
            - 0.5 * Sin_fom @ Sout_fom.T \
            + 0.25 * Sin_rom @ Sout_rom_rom.T

        # calculate parameter update
        update_fom = K2 @ np.linalg.solve(K1, output_differences_fom)
        update_rom = K2 @ np.linalg.solve(K1, output_differences_rom_rom)
        parameter_ensemble_fom += update_fom.T
        parameter_ensemble_rom += update_rom.T
        parameter_ensemble_fom_phys = multi_transform(parameter_ensemble_fom, param_transformers, inverse=True)
        parameter_ensemble_rom_phys = multi_transform(parameter_ensemble_rom, param_transformers, inverse=True)

        timer_enkf.append(time.time() - t1)

        # Check ROM accuracy
        # TODO: hard coded as max error over overlap set. Could be abstracted
        rom_obs_errors = np.linalg.norm(ensemble_outputs_fom - ensemble_outputs_rom_fom, axis=0) / np.linalg.norm(ensemble_outputs_fom, axis=0)
        rom_err_indicator = np.max(rom_obs_errors)
        if rom_err_indicator > rom_tol:
            enkf_file.write("Retrain ROM\n")
            train_rom = True
        else:
            enkf_file.write("Keep ROM\n")

        mean_input_fom = np.mean(parameter_ensemble_fom, axis=0)
        mean_input_rom = np.mean(parameter_ensemble_rom, axis=0)
        mean_input_fom_phys = multi_transform(mean_input_fom, param_transformers, inverse=True)
        mean_input_rom_phys = multi_transform(mean_input_rom, param_transformers, inverse=True)

        output_from_mean_input_diff_fom     = observation_data - output_from_mean_input_fom
        output_from_mean_input_diff_rom_fom = observation_data - output_from_mean_input_rom_fom
        output_from_mean_input_diff_rom_rom = observation_data - output_from_mean_input_rom_rom

        # compute output for logging
        rom_data_indicator.append(rom_err_indicator)
        input_mean_phys_fom.append(mean_input_fom_phys.copy())
        input_mean_phys_rom.append(mean_input_rom_phys.copy())
        input_norm_phys_fom.append(np.linalg.norm(parameter_ensemble_fom_phys, axis=0))
        input_norm_phys_rom.append(np.linalg.norm(parameter_ensemble_rom_phys, axis=0))
        input_variance_phys_fom.append(np.linalg.norm(np.var(parameter_ensemble_fom_phys, axis=0)))
        input_variance_phys_rom.append(np.linalg.norm(np.var(parameter_ensemble_rom_phys, axis=0)))
        output_diff_L2_fom.append(np.linalg.norm(output_differences_fom, axis=0) / np.linalg.norm(observation_data, axis=0))
        output_diff_L2_rom.append(np.linalg.norm(output_differences_rom_rom, axis=0) / np.linalg.norm(observation_data, axis=0))
        output_diff_L2_rom_fom.append(np.linalg.norm(output_differences_rom_fom, axis=0) / np.linalg.norm(observation_data, axis=0))
        output_from_mean_input_diff_L2_fom.append(np.linalg.norm(output_from_mean_input_diff_fom, axis=0) / np.linalg.norm(observation_data, axis=0))
        output_from_mean_input_diff_L2_rom_fom.append(np.linalg.norm(output_from_mean_input_diff_rom_fom, axis=0) / np.linalg.norm(observation_data, axis=0))
        output_from_mean_input_diff_L2_rom.append(np.linalg.norm(output_from_mean_input_diff_rom_rom, axis=0) / np.linalg.norm(observation_data, axis=0))

        # save stats online
        np.savez(
            f"{enkf_directory}/enkf_stats",
            rom_data_indicator=rom_data_indicator,
            input_mean_phys_fom=input_mean_phys_fom,
            input_mean_phys_rom=input_mean_phys_rom,
            input_norm_phys_fom=input_norm_phys_fom,
            input_norm_phys_rom=input_norm_phys_rom,
            input_variance_phys_fom=input_variance_phys_fom,
            input_variance_phys_rom=input_variance_phys_rom,
            output_diff_L2_fom=output_diff_L2_fom,
            output_diff_L2_rom=output_diff_L2_rom,
            output_diff_L2_rom_fom=output_diff_L2_rom_fom,
            output_from_mean_input_diff_L2_fom=output_from_mean_input_diff_L2_fom,
            output_from_mean_input_diff_L2_rom_fom=output_from_mean_input_diff_L2_rom_fom,
            output_from_mean_input_diff_L2_rom=output_from_mean_input_diff_L2_rom,
            timer_runtime_fom=timer_runtime_fom,
            timer_runtime_rom=timer_runtime_rom,
            timer_enkf=timer_enkf,
            timer_training=timer_training,
        )

    return mean_input_fom_phys, mean_input_rom_phys