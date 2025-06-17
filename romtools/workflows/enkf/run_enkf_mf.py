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

import romtools.linalg.linalg as la
from romtools.workflows.models import QoiModel
from romtools.workflows.model_builders import QoiModelBuilder
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows.workflow_utils import create_empty_dir
from romtools.workflows.enkf.run_enkf import *

def _create_parameter_dict(parameter_names, parameter_values):
    return dict(zip(parameter_names, parameter_values))


def run_enkf_mf(
    fom_model: QoiModel,
    rom_model_builder: QoiModelBuilder,
    observation_data: np.array, # could be defined as function for an "online" sensor
    prior: callable,
    parameter_names: list,
    absolute_enkf_work_directory: str,
    noise: float,     # TODO: Assuming Gaussian IID noise, could be general noise matrix
    rom_tol: float,
    n_ensemble_fom: int = 5,
    n_ensemble_rom: int = 15,
    n_enkf_iter: int = 5,
    random_seed: int = 1
):
    '''
    Main implementation of the MF enkf algorithm.
    '''
    enkf_directory = absolute_enkf_work_directory
    create_empty_dir(enkf_directory)

    run_directory_prefix = "run_"
    offline_directory_prefix = "data_for_rom"
    enkf_file = open(f"{enkf_directory}/enkf_status.log", "w", encoding="utf-8")

    # Generate "prior" guess of parameters
    parameter_ensemble_fom_phys = prior(n_ensemble_fom, random_seed)
    parameter_ensemble_rom_phys = prior(n_ensemble_rom, random_seed)
    # TODO: Algorithm is set up to work on nondim'd inputs. Need a more general way to get these scales
    param_min = min(parameter_ensemble_fom_phys.min(), parameter_ensemble_rom_phys.min())
    param_max = max(parameter_ensemble_fom_phys.max(), parameter_ensemble_rom_phys.max())

    # TODO: Note: In provided code, this is nondimensionalized with set constants (input_min, input_max)
    parameter_ensemble_fom = transform(parameter_ensemble_fom_phys, param_min, param_max)
    parameter_ensemble_rom = transform(parameter_ensemble_rom_phys, param_min, param_max)

    mean_input_fom = np.mean(parameter_ensemble_fom, axis=0)
    mean_input_rom = np.mean(parameter_ensemble_rom, axis=0)

    mean_input_fom_phys = inverse_transform(mean_input_fom, param_min, param_max)
    mean_input_rom_phys = inverse_transform(mean_input_rom, param_min, param_max)

    # Read measurements
    observation_data = observation_data.flatten()
    obs_min = observation_data.min()
    obs_max = observation_data.max()
    observation_data = transform(observation_data, obs_min, obs_max)
    n_outputs = len(observation_data)

    # Set output covariance
    output_cov = np.eye(n_outputs) * noise

    # Initialze data to collect
    input_mean_norm_fom = [np.linalg.norm(parameter_ensemble_fom_phys)]
    input_variance_fom = [np.linalg.norm(np.var(parameter_ensemble_fom_phys,axis=0))]
    output_diff_L2_fom = []
    output_from_mean_input_diff_L2_fom = []

    rom_data_indicator = []
    output_diff_L2_rom_fom = []
    output_from_mean_input_diff_L2_rom_fom = []

    input_mean_norm_rom = [np.linalg.norm(parameter_ensemble_rom_phys)]
    input_variance_rom = [np.linalg.norm(np.var(parameter_ensemble_rom_phys,axis=0))]
    output_diff_L2_rom = []
    output_from_mean_input_diff_L2_rom = []

    training_dirs = []
    train_rom = True
    for iiter in range(n_enkf_iter):
        enkf_file.write(f"ENKF iteration {iiter}\n")

        fom_run_directory_mean = f"{enkf_directory}/enkf_iter_{iiter}/fom/{run_directory_prefix}mean"

        # run FOM at mean input
        create_empty_dir(fom_run_directory_mean)
        parameter_dict = _create_parameter_dict(parameter_names, mean_input_fom_phys)
        fom_model.populate_run_directory(fom_run_directory_mean, parameter_dict)
        enkf_file.write(f"Iter {iiter}: Running FOM at mean \n")
        fom_model.run_model(fom_run_directory_mean, parameter_dict)

        # get (physical) output from mean
        output_from_mean_input_phys = fom_model.compute_qoi(fom_run_directory_mean, parameter_dict)
        # CRW: flatten?
        output_from_mean_input_phys = output_from_mean_input_phys.flatten()

        # normalize output
        output_from_mean_input_fom = transform(output_from_mean_input_phys, obs_min, obs_max)

        # run FOM at current ensemble
        for i in range(n_ensemble_fom):
            sample_index = i
            enkf_file.write(f"Iter {iiter}: Running FOM on FOM sample {sample_index} \n")
            parameter_input_phys = inverse_transform(parameter_ensemble_fom[sample_index,:], param_min, param_max)
            parameter_dict = _create_parameter_dict(parameter_names, parameter_input_phys)
            fom_run_directory = f"{enkf_directory}/enkf_iter_{iiter}/fom/{run_directory_prefix}{sample_index}"
            create_empty_dir(fom_run_directory)
            fom_model.populate_run_directory(fom_run_directory, parameter_dict)
            fom_model.run_model(fom_run_directory, parameter_dict)

            # get output of FOM
            fom_output_phys = fom_model.compute_qoi(fom_run_directory, parameter_dict)
            # CRW: flatten?
            fom_output_phys = fom_output_phys.flatten()
            fom_output = transform(fom_output_phys, obs_min, obs_max)
            if i == 0:
                outputs_fom = fom_output[None]
            else:
                outputs_fom = np.append(outputs_fom, fom_output[None], axis=0)
            training_dirs.append(fom_run_directory)

        # Train ROM
        if train_rom:
            updated_offline_data_dir = f"{enkf_directory}/enkf_iter_{iiter}/rom/{offline_directory_prefix}/"
            create_empty_dir(updated_offline_data_dir)
            rom_model = rom_model_builder.build_from_training_dirs(updated_offline_data_dir, training_dirs)

        # run ROM at FOM mean input
        rom_run_directory_mean_fom =  f"{enkf_directory}/enkf_iter_{iiter}/rom/rom_fom/{run_directory_prefix}mean"

        create_empty_dir(rom_run_directory_mean_fom)

        parameter_dict = _create_parameter_dict(parameter_names, mean_input_fom)
        rom_model.populate_run_directory(rom_run_directory_mean_fom, parameter_dict)
        rom_model.run_model(rom_run_directory_mean_fom, parameter_dict)

        # get output from mean
        output_from_mean_input_phys = rom_model.compute_qoi(rom_run_directory_mean_fom, parameter_dict)
        # CRW: flatten?
        output_from_mean_input_phys = output_from_mean_input_phys.flatten()
        output_from_mean_input_rom_fom = transform(output_from_mean_input_phys, obs_min, obs_max)

        # run ROM at current FOM ensemble
        for i in range(n_ensemble_fom):
            sample_index = i
            enkf_file.write(f"Iter {iiter}: Running ROM on FOM sample {sample_index} \n")
            parameter_input_phys = inverse_transform(parameter_ensemble_fom[sample_index,:], param_min, param_max)
            parameter_dict = _create_parameter_dict(parameter_names, parameter_input_phys)
            rom_run_directory =  f"{enkf_directory}/enkf_iter_{iiter}/rom/rom_fom/{run_directory_prefix}{sample_index}"
            create_empty_dir(rom_run_directory)
            rom_model.populate_run_directory(rom_run_directory, parameter_dict)
            rom_model.run_model(rom_run_directory, parameter_dict)

            # get output of FOM
            rom_output_phys = rom_model.compute_qoi(rom_run_directory, parameter_dict)
            # CRW: flatten?
            rom_output_phys = rom_output_phys.flatten()
            rom_output = transform(rom_output_phys, obs_min, obs_max)
            if i == 0:
                outputs_rom_fom = rom_output[None]
            else:
                outputs_rom_fom = np.append(outputs_rom_fom, rom_output[None],axis=0)

        # run ROM at mean rom input
        rom_run_directory_mean_rom =  f"{enkf_directory}/enkf_iter_{iiter}/rom/rom_rom/{run_directory_prefix}mean"

        create_empty_dir(rom_run_directory_mean_rom)

        parameter_dict = _create_parameter_dict(parameter_names, mean_input_rom_phys)
        rom_model.populate_run_directory(rom_run_directory_mean_rom, parameter_dict)
        rom_model.run_model(rom_run_directory_mean_rom, parameter_dict)

        # get output from mean
        output_from_mean_input_phys = rom_model.compute_qoi(rom_run_directory_mean_rom, parameter_dict)
        # CRW: flatten?
        output_from_mean_input_phys = output_from_mean_input_phys.flatten()
        output_from_mean_input_rom_rom = transform(output_from_mean_input_phys, obs_min, obs_max)

        # run ROM at current ROM ensemble
        for i in range(n_ensemble_rom):
            sample_index = i
            enkf_file.write(f"Iter {iiter}: Running ROM on ROM sample {sample_index} \n")
            parameter_input_dim = inverse_transform(parameter_ensemble_rom[sample_index,:], param_min, param_max)
            parameter_dict = _create_parameter_dict(parameter_names, parameter_input_dim)
            rom_run_directory =  f"{enkf_directory}/enkf_iter_{iiter}/rom/rom_rom/{run_directory_prefix}{sample_index}"
            create_empty_dir(rom_run_directory)
            rom_model.populate_run_directory(rom_run_directory, parameter_dict)
            rom_model.run_model(rom_run_directory, parameter_dict)

            # get output of FOM
            rom_output_phys = rom_model.compute_qoi(rom_run_directory, parameter_dict)
            # CRW: flatten?
            rom_output_phys = rom_output_phys.flatten()
            rom_output = transform(rom_output_phys, obs_min, obs_max)
            if i == 0:
                outputs_rom_rom = rom_output[None]
            else:
                outputs_rom_rom = np.append(outputs_rom_rom, rom_output[None],axis=0)

        ensemble_outputs_fom = np.array(outputs_fom).T
        ensemble_outputs_rom_fom = np.array(outputs_rom_fom).T
        ensemble_outputs_rom_rom = np.array(outputs_rom_rom).T

        Sin_fom = (parameter_ensemble_fom.T - mean_input_fom[:,np.newaxis]) / np.sqrt(n_ensemble_fom - 1)
        Sin_rom = (parameter_ensemble_rom.T - mean_input_rom[:,np.newaxis]) / np.sqrt(n_ensemble_rom - 1)

        Sout_fom = (ensemble_outputs_fom - output_from_mean_input_fom[:,np.newaxis]) / np.sqrt(n_ensemble_fom - 1)
        Sout_rom_fom = (ensemble_outputs_rom_fom - output_from_mean_input_rom_fom[:,np.newaxis]) / np.sqrt(n_ensemble_fom - 1)
        Sout_rom_rom = (ensemble_outputs_rom_rom - output_from_mean_input_rom_rom[:,np.newaxis]) / np.sqrt(n_ensemble_rom - 1)

        output_differences_fom = -ensemble_outputs_fom + observation_data[:, np.newaxis]
        output_differences_rom_fom = -ensemble_outputs_rom_fom + observation_data[:, np.newaxis]
        output_differences_rom_rom = -ensemble_outputs_rom_rom + observation_data[:, np.newaxis]

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

        update_fom = K2 @ np.linalg.solve(K1, output_differences_fom)
        update_rom = K2 @ np.linalg.solve(K1, output_differences_rom_rom)

        parameter_ensemble_fom += update_fom.T
        parameter_ensemble_rom += update_rom.T

        #Check ROM accuracy
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

        mean_input_fom_phys = inverse_transform(mean_input_fom, param_min, param_max)
        mean_input_rom_phys = inverse_transform(mean_input_rom, param_min, param_max)

        output_from_mean_input_diff_fom = observation_data - output_from_mean_input_fom
        output_from_mean_input_diff_rom_fom = observation_data - output_from_mean_input_rom_fom
        output_from_mean_input_diff_rom_rom = observation_data - output_from_mean_input_rom_rom

        input_mean_norm_fom.append(np.linalg.norm(mean_input_fom_phys))
        input_variance_fom.append(np.linalg.norm(np.var(parameter_ensemble_fom_phys, axis=0)))
        output_diff_L2_fom.append(np.linalg.norm(output_differences_fom, axis=0) / np.linalg.norm(observation_data))
        output_from_mean_input_diff_L2_fom.append(np.linalg.norm(output_from_mean_input_diff_fom, axis=0) / np.linalg.norm(observation_data))

        rom_data_indicator.append(rom_err_indicator)
        output_diff_L2_rom_fom.append(np.linalg.norm(output_differences_rom_fom, axis=0) / np.linalg.norm(observation_data))
        output_from_mean_input_diff_L2_rom_fom.append(np.linalg.norm(output_from_mean_input_diff_rom_fom, axis=0) / np.linalg.norm(observation_data))

        input_mean_norm_rom.append(np.linalg.norm(mean_input_rom_phys))
        input_variance_rom.append(np.linalg.norm(np.var(parameter_ensemble_rom_phys, axis=0)))
        output_diff_L2_rom.append(np.linalg.norm(output_differences_rom_rom, axis=0) / np.linalg.norm(observation_data))
        output_from_mean_input_diff_L2_rom.append(np.linalg.norm(output_from_mean_input_diff_rom_rom, axis=0) / np.linalg.norm(observation_data))

    np.savez(f"{enkf_directory}/enkf_stats",
        rom_data_indicator=rom_data_indicator,
        input_mean_norm_fom=input_mean_norm_fom,
        input_variance_fom=input_variance_fom,
        output_diff_L2_fom=output_diff_L2_fom,
        output_from_mean_input_diff_L2_fom=output_from_mean_input_diff_L2_fom,
        output_diff_L2_rom_fom=output_diff_L2_rom_fom,
        output_from_mean_input_diff_L2_rom_fom=output_from_mean_input_diff_L2_rom_fom,
        input_mean_norm_rom=input_mean_norm_rom,
        input_variance_rom=input_variance_rom,
        output_diff_L2_rom=output_diff_L2_rom,
        output_from_mean_input_diff_L2_rom=output_from_mean_input_diff_L2_rom)

    return mean_input_fom_phys, mean_input_rom_phys