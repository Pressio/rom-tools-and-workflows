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
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows.workflow_utils import create_empty_dir

def _create_parameter_dict(parameter_names, parameter_values):
    return dict(zip(parameter_names, parameter_values))

def transform(array, amin, amax):
    # Move from data in \mathbb{R} to [0,1]
    return (array-amin)/(amax-amin)

def inverse_transform(array, amin, amax):    
    # Move from data in [0,1] to \mathbb{R}
    return (amax-amin)*array+amin

def run_enkf(  fom_model: QoiModel,
               observation_data: np.array, # could be defined as function for an "online" sensor
               prior: callable, 
               parameter_names: list, 
               absolute_enkf_work_directory: str,
               noise: float,     # TODO: Assuming Gaussian IID noise, could be general noise matrix
               n_ensemble: int = 10,
               n_enkf_iter: int = 5,
               random_seed: int = 1):
    '''
    Main implementation of the enkf algorithm.
    '''
    enkf_directory = absolute_enkf_work_directory
    create_empty_dir(enkf_directory)

    run_directory_prefix = "run_"
    enkf_file = open(f"{enkf_directory}/enkf_status.log", "w", encoding="utf-8")

    # Generate "prior" guess of parameters
    parameter_ensemble_phys = prior(n_ensemble, random_seed)

    #TODO: Algorithm is set up to work on nondim'd inputs. Need a more general way to get these scales
    param_min = parameter_ensemble_phys.min()
    param_max = parameter_ensemble_phys.max()

    # TODO: Note: In provided code, this is nondimensionalized with set constants (input_min, input_max)
    parameter_ensemble = transform(parameter_ensemble_phys, param_min, param_max)

    mean_input = np.mean(parameter_ensemble, axis=0)
    iiter = 0


    # Read measurements
    observation_data = observation_data.flatten()
    obs_min = observation_data.min()
    obs_max = observation_data.max()
    observation_data = transform(observation_data, obs_min, obs_max)
    n_outputs = len(observation_data)

    # Set output covariance
    output_cov = np.eye(n_outputs) * noise

    # Initialze data to collect
    input_mean_norm = [np.linalg.norm(parameter_ensemble_phys)]
    input_variance = [np.linalg.norm(np.var(parameter_ensemble_phys,axis=0))]
    output_diff_L2 = []
    output_from_mean_input_diff_L2 = []

    for iiter in range(n_enkf_iter):
        enkf_file.write(f"ENKF iteration {iiter}\n")

        fom_run_directory_mean =  f'{enkf_directory}/enkf_iter_{iiter}/{run_directory_prefix}mean'

        # run FOM at mean input
        create_empty_dir(fom_run_directory_mean)

        mean_input_phys = inverse_transform(mean_input, param_min, param_max)
        parameter_dict = _create_parameter_dict(parameter_names, mean_input_phys)
        fom_model.populate_run_directory(fom_run_directory_mean, parameter_dict)
        enkf_file.write(f"Iter {iiter}: Running FOM at mean \n")
        fom_model.run_model(fom_run_directory_mean,parameter_dict)

        # get (physical) output from mean
        output_from_mean_input_phys = fom_model.compute_qoi(fom_run_directory_mean,parameter_dict)

        # normalize output
        output_from_mean_input = transform(output_from_mean_input_phys, obs_min, obs_max)

        # run FOM at current ensemble
        for i in range(n_ensemble):
            sample_index = i
            enkf_file.write(f"Iter {iiter}: Running FOM sample {sample_index} \n")
            parameter_input_phys = inverse_transform(parameter_ensemble[sample_index,:], param_min, param_max)
            parameter_dict = _create_parameter_dict(parameter_names, parameter_input_phys)
            fom_run_directory =  f'{enkf_directory}/enkf_iter_{iiter}/{run_directory_prefix}{sample_index}'
            create_empty_dir(fom_run_directory)
            fom_model.populate_run_directory(fom_run_directory,parameter_dict)
            fom_model.run_model(fom_run_directory,parameter_dict)

            # get output of FOM
            fom_output_phys = fom_model.compute_qoi(fom_run_directory,parameter_dict)
            fom_output = transform(fom_output_phys, obs_min, obs_max)
            if i == 0:
                outputs = fom_output[None]
            else:
                outputs = np.append(outputs,fom_output[None],axis=0)

        # compute correlation matrices
        ensemble_outputs = np.array(outputs).T

        # compute gain_matrix
        Sin = (parameter_ensemble.T - mean_input[:,np.newaxis]) / np.sqrt(n_ensemble - 1)
        Sout = (ensemble_outputs - output_from_mean_input[:,np.newaxis]) / np.sqrt(n_ensemble - 1)

        output_differences = -ensemble_outputs + observation_data[:, np.newaxis]

        K1 = Sout @ Sout.T + output_cov
        K2 = Sin @ Sout.T
        update = K2 @ np.linalg.solve(K1, output_differences)

        parameter_ensemble += update.T
        enkf_file.write(f'{parameter_ensemble}')
        # parameter_ensemble = np.clip(parameter_ensemble, a_min=0, a_max = 1)
        mean_input = np.mean(parameter_ensemble, axis=0)
        output_from_mean_input_diff = observation_data - output_from_mean_input

        input_mean_norm.append(np.linalg.norm(mean_input_phys))
        input_variance.append(np.linalg.norm(np.var(parameter_ensemble_phys,axis=0)))
        output_diff_L2.append(np.linalg.norm(output_differences,axis=0) / np.linalg.norm(observation_data))
        output_from_mean_input_diff_L2.append(np.linalg.norm(output_from_mean_input_diff)/np.linalg.norm(observation_data))
    
    np.savez(f'{enkf_directory}/enkf_stats',
            input_mean_norm=input_mean_norm,
            input_variance=input_variance,
            output_diff_L2=output_diff_L2,
            output_from_mean_input_diff_L2=output_from_mean_input_diff_L2)
    # TODO: add timings
    
    return mean_input #TODO: what should we return? 