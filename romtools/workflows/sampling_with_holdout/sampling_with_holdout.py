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

from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows.workflow_utils import create_empty_dir
from romtools.workflows.sampling import *
from romtools.workflows.model_builders import QoiModelBuilder


def _create_parameter_dict(parameter_names, parameter_values):
    return dict(zip(parameter_names, parameter_values))


def run_sampling_with_holdout(
    fom_model: QoiModel,
    rom_model_builder: QoiModelBuilder,
    parameter_space: ParameterSpace,
    absolute_work_directory: str,
    holdout_set_size: int = 5,
    max_number_of_rom_samples: int = 20,
    tolerance=1e-5,
    random_seed: int = 1,
):
    """
    Core algorithm
    """
    sampling_directory = absolute_work_directory
    create_empty_dir(sampling_directory)
    offline_directory_prefix = "offline_data"

    run_directory_prefix = "run_"
    sampling_file = open(
        f"{sampling_directory}/sampling_with_holdout_status.log", "w", encoding="utf-8"
    )
    sampling_file.write("Holdout sampling status \n")
    fom_time = 0.0
    rom_time = 0.0
    basis_time = 0.0

    np.random.seed(random_seed)

    parameter_samples = parameter_space.generate_samples(
        holdout_set_size + max_number_of_rom_samples
    )
    parameter_names = parameter_space.get_names()

    holdout_sample_indices = range(holdout_set_size)
    training_sample_indices = range(
        holdout_set_size, holdout_set_size + max_number_of_rom_samples
    )
    holdout_samples = parameter_samples[holdout_sample_indices]
    training_samples = parameter_samples[training_sample_indices]

    # Setup FOM directories for potential training runs
    for sample_index, sample in enumerate(training_samples):
        fom_run_directory = f"{sampling_directory}/fom/training_set/{run_directory_prefix}{sample_index}"
        create_empty_dir(fom_run_directory)

        parameter_dict = _create_parameter_dict(parameter_names, sample)
        fom_model.populate_run_directory(fom_run_directory, parameter_dict)

    # Setup FOM directories and run samples to build holdout set.
    t0 = time.time()
    fom_qois_holdout_set = np.zeros(holdout_set_size)
    sampling_file.write(f"Building holdout set \n")
    for sample_index in holdout_sample_indices:
        sampling_file.write(f"Running holdout FOM sample {sample_index} \n")
        parameter_dict = _create_parameter_dict(
            parameter_names, parameter_samples[sample_index]
        )
        fom_run_directory = (
            f"{sampling_directory}/fom/holdout_set/{run_directory_prefix}{sample_index}"
        )
        create_empty_dir(fom_run_directory)
        fom_model.populate_run_directory(fom_run_directory, parameter_dict)
        fom_model.run_model(fom_run_directory, parameter_dict)
        fom_qoi = fom_model.compute_qoi(fom_run_directory, parameter_dict)
        if sample_index == 0:
            fom_qois_holdout_set = fom_qoi[None]
        else:
            fom_qois_holdout_set = np.append(
                fom_qois_holdout_set, fom_qoi[None], axis=0
            )
    fom_time += time.time() - t0

    # Create ROM bases
    sampling_file.write(f"Beginning sampling procedure \n")

    converged = False
    training_dirs = []
    training_params = []
    trained_samples = []
    rom_qois_holdout_set = np.zeros(holdout_set_size)
    holdout_set_errs = np.array([])
    sample_index = 0
    while converged is False and sample_index < max_number_of_rom_samples:
        # Initialize FOM to be run at training set sample
        sampling_file.write(f"Holdout set iteration # {sample_index}\n")

        parameter_dict = _create_parameter_dict(
            parameter_names, training_samples[sample_index]
        )
        fom_run_directory = f"{sampling_directory}/fom/training_set/{run_directory_prefix}{sample_index}"

        # Run FOM at training parameter
        t0 = time.time()
        sampling_file.write(f"Running training FOM sample {sample_index} \n")
        fom_model.run_model(fom_run_directory, parameter_dict)
        fom_time += time.time() - t0

        sampling_file.write(f"Adding training FOM sample {sample_index} to basis \n")
        training_params.append(parameter_samples[sample_index])
        training_dirs.append(fom_run_directory)
        trained_samples.append(sample_index)
        sampling_file.write(f"Parameter samples: \n {np.asarray(training_params)}\n")

        # Add FOM sample to ROM basis
        t0 = time.time()
        sampling_file.write(f"Constructing ROM iteration {sample_index} \n")
        updated_offline_data_dir = f"{sampling_directory}/rom_iteration_{sample_index}/{offline_directory_prefix}/"
        create_empty_dir(updated_offline_data_dir)
        rom_model = rom_model_builder.build_from_training_dirs(
            updated_offline_data_dir, training_dirs
        )
        basis_time += time.time() - t0

        # Evaluate ROM at holdout set and compute QOI errors
        t0 = time.time()
        for holdout_sample_index in holdout_sample_indices:
            sampling_file.write(
                f"  Running ROM at holdout sample {holdout_sample_index}\n"
            )
            rom_run_directory = (
                f"{sampling_directory}/rom/{run_directory_prefix}{holdout_sample_index}"
            )

            parameter_dict = _create_parameter_dict(
                parameter_names, parameter_samples[holdout_sample_index]
            )
            create_empty_dir(rom_run_directory)
            rom_model.populate_run_directory(rom_run_directory, parameter_dict)
            rom_model.run_model(rom_run_directory, parameter_dict)
            rom_qois_holdout_set[holdout_sample_index] = rom_model.compute_qoi(
                rom_run_directory, parameter_dict
            )
        rom_time += time.time() - t0

        # If Max QOI error is less than tolerance, converged = True
        holdout_set_err = np.linalg.norm(
            rom_qois_holdout_set - fom_qois_holdout_set, np.inf
        )
        holdout_set_errs = np.append(holdout_set_errs, holdout_set_err)
        if holdout_set_err < tolerance:
            converged = True
            print(f"Holdout sampling run converged with QoI error {holdout_set_err}\n")

        sample_index += 1
    if sample_index == max_number_of_rom_samples:
        print("Warning: Max number of iterations reached for holdout sampling")

    np.savez(
        f"{sampling_directory}/holdout_stats",
        holdout_set_errs=holdout_set_errs,
        trained_samples=trained_samples,
        fom_time=fom_time,
        rom_time=rom_time,
        basis_time=basis_time,
    )

    sampling_file.close()
