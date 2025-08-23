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
This module implements the class required to couple a model to Dakota.
To couple to Dakota, a user should
1. Complete a Model class for their application of interest
2. Complete a driver script that instantiates the model and calls this coupler for use as the Dakota analysis driver
'''

import os
import sys
import time
import numpy as np
import pathlib
from typing import List
import subprocess

from romtools.workflows.models import QoiModel
import dakota.interfacing as di

def _create_parameter_dict(param_data):
    data_floats = []
    data_strings = []
    for line in param_data:
        cleaned_string = line.strip()
        number, word = cleaned_string.split()
        try:
            number = float(number)
        except:
            number = 0
        data_floats.append(number)
        data_strings.append(word)

    num_vars = int(data_floats[0])
    parameter_values = np.array(data_floats[1 : 1 + num_vars])
    parameter_names = data_strings[1 : 1 + num_vars]
    return dict(zip(parameter_names, parameter_values))

def _create_run_directory(run_directory_base: str, eval_num: int) -> pathlib.Path:
    """Create tagged run directory

    run_directory_base: untagged run directory name
    eval_num: evaluation number for tag
    link_files: List of files to link into the workdir

    return: name of tagged workdir
    """
    run_directory = pathlib.Path(run_directory_base + f".{eval_num}")
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory


def run_batch_models_for_dakota(
    model: QoiModel,
    multifidelity_flag: bool = False,
    add_core_time_metadata: bool = False,
    num_responses: int = 1,
    base_path: str = os.getcwd()
    ):
    '''
    This function should be used in a driver script that will be called with
        `python <driver.py> batch_params.in batch_results.out`

    by Dakota. The parameter space is built by parsing batch_params.in,
    while the output QoIs will be saved to batch_results.out. Note that this
    function assumes that flux is used to run each model. 

    Args:
        model: rom-tools model instance (FOM or ROM)
        multifidelity_flag: True if being used with Dakota's multifidelity (MF) tools
        add_core_time_metadata: True if computational cost should be included in QoI file
                                Useful for MF UQ when model does not have a cost model.

    '''

    # Read parameters from batch_params_file
    batch_params_file = sys.argv[1]
    batch_results_file = sys.argv[2]

    # set up run directories
    run_directory_base = f"{base_path}/run"
    splitter = di.BatchSplitter(batch_params_file)
    eval_run_directory_map = {}
    run_times_map = {}
    for i, params in enumerate(splitter):
        eval_num = splitter.eval_nums[i]
        run_directory = _create_run_directory(run_directory_base, eval_num)
        eval_run_directory_map[eval_num] = run_directory 
        parameter_sample = _create_parameter_dict(params)

        # Initialize and run ROM
        model.populate_run_directory(str(run_directory), parameter_sample)
        t0 = time.time()
        model.run_model(str(run_directory), parameter_sample)
        run_times_map[eval_num] = time.time() - t0

    # wait for all flux jobs to complete before getting QoIs
    subprocess.run(["flux","job","wait","--all"],env=dict(os.environ))

    with open(batch_results_file, 'w') as file:
        for i in range(len(splitter)):
            eval_num = splitter.eval_nums[i]

            # Compute model QoI and save it to file
            run_directory = eval_run_directory_map[eval_num]
            
            # Check if model ran successfully
            code = model.check_for_model_failure(str(run_directory))
            if code == 0:
                qoi = model.compute_qoi(str(run_directory), parameter_sample)
            else:
                qoi = np.nan*np.ones(num_responses)

            #if multifidelity_flag:
            #    assert qoi.size == 1, "For MF UQ, a scalar QoI is required"

            if add_core_time_metadata:
                n_cores = model.get_number_of_processors()
                qoi = np.append(qoi, n_cores*run_times_map[eval_num])  # Cost metadata for Q

            for q in qoi:
                file.write(f"{q}\n")
            file.write("#\n")
