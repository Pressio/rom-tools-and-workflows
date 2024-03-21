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

from romtools.workflows.parameter_spaces import monte_carlo_sample
from romtools.workflows.workflow_utils import create_empty_dir
from romtools.workflows.models import Model
from romtools.workflows.parameter_spaces import ParameterSpace


def _create_parameter_dict(parameter_names, parameter_values):
    return dict(zip(parameter_names, parameter_values))


def run_sampling(model: Model,
                 parameter_space: ParameterSpace,
                 run_directory_prefix: str = "./run_",
                 number_of_samples: int = 10,
                 random_seed: int = 1):
    '''
    Core algorithm
    '''

    np.random.seed(random_seed)

    # create parameter samples
    parameter_samples = monte_carlo_sample(parameter_space,
                                           number_of_samples)

    parameter_names = parameter_space.get_names()

    # Setup model directories
    starting_sample_index = 0
    end_sample_index = starting_sample_index + parameter_samples.shape[0]
    for sample_index in range(starting_sample_index, end_sample_index):
        run_directory = f'{run_directory_prefix}{sample_index}'
        create_empty_dir(run_directory)
        parameter_dict = _create_parameter_dict(parameter_names, parameter_samples[sample_index - starting_sample_index])
        model.populate_run_directory(run_directory, parameter_dict)

    # Run cases
    run_times = np.zeros(number_of_samples)
    for sample_index in range(0, number_of_samples):
        print("=======  Sample " + str(sample_index) + " ============")
        print("Running")
        run_directory = f'{run_directory_prefix}{sample_index}'
        parameter_dict = _create_parameter_dict(parameter_names, parameter_samples[sample_index])
        run_times[sample_index] = run_sample(run_directory, model,
                                             parameter_dict)
        sample_stats_save_directory = f'{run_directory_prefix}{sample_index}/../'
        np.savez(f'{sample_stats_save_directory}/sampling_stats',
                 run_times=run_times)


def run_sample(run_directory: str, model: Model,
               parameter_sample: dict):
    '''
    Execute individual sample
    '''

    ts = time.time()
    flag = model.run_model(run_directory, parameter_sample)
    tf = time.time()
    run_time = tf - ts

    if flag == 0:
        print(f"Sample complete, run time = {run_time}")
    else:
        print(f"Sample failed, run time = {run_time}")
    print(" ")
    return run_time
