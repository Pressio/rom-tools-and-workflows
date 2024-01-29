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

from romtools.workflows.sampling.\
    sampling_coupler_base import SamplingCouplerBase
from romtools.workflows.parameter_spaces import monte_carlo_sample


def run_sampling(sampling_coupler: SamplingCouplerBase,
                 testing_sample_size: int = 10,
                 random_seed: int = 1):
    '''
    Core algorithm
    '''
    np.random.seed(random_seed)

    # create parameter domain
    parameter_space = sampling_coupler.get_parameter_space()
    parameter_samples = monte_carlo_sample(parameter_space,
                                           testing_sample_size)

    # Make FOM/ROM directories
    starting_sample_index = 0
    sampling_coupler.create_cases(starting_sample_index, parameter_samples)

    # Run first FOM case
    run_times = np.zeros(testing_sample_size)
    for sample_index in range(0, testing_sample_size):
        print("=======  Sample " + str(sample_index) + " ============")
        print("Running")
        case_directory = sampling_coupler.get_sol_directory(sample_index)

        run_times[sample_index] = run_sample(sampling_coupler,
                                             case_directory,
                                             parameter_samples[sample_index])

        np.savez(f'{sampling_coupler.get_base_directory()}/sampling_stats',
                 run_times=run_times)


def run_sample(sampling_coupler: SamplingCouplerBase,
               case_directory: str,
               parameter_values):
    '''
    Execute individual sample
    '''

    os.chdir(case_directory)

    ts = time.time()
    flag = sampling_coupler.run_model(sampling_coupler.get_input_filename(),
                                      parameter_values)
    tf = time.time()
    run_time = tf - ts

    if flag == 0:
        print(f"Sample complete, run time = {run_time}")
    else:
        print(f"Sample failed, run time = {run_time}")
    print(" ")

    os.chdir(sampling_coupler.get_base_directory())
    return run_time
