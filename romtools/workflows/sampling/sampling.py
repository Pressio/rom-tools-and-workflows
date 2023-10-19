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
import numpy as np
import time
import os
import numpy as np
import time


def runSampling(samplingCoupler,testing_sample_size=10,random_seed=1):
  '''
  Core algorithm
  '''
  np.random.seed(random_seed)
  ## create parameter domain
  parameterSpace = samplingCoupler.getParameterSpace()

  n_params = parameterSpace.getDimensionality()
  parameter_samples = parameterSpace.generateSamples(testing_sample_size)

  # Make FOM/ROM directories
  starting_sample_index = 0
  samplingCoupler.createCases(starting_sample_index,parameter_samples)

  # Run first FOM case
  run_times = np.zeros(testing_sample_size)
  for sample_index in range(0,testing_sample_size):
    print("=======  Sample " + str(sample_index) + " ============")
    print("Running")
    case_directory = samplingCoupler.getSolDirectoryBaseName() + str(sample_index) + '/'
    os.chdir(case_directory)
    ts = time.time()
    flag = samplingCoupler.runModel(samplingCoupler.getInputFileName(),parameter_samples[sample_index])
    tf = time.time()
    run_times[sample_index] = tf - ts
    if flag == 0:
      print("Sample complete, run time = " + str(tf - ts) )
    else:
      print("Sample failed, run time = " + str(tf - ts) )

    print(" ")
    work_dir = samplingCoupler.getBaseDirectory() + '/' + samplingCoupler.getWorkDirectoryBaseName()
    np.savez(work_dir + '/sampling_stats',run_times=run_times)
    os.chdir(samplingCoupler.getBaseDirectory())
