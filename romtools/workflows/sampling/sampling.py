import numpy as np
import time
import os
import csv
import sys
import scipy
import numpy as np
import argparse
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



