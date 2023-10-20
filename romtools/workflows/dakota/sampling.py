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
This module implements the abstract class and function required to
couple a model to Dakota for use in random sampling. To couple to Dakota,
a user should
1. Complete the DakotaSamplingCouplerBase for their application of interest
2. Use the "run_dakota_sampling" function as their Dakota analysis driver
'''

import os
import abc
import sys
import numpy as np


class DakotaSamplingCouplerBase(abc.ABC):
    '''
     Abstract class implementation
    '''
    __base_directory = os.getcwd() + '/'

    def __init__(self,template_directory,template_file):
      '''
      Initializes a DakotaSamplingCouplerBase object.

      Args:
          template_directory (str): The directory containing the template file.
          template_file (str): The name of the template file.
      '''
      self.__template_directory = template_directory
      self.__template_file = template_file

    def copyTemplateFile(self):
      '''
      Copies the template file from the specified directory to the current working directory.
      '''
      os.system('cp ' + self.__template_directory + self.__template_file + ' .')

    @abc.abstractmethod
    def setParametersInInput(self,parameter_sample):
        '''
        This function is called from a run directory. It needs to update a
        template file with parameter values defined in parameter_sample.
        For example, this could be done with dprepro
        '''
        pass

    @abc.abstractmethod
    def runModel(self):
        '''
        This function is called from a run directory. It needs to execute our model.
        '''
        pass

    @abc.abstractmethod
    def computeQoiAndSaveToFile(self):
        '''
        This function should compute a QoI and save to file. The output file should match what is specified in the
        Dakota input script
        '''
        pass


def run_dakota_sampling(DakotaSamplingCoupler):
    '''
    Basic Dakota analysis driver leveraging the DakotaSamplingCoupler API

    Args:
        DakotaSamplingCoupler (DakotaSamplingCouplerBase): An instance of a DakotaSamplingCouplerBase-derived class.

    '''
    data = np.genfromtxt(sys.argv[1])[:,0]
    num_vars = int(data[0])
    param_values = np.array(data[1:1+num_vars])
    DakotaSamplingCoupler.copyTemplateFile()
    DakotaSamplingCoupler.setParametersInInput(param_values)
    DakotaSamplingCoupler.runModel()
    DakotaSamplingCoupler.computeQoiAndSaveToFile()
