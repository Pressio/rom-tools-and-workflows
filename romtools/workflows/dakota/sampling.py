'''
This module implements the abstract class and function required to
couple a model to Dakota for use in random sampling. To couple to Dakota,
a user should
1. Complete the DakotaSamplingCouplerBase for their application of interest
2. Use the "run_dakota_sampling" function as their Dakota analysis driver
'''

import abc
import sys
import os
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
