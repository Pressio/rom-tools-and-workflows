'''
Abstract implementation of the SamplingCoupler class. To gain access to the sampling algorithm, the user needs to
complete this class for their application
'''
import os
import abc
import numpy as np
class SamplingCouplerBase(abc.ABC):
    '''
    Partially explicit implementation
    '''

    def __init__(self, template_directory, template_input_file, \
                 work_directory = None,
                 work_directory_base_name = 'work',\
                 sol_directory_base_name = 'run_'):
      '''
      Initialize a SamplingCouplerBase object.

      Args:
          template_directory (str): The directory containing input file templates.
          template_input_file (str): The template input file for the model.
          work_directory (str, optional): The working directory for the sampling. If not provided, the current directory is used.
          work_directory_base_name (str, optional): The base name for the working directory. Defaults to 'work'.
          sol_directory_base_name (str, optional): The base name for the solution directories within the working directory. Defaults to 'run_'
      '''

      self.__base_directory = os.getcwd() + '/' if work_directory == None else work_directory

      ## check if template_directory is an absolute path
      strings_to_check = min(5,np.size(template_directory))
      assert self.__base_directory[0:strings_to_check] == template_directory[0:strings_to_check], 'Path to template directory must be an absolute path'
      self.__template_directory = template_directory
      self.__template_input_file = template_input_file
      self.__work_directory_base_name = work_directory_base_name
      self.__sol_directory_base_name = sol_directory_base_name


    def getInputFileName(self):
      '''Get the name of the template input file.'''
      return self.__template_input_file

    def getWorkDirectoryBaseName(self):
      '''Get the base name for the working directory.'''
      return self.__work_directory_base_name

    def getSolDirectoryBaseName(self):
      '''Get the base name for the solution directories within the working directory.'''
      return self.__work_directory_base_name + '/' + self.__sol_directory_base_name

    def getBaseDirectory(self):
      '''Get the base directory for sampling.'''
      return self.__base_directory

    def createCases(self,starting_sample_no,parameter_samples):
      '''
      Create sampling cases with parameter samples.

      Args:
          starting_sample_no (int): The starting sample number.
          parameter_samples (np.ndarray): An array of parameter samples.
      '''
      n_samples = parameter_samples.shape[0]
      path_to_work_dir = self.__base_directory + '/' + self.__work_directory_base_name + '/'
      if os.path.isdir(path_to_work_dir):
          pass
      else:
          os.mkdir(path_to_work_dir)

      for sample_no in range(starting_sample_no,starting_sample_no + n_samples):
          path_to_dir = self.__base_directory + '/' + self.__work_directory_base_name + '/' + self.__sol_directory_base_name + str(sample_no)
          if os.path.isdir(path_to_dir):
              pass
          else:
              os.mkdir(path_to_dir)
          self.__setupCase(path_to_dir,parameter_samples[sample_no - starting_sample_no])

    def __setupCase(self,path_to_case,parameter_samples):
      '''
      Set up a specific sampling case.

      Args:
          path_to_case (str): The path to the directory for the sampling case.
          parameter_samples (np.ndarray): Parameter samples for the case.
      '''
      os.chdir(path_to_case)
      os.system('cp ' + self.__template_directory + '/' + self.__template_input_file + ' . ')
      self.setParametersInInput(self.__template_input_file,parameter_samples)
      os.chdir(self.getBaseDirectory())

    @abc.abstractmethod
    def setParametersInInput(self,filename,parameter_sample):
        '''
        This function is called from a run directory. It needs to update a
        template file with parameter values defined in parameter_sample.
        For example, this could be done with dprepro
        '''
        pass

    @abc.abstractmethod
    def runModel(self,filename,parameter_values):
        '''
        This function is called from a run directory. It needs to execute our model.
        If the model runs succesfully, return 0
        If fails, return 1
        '''
        return 0

    @abc.abstractmethod
    def getParameterSpace(self):
        '''
        This function should return a ParameterSpace class defining our parameter space.
        '''
