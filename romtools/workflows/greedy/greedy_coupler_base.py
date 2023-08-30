'''
The GreedyCoupler is the primary way we interface with an application to perform 
greedy sampling. We can leverage **any** class meeting this API to perform greedy.

We provide a partially concrete implementation of the greedy class. Our implementation 
establishes aspects such as directory structure, but leaves the definition of, e.g., how 
a ROM should be run or how the input file should be modified abstract. 

'''

import os
import abc
import numpy as np

class GreedyCouplerBase(abc.ABC):

    __base_directory = os.getcwd() + '/'
    __fom_directory_base_name = 'fom_run'
    __rom_directory_base_name = 'rom_run'

    def __init__(self,template_directory,template_fom_file,template_rom_file,work_directory_base_name = 'work'):
      strings_to_check = min(5,np.size(template_directory))
      assert self.__base_directory[0:strings_to_check] == template_directory[0:strings_to_check], 'Path to template directory must be an absolute path' 
      self.__template_directory = template_directory 
      self.__template_rom_file = template_rom_file
      self.__template_fom_file = template_fom_file
      self.__work_directory_base_name = work_directory_base_name


    def getRomInputFileName(self):
      return self.__template_rom_file

    def getFomInputFileName(self):
      return self.__template_fom_file

    def getFomDirectoryBaseName(self):
      return self.__fom_directory_base_name

    def getRomDirectoryBaseName(self):
      return self.__rom_directory_base_name

    def getBaseDirectory(self):
      return self.__base_directory

    def getWorkDirectoryBaseName(self):
      return self.__work_directory_base_name

    def createFomAndRomCases(self,starting_sample_no,parameter_samples):
        n_samples = parameter_samples.shape[0]

        path_to_work_dir = self.__base_directory + '/' + self.__work_directory_base_name + '/'
        if os.path.isdir(path_to_work_dir):
            pass
        else:
            os.mkdir(path_to_work_dir)

        for sample_no in range(starting_sample_no,starting_sample_no + n_samples):
            path_to_dir = self.__base_directory +  '/' + self.__work_directory_base_name + '/' + '/fom_run_' + str(sample_no) 
            if os.path.isdir(path_to_dir):
                pass
            else:
                os.mkdir(path_to_dir)
            self.__setupFomCase(path_to_dir,parameter_samples[sample_no - starting_sample_no])

            path_to_dir = self.__base_directory + '/' + self.__work_directory_base_name + '/' + '/rom_run_' + str(sample_no) 
            if os.path.isdir(path_to_dir):
                pass
            else:
                os.mkdir(path_to_dir)
            self.__setupRomCase(path_to_dir,parameter_samples[sample_no - starting_sample_no])

    def computeError(self,rom_directory,fom_directory):
      os.chdir(rom_directory)
      rom_qoi = self.computeQoi()
      os.chdir(self.__base_directory)
      os.chdir(fom_directory)
      fom_qoi = self.computeQoi()
      os.chdir(self.__base_directory)
      error = np.linalg.norm(fom_qoi - rom_qoi) / np.linalg.norm(fom_qoi)
      return error 
 
    def __setupRomCase(self,path_to_rom_case,parameter_samples):
      os.chdir(path_to_rom_case)
      os.system('cp ' + self.__template_directory + '/' + self.__template_rom_file + ' . ')
      self.setParametersInRomInput(self.__template_rom_file,parameter_samples)
      os.chdir(self.getBaseDirectory())

    def __setupFomCase(self,path_to_fom_case,parameter_samples):
      os.chdir(path_to_fom_case)
      os.system('cp ' + self.__template_directory + '/' + self.__template_fom_file + ' . ')
      self.setParametersInFomInput(self.__template_fom_file,parameter_samples)
      os.chdir(self.getBaseDirectory())

    @abc.abstractmethod
    def setParametersInRomInput(self,filename,parameter_sample):
        '''
        This function is called from a run directory. It needs to update a 
        template ROM file with parameter values defined in parameter_sample. 
        For example, this could be done with dprepro
        '''  
        pass

    @abc.abstractmethod
    def setParametersInFomInput(self,filename,parameter_sample):
        '''
        This function is called from a run directory. It needs to update a 
        template FOM file with parameter values defined in parameter_sample. 
        For example, this could be done with dprepro
        '''  
        pass

    @abc.abstractmethod
    def computeQoi(self):
        '''
        This function needs to return a scalar qoi. 
        '''  
        return 0

    @abc.abstractmethod
    def computeErrorIndicator(self):
        '''
        This function is called from a run directory. It needs to return a scalar error estimate. 
        '''  
        return 0

    @abc.abstractmethod
    def runRom(self,filename,parameter_values):
        '''
        This function is called from a run directory. It needs to execute a ROM. 
        '''  
        pass

    @abc.abstractmethod
    def runFom(self,filename,parameter_values):
        '''
        This function is called from a run directory. It needs to execute a FOM. 
        '''  
        pass 

    @abc.abstractmethod
    def createTrialSpace(self,training_sample_indices):
        '''
        This function is called the base directory. Given the FOM runs as defined by training_sample_indices, it needs to compute and save a trial space 
        '''  
        pass 

    @abc.abstractmethod
    def getParameterSpace():
        '''
        This function should return a ParameterSpace class defining our parameter space.
        '''
        pass 


