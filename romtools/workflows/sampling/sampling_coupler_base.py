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
Abstract implementation of the SamplingCoupler class. To gain access to the
sampling algorithm, the user needs to complete this class for their application
'''
import os
import abc
from typing import Iterable

from romtools.workflows.workflow_utils import setup_directory


class SamplingCouplerBase(abc.ABC):
    '''
    Partially explicit implementation
    '''

    def __init__(self, template_directory: str,
                 template_input_file: str,
                 other_required_files: Iterable = (),
                 base_directory: str = None,
                 sol_directory_basename: str = 'run_'):
        '''
        Initialize a SamplingCouplerBase object.

        Args:
            template_directory (str): The directory containing input file
                templates.
            template_input_file (str): The template input file for the model.
            other_required_files (Iterable, optional): other files (besides
                `template_input_file`) which should be copied into each case
                directory.
            base_directory (str, optional): The working directory for the
                sampling. If not provided, the current directory is used.
            sol_directory_basename (str, optional): The base name for the
                solution directories within the working directory.
                Defaults to 'run_'
        '''

        self.__base_directory = (os.path.realpath(os.getcwd())
                                 if base_directory is None
                                 else base_directory
                                 )
        self.__template_directory = os.path.realpath(template_directory)
        self.__sol_directory_basename = sol_directory_basename

        self.__template_input_file = template_input_file
        self.__required_files = [template_input_file] + list(other_required_files)

    def get_input_filename(self):
        '''Get the name of the template input file.'''
        return self.__template_input_file

    def get_sol_directory_basename(self):
        '''
        Get the base name for the solution directories within the working
        directory.
        '''
        return self.__sol_directory_basename

    def get_sol_directory(self, idx):
        '''
        Get the solution directory for a specific case
        '''
        base_dir = self.get_base_directory()
        return base_dir + f'/{self.get_sol_directory_basename()}{idx}'

    def get_base_directory(self):
        '''Get the base directory for sampling.'''
        return self.__base_directory

    def create_cases(self, starting_sample_no, parameter_samples):
        '''
        Create sampling cases with parameter samples.

        Args:
            starting_sample_no (int): The starting sample number.
            parameter_samples (np.ndarray): An array of parameter samples.
        '''
        end_sample_no = starting_sample_no + parameter_samples.shape[0]

        for sample_no in range(starting_sample_no, end_sample_no):
            setup_directory(source_dir=self.__template_directory,
                            target_dir=self.get_sol_directory(sample_no),
                            files2copy=self.__required_files)

            os.chdir(self.get_sol_directory(sample_no))
            self.set_parameters_in_input(self.__template_input_file,
                                         parameter_samples[sample_no])
            os.chdir(self.get_base_directory())

    @abc.abstractmethod
    def set_parameters_in_input(self, filename, parameter_sample):
        '''
        This function is called from a run directory. It needs to update a
        template file with parameter values defined in parameter_sample.
        For example, this could be done with dprepro
        '''

    @abc.abstractmethod
    def run_model(self, filename, parameter_values):
        '''
        This function is called from a run directory. It needs to execute our
        model.  If the model runs succesfully, return 0.  If fails, return 1.
        '''
        return 0

    @abc.abstractmethod
    def get_parameter_space(self):
        '''
        This function should return a ParameterSpace class defining our
        parameter space.
        '''
