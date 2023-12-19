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
The GreedyCoupler is the primary way we interface with an application to
perform greedy sampling. We can leverage **any** class meeting this API to
perform greedy.

We provide a partially concrete implementation of the greedy class. Our
implementation establishes aspects such as directory structure, but leaves the
definition of, e.g., how a ROM should be run or how the input file should be
modified abstract.
'''

import os
import abc
import numpy as np

from romtools.workflows.workflow_utils import setup_directory
from romtools.workflows.sampling.\
    sampling_coupler_base import SamplingCouplerBase


class GreedyCouplerBase(abc.ABC):
    def __init__(self, rom_coupler: SamplingCouplerBase,
                 fom_coupler: SamplingCouplerBase,
                 base_directory: str = None):
        '''
        Initializes a GreedyCouplerBase object.

        Args:
            rom_coupler (SamplingCouplerBase):
            fom_coupler (SamplingCouplerBase):
            base_directory (str, optional): The working directory. Defaults to None (current directory).
        '''
        self.rom_coupler = rom_coupler
        self.fom_coupler = fom_coupler
        self.__base_directory = (os.path.realpath(os.getcwd())
                                 if base_directory is None
                                 else base_directory
                                 )

    def get_rom_input_filename(self):
        '''Get the name of the ROM input file.'''
        return self.rom_coupler.get_input_filename()

    def get_fom_input_filename(self):
        '''Get the name of the FOM input file.'''
        return self.fom_coupler.get_input_filename()

    def get_fom_directory_basename(self):
        '''Get the base name for the FOM directory.'''
        return self.fom_coupler.get_base_directory()

    def get_rom_directory_basename(self):
        '''Get the base name for the ROM directory.'''
        return self.rom_coupler.get_base_directory()

    def get_base_directory(self):
        '''Get the base directory for the coupler.'''
        return self.__base_directory

    def create_fom_and_rom_cases(self, starting_sample_no, parameter_samples):
        '''
        Create FOM and ROM cases with parameter samples.

        Args:
            starting_sample_no (int): The starting sample number.
            parameter_samples (np.ndarray): Parameter samples to be used for
                creating cases.
        '''
        self.fom_coupler.create_cases(starting_sample_no, parameter_samples)
        self.rom_coupler.create_cases(starting_sample_no, parameter_samples)

    def compute_error(self, case_num):
        '''
        Compute the error between ROM and FOM cases.

        Args:
            rom_directory (str): Directory path for the ROM case.
            fom_directory (str): Directory path for the FOM case.

        Returns:
            float: The computed error.
        '''
        rom_directory = self.rom_coupler.get_sol_directory(case_num)
        fom_directory = self.fom_coupler.get_sol_directory(case_num)
        os.chdir(rom_directory)
        rom_qoi = self.compute_qoi()
        os.chdir(self.get_base_directory())
        os.chdir(fom_directory)
        fom_qoi = self.compute_qoi()
        os.chdir(self.get_base_directory())
        error = np.linalg.norm(fom_qoi - rom_qoi) / np.linalg.norm(fom_qoi)
        return error

    def run_rom(self, filename, parameter_values):
        '''
        This function is called from a run directory. It needs to execute a
        ROM.
        Args:
          filename (str): The name of the ROM input file.
          parameter_values (np.ndarray): Parameter values for the ROM run.
        '''
        self.rom_coupler.run_model(filename, parameter_values)

    def run_fom(self, filename, parameter_values):
        '''
        This function is called from a run directory. It needs to execute a
        FOM.

        Args:
          filename (str): The name of the FOM input file.
          parameter_values (np.ndarray): Parameter values for the FOM run.
        '''
        self.fom_coupler.run_model(filename, parameter_values)

    @abc.abstractmethod
    def compute_qoi(self):
        '''
        This function needs to return a scalar qoi.
        '''
        return 0

    @abc.abstractmethod
    def compute_error_indicator(self):
        '''
        This function is called from a run directory. It needs to return a
        scalar error estimate.
        '''
        return 0

    @abc.abstractmethod
    def create_trial_space(self, training_sample_indices):
        '''
        This function is called the base directory. Given the FOM runs as
        defined by training_sample_indices, it needs to compute and save a
        trial space.

        Args:
          training_sample_indices (list): Indices of the training samples.
        '''
        pass

    @abc.abstractmethod
    def get_parameter_space(self):
        '''
        This function should return a ParameterSpace class defining our
        parameter space.
        '''
        pass
