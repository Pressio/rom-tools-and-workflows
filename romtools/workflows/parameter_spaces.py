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
Model reduction is often focused on parameterized PDEs, where $\\boldsymbol \\mu$ is the parameter set.
The ParameterSpace class encapsulates the notion of the parameter space.
'''
import abc
import numpy as np
from typing import Iterable


class Parameter(abc.ABC):
    '''Abstract implementation'''

    @abc.abstractmethod
    def get_name(self) -> str:
        '''
        Returns parameter name
        '''

    @abc.abstractmethod
    def get_dimensionality(self) -> int:
        '''
        Returns dimensionality of parameter for vector quantities.
        Returns 1 for scalar parameters
        '''

    @abc.abstractmethod
    def generate_samples(self, number_of_samples) -> np.array:
        '''
        Generates and returns number of parameter samples

        Returns np.array of shape (number_of_samples, self.get_dimensionality())
        '''


class ParameterSpace(abc.ABC):

    ''' Abstract implementation'''
    @abc.abstractmethod
    def get_parameter_list(self) -> Iterable[Parameter]:
        '''
        Returns a list of Parameter objects
        '''

    def get_names(self) -> Iterable[str]:
        '''
        Returns a list of parameter names
        # e.g., ['sigma','beta',...]
        '''
        return [p.get_name() for p in self.get_parameter_list()]

    def get_dimensionality(self) -> int:
        '''
        Returns an integer for the size
        of the parameter domain
        '''
        return sum(p.get_dimensionality() for p in self.get_parameter_list())

    def generate_samples(self, number_of_samples) -> np.array:
        '''
        Generates and returns number of parameter samples

        Returns np.array of shape (number_of_samples, self.get_dimensionality())
        '''
        samples = [p.generate_samples(number_of_samples)
                   for p in self.get_parameter_list()]
        return np.concatenate(samples, axis=1)


class UniformParameter(Parameter):
    '''
    Uniformly distributed floating point
    '''
    def __init__(self, parameter_name: str,
                 lower_bound: float = 0,
                 upper_bound: float = 1):
        self._parameter_name = parameter_name

        try:
            assert len(lower_bound) == len(upper_bound)
            self._dimension = len(lower_bound)
        except TypeError:
            self._dimension = 1
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def get_name(self) -> str:
        return self._parameter_name

    def get_dimensionality(self) -> int:
        return self._dimension

    def generate_samples(self, number_of_samples) -> np.array:
        return np.random.uniform(self._lower_bound, self._upper_bound,
                                 size=(number_of_samples,
                                       self.get_dimensionality()))


class StringParameter(Parameter):
    '''
    Constant string-valued parameter
    '''
    def __init__(self, parameter_name: str, value):
        self._parameter_name = parameter_name
        self._parameter_value = value

    def get_name(self) -> str:
        return self._parameter_name

    def get_dimensionality(self) -> int:
        return 1

    def generate_samples(self, number_of_samples) -> np.array:
        return np.array([[self._parameter_value]] * number_of_samples)


class UniformParameterSpace(ParameterSpace):
    '''
    Concrete implementation for a uniform parameter space with random sampling
    '''

    def __init__(self, parameter_names: Iterable[str],
                 lower_bounds, upper_bounds):
        self.parameters = [UniformParameter(name, lb, ub)
                           for name, lb, ub
                           in zip(parameter_names, lower_bounds, upper_bounds)]

    def get_parameter_list(self) -> Iterable[Parameter]:
        return self.parameters


class ConstParameterSpace(ParameterSpace):
    '''
    Constant parameter space which converts all constant values to str-type

    Useful if you need to execute workflows in a non-stochastic setting
    '''
    def __init__(self, parameter_names: Iterable[str], parameter_values):
        self.parameters = [StringParameter(name, val)
                           for name, val
                           in zip(parameter_names, parameter_values)]

    def get_parameter_list(self) -> Iterable[Parameter]:
        return self.parameters


class HeterogeneousParameterSpace(ParameterSpace):
    '''
    Heterogeneous parameter space consisting of a list of arbitrary Parameter objects
    '''
    def __init__(self, parameter_objs: Iterable[Parameter]):
        self.parameters = parameter_objs

    def get_parameter_list(self) -> Iterable[Parameter]:
        return self.parameters
