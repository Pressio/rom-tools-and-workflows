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
Model reduction is often focused on parameterized PDEs, where
$\\boldsymbol \\mu$ is the parameter set.
The ParameterSpace class encapsulates the notion of the parameter space.
'''
import abc
from typing import Iterable
import numpy as np
from scipy.stats import qmc


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
    def generate_samples(self, uniform_dist_samples) -> np.array:
        '''
        Generates samples from the desired distribution given a set of samples
        from a uniform distribution on (0,1)

        uniform_dist_samples should be of shape
        (number_of_samples, self.get_dimensionality())

        Returns np.array of the same shape
        '''


class ParameterSpace(abc.ABC):

    ''' Abstract implementation'''
    @abc.abstractmethod
    def get_names(self) -> Iterable[str]:
        '''
        Returns a list of parameter names
        # e.g., ['sigma','beta',...]
        '''

    @abc.abstractmethod
    def get_dimensionality(self) -> int:
        '''
        Returns an integer for the size
        of the parameter domain
        '''

    @abc.abstractmethod
    def generate_samples(self, uniform_dist_samples: np.array) -> np.array:
        '''
        Generates samples from the parameter space given a set of samples
        from a uniform distribution on (0,1)

        uniform_dist_samples should be of shape
        (number_of_samples, self.get_dimensionality())

        Returns np.array of the same shape
        '''


##########################################
# Sampling Methods
##########################################


def monte_carlo_sample(param_space: ParameterSpace,
                       number_of_samples: int = 1,
                       seed=None):
    '''
    Generate Monte Carlo samples from a given parameter space
    '''
    if seed is not None:
        np.random.seed(seed)
    uniform_dist_sample = np.random.uniform(
        size=(number_of_samples, param_space.get_dimensionality()))
    return param_space.generate_samples(uniform_dist_sample)


def latin_hypercube_sample(param_space: ParameterSpace,
                           number_of_samples: int = 1,
                           seed=None):
    '''
    Generate LHS samples from a given parameter space
    '''
    sampler = qmc.LatinHypercube(param_space.get_dimensionality(), seed=seed)
    uniform_dist_sample = sampler.random(n=number_of_samples)
    return param_space.generate_samples(uniform_dist_sample)


##########################################
# Concrete Parameter Classes
##########################################


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

    def generate_samples(self, uniform_dist_samples: np.array) -> np.array:
        assert uniform_dist_samples.shape[1] == self.get_dimensionality()
        return qmc.scale(uniform_dist_samples,
                         self._lower_bound,
                         self._upper_bound)


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

    def generate_samples(self, uniform_dist_samples: np.array) -> np.array:
        assert uniform_dist_samples.shape[1] == self.get_dimensionality()
        number_of_samples = uniform_dist_samples.shape[0]
        return np.array([[self._parameter_value]] * number_of_samples)


##########################################
# Concrete ParameterSpace Classes
##########################################


class HeterogeneousParameterSpace(ParameterSpace):
    '''
    Heterogeneous parameter space consisting of a list of arbitrary Parameter
    objects
    '''
    def __init__(self, parameter_objs: Iterable[Parameter]):
        self._parameters = parameter_objs

    def _get_parameter_list(self) -> Iterable[Parameter]:
        '''
        Returns a list of Parameter objects
        '''
        return self._parameters

    def get_names(self) -> Iterable[str]:
        return [p.get_name() for p in self._get_parameter_list()]

    def get_dimensionality(self) -> int:
        return sum(p.get_dimensionality() for p in self._get_parameter_list())

    def generate_samples(self, uniform_dist_samples: np.array) -> np.array:
        samples = []
        param_idx = 0
        for param in self._get_parameter_list():
            next_param_idx = param_idx+param.get_dimensionality()
            param_samples = param.generate_samples(
                uniform_dist_samples[:, param_idx:next_param_idx])
            samples.append(param_samples)
            param_idx = next_param_idx
        return np.concatenate(samples, axis=1)


class HomogeneousParameterSpace(HeterogeneousParameterSpace):
    '''
    Homogenous parameter space in which every parameter is of the same type
    '''
    def __init__(self, parameter_names: Iterable[str],
                 param_constructor, **kwargs):
        parameters = []
        for param_num, param_name in enumerate(parameter_names):
            args = {key: val[param_num] for key, val in kwargs.items()}
            parameters.append(param_constructor(parameter_name=param_name,
                                                **args))
        super().__init__(parameters)


class UniformParameterSpace(HomogeneousParameterSpace):
    '''
    Homogeneous parameter space in which every parameter is a UniformParameter
    '''

    def __init__(self, parameter_names: Iterable[str],
                 lower_bounds, upper_bounds):
        super().__init__(parameter_names, UniformParameter,
                         lower_bound=lower_bounds,
                         upper_bound=upper_bounds)


class ConstParameterSpace(HomogeneousParameterSpace):
    '''
    Homogeneous parameter space in which every parameter is a constant
    StringParameter. All numeric values are converted to str-type.

    Useful if you need to execute workflows in a non-stochastic setting
    '''
    def __init__(self, parameter_names: Iterable[str], parameter_values):
        super().__init__(parameter_names, StringParameter,
                         value=parameter_values)
