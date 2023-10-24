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

"""
Model reduction is often focused on parameterized PDEs, where $\\boldsymbol \\mu$ is the parameter set.
The ParameterSpace class encapsulates the notion of the parameter space.
"""
import abc
import numpy as np
"""The abstract parameter space"""
class AbstractParameterSpace(abc.ABC):

    """ Abstract implementation"""
    @abc.abstractmethod
    def getNames() -> list:
        """
        return a list of parameter names
        # e.g., ['sigma','beta',...]
        """
        pass

    @abc.abstractmethod
    def getDimensionality() -> int:
        """
        returns an integer for the size
        of the parameter domain
        """
        pass

    @abc.abstractmethod
    def generateSamples(self,number_of_samples):
        """
        generates and returns number of parameter samples
        """
        pass


class UniformParameterSpace(AbstractParameterSpace):
    """Concrete implementation for a uniform parameter space with random sampling"""

    def __init__(self,parameter_names,lower_bounds,upper_bounds):
        self.__parameter_names = parameter_names
        self.__lower_bounds = lower_bounds
        self.__upper_bounds = upper_bounds
        self.__n_params = len(self.__lower_bounds)

    def getNames():
        return self.__parameter_names

    def getDimensionality(self):
        return self.__n_params

    def generateSamples(self,number_of_samples):
        samples = np.random.uniform(self.__lower_bounds,self.__upper_bounds,\
                                    size=(number_of_samples,self.__n_params))
        return samples
