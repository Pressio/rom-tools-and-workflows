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
Constructing a basis via POD typically entails computing the SVD of a snapshot matrix,
$$ \\mathbf{U} ,\\mathbf{\\Sigma} = \\mathrm{svd}(\\mathbf{S})$$
and then selecting the first $K$ left singular vectors (i.e., the first $K$
columns of $\\mathbf{U}$). Typically, $K$ is determined through the decay of
the singular values.

The truncater class is desined to truncate a basis.
We provide concrete implementations that truncate based on a specified number
of basis vectors and the decay of the singular values
'''

from typing import Protocol
import numpy as np


class LeftSingularVectorTruncater(Protocol):
    '''
    Interface for the Truncater class.
    '''

    def truncate(self, basis: np.ndarray,  singular_values: np.ndarray) -> np.ndarray:
        '''
        Truncate left singular vectors
        '''
        ...


class NoOpTruncater():
    '''
    No op implementation

    This class conforms to `LeftSingularVectorTruncater` protocol.
    '''
    def __init__(self) -> None:
        pass

    def truncate(self, basis: np.ndarray,  singular_values: np.ndarray) -> np.ndarray:
        return basis


class BasisSizeTruncater(LeftSingularVectorTruncater):
    '''
    Truncates to a specified number of singular vectors, as specified in the constructor

    This class conforms to `LeftSingularVectorTruncater` protocol.
    '''
    def __init__(self, basis_dimension: int) -> None:
        '''
        Constructor for the BasisSizeTruncater class.

        Args:
            basis_dimension (int): The desired dimension of the truncated basis.
        '''
        # Check if basis dimension is less than or equal to zero
        if basis_dimension <= 0:
            raise ValueError('Given basis dimension is <= 0: ', basis_dimension)

        self.__basis_dimension = basis_dimension

    def truncate(self, basis: np.ndarray, singular_values: np.ndarray) -> np.ndarray:
        '''
        Truncate the basis based on the specified dimension.

        Args:
            basis (np.ndarray): The original basis matrix.
            singular_values (np.ndarray): The array of singular values associated with the basis matrix.

        Returns:
            np.ndarray: The truncated basis matrix with the specified dimension.
        '''
        # Check if basis dimension is larger than array and give error.
        if self.__basis_dimension > np.shape(basis)[1]:
            raise ValueError('Given basis dimension is greater than size of basis array: ',
                             self.__basis_dimension, ' > ', np.shape(basis)[1])

        return basis[:, :self.__basis_dimension]


class EnergyBasedTruncater(LeftSingularVectorTruncater):
    '''
    Truncates based on the decay of singular values, i.e., will define $K$ to
    be the number of singular values such that the cumulative energy retained
    is greater than some threshold.

    This class conforms to `LeftSingularVectorTruncater` protocol.
    '''
    def __init__(self, threshold: float) -> None:
        '''
        Constructor for the EnergyTruncater class.

        Args:
            threshold (float): The cumulative energy threshold.
        '''
        self.__energy_threshold_ = threshold

    def truncate(self, basis: np.ndarray, singular_values: np.ndarray) -> np.ndarray:
        '''
        Truncate the basis based on the energy threshold.

        Args:
            basis (np.ndarray): The original basis matrix.
            singular_values (np.ndarray): The array of singular values associated with the basis matrix.

        Returns:
            np.ndarray: The truncated basis matrix based on the energy threshold.
        '''
        energy = np.cumsum(singular_values**2)/np.sum(singular_values**2)
        basis_dimension = np.argmax(energy > self.__energy_threshold_) + 1
        return basis[:, 0:basis_dimension]
