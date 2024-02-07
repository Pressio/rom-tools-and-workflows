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
___
##**Notes**
The vector defining the affine offset for a linear subspace is viewed as a matrix of shape
$$\\mathbf{u}_{\\mathrm{shift}} \in \\mathbb{R}^{N_{\\mathrm{vars}} \\times N_{\mathrm{x}} }$$

___
##**Theory**

*What is a shift vector, and why would I use it?* In ROMs, we restrict a state to belong to a low-dimensional affine
vector space,
$$\\mathbf{u} \\approx \\tilde{\\mathbf{u}} \\in \\mathcal{V} + \\mathbf{u}_{\\mathrm{shift}}$$
where
$\\mathcal{V} \\equiv \\mathrm{range}(\\boldsymbol \\Phi) $. Here $\\mathbf{u}_{\\mathrm{shift}}$ defines an affine offset.
Affine offsets can be useful for a variety of reasons, including satisfying boundary conditions, and satisfying initial
conditions.

The Shifter class encapsulates the affine offset.

___
##**API**
'''

from typing import Tuple
import abc
import numpy as np


class __Shifter():
    '''
    Shifts the data by a vector.
    '''
    def __init__(self, shift_vector: np.ndarray, my_array: np.ndarray) -> None:
        '''
        Constructor for VectorShifter.

        Args:
            shift_vector (np.ndarray): The vector to shift the data by.
        '''
        self.__shift_vector = shift_vector.copy()
        self.__my_array = my_array

    def apply_shift(self):
        self.__my_array -= self.__shift_vector[..., None]

    def apply_inverse_shift(self):
        self.__my_array += self.__shift_vector[..., None]

    def get_shift_vector(self):
        return self.__shift_vector


def create_noop_shifter(my_array: np.ndarray):
    shift_vector = np.zeros((my_array.shape[0],my_array.shape[1]))
    shifter = __Shifter(shift_vector, my_array)
    return shifter

def create_constant_shifter(shift_value: np.ndarray, my_array: np.ndarray):
    shift_vector = np.empty((my_array.shape[0], my_array.shape[1],))
    assert my_array.shape[0] == shift_value.size
    for i in range(0, my_array.shape[0]):
        shift_vector[i] = shift_value[i]
    shifter = __Shifter(shift_vector, my_array)
    return shifter

def create_vector_shifter(shift_vector: np.ndarray, my_array: np.ndarray):
    shifter = __Shifter(shift_vector, my_array)
    return shifter

def create_average_shifter(my_array: np.ndarray):
    shift_vector = np.mean(my_array, axis=2)
    shifter = __Shifter(shift_vector, my_array)
    return shifter

def create_firstvec_shifter(my_array: np.ndarray):
    shift_vector = my_array[:, :, 0]
    shifter = __Shifter(shift_vector, my_array)
    return shifter
