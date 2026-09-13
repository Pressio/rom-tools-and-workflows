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

from typing import List
import numpy as np
from romtools.vector_space import VectorSpace

__all__ = ["CompositeVectorSpace"]


class CompositeVectorSpace:
    '''
    Constructs a composite vector space out of a list of vector spaces
    Different vector spaces need to have the same number of spatial DOFs
    '''

    def __init__(self, list_of_vector_spaces: List[VectorSpace]):
        '''
        Inputs: list_of_vector_spaces: list[VectorSpace] containing the list of vector spaces to combine
        '''
        # Compute dimensions and ensure vector spaces are compatible
        self.__get_extent_and_check_compatability(list_of_vector_spaces)

        # Construct basis as a list of local bases
        self.__construct_compact_basis(list_of_vector_spaces)

    def extents(self) -> np.ndarray:
        return self.__extent

    def get_shift_vector(self) -> np.ndarray:
        return self.__construct_global_shift_vector()

    def get_basis(self) -> np.ndarray:
        return self.__construct_full_basis()

    def get_compact_basis(self) -> List[np.ndarray]:
        return self.__compact_basis

    def get_compact_shift_vector(self) -> List[np.ndarray]:
        return self.__compact_shift_vector

    def __get_extent_and_check_compatability(self, list_of_vector_spaces):
        # Checks that dimensions of the vector spaces match
        # and assigns self.__extent
        dims = np.zeros(len(list_of_vector_spaces))
        n_vector_spaces = len(list_of_vector_spaces)
        n_vars = 0
        total_number_of_bases = 0
        for i in range(0, n_vector_spaces):
            local_vector_space = list_of_vector_spaces[i]
            local_vector_space_dimensions = local_vector_space.get_basis().shape
            n_vars += local_vector_space_dimensions[0]
            dims[i] = local_vector_space_dimensions[1]
            total_number_of_bases += local_vector_space_dimensions[2]

        nx = int(dims[0])
        n_vars = int(n_vars)
        total_number_of_bases = int(total_number_of_bases)
        self.__extent = np.array([n_vars, nx, total_number_of_bases], dtype='int')
        assert np.allclose(np.diff(dims),
                           np.zeros(dims.size-1)
               ), "Error constructing composite vector space, not all spaces have the same spatial dimension"

    def __construct_global_shift_vector(self):
        # Constructs the full shift vector
        return np.concatenate(self.__compact_shift_vector, axis=0)

    def __construct_full_basis(self):
        # Constructs a full block-diagonal basis from the compact bases
        full_basis = np.zeros(self.__extent, dtype=self.__compact_basis[0].dtype)
        var_offset = 0
        basis_offset = 0
        for local_basis in self.__compact_basis:
            n_vars, _, n_basis = local_basis.shape
            full_basis[
                var_offset:var_offset+n_vars,
                :,
                basis_offset:basis_offset+n_basis,
            ] = local_basis
            var_offset += n_vars
            basis_offset += n_basis
        return full_basis

    def __construct_compact_basis(self, list_of_vector_spaces):
        self.__compact_basis = [
            vector_space.get_basis() for vector_space in list_of_vector_spaces
        ]
        self.__compact_shift_vector = [
            vector_space.get_shift_vector() for vector_space in list_of_vector_spaces
        ]
