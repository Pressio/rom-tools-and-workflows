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

from romtools.vector_space import VectorSpace
from typing import List
import numpy as np


class CompositeVectorSpace(VectorSpace):
    '''
    Constructs a composite vector space out of a list of vector spaces
    Different vector spaces need to have the same number of spatial DOFs
    '''

    def __init__(self, list_of_vector_spaces: List[VectorSpace], variable_ordering=None):
        '''
        Inputs: list_of_vector_spaces: list[VectorSpace] containing the list of vector spaces to combine
                variable_ordering: Either None or np.ndarray of integers. Specifies the desired variable ordering.
        '''
        if variable_ordering is not None:
            print('Warning, compact representations do not reflect variable ordering')

        # Computed dimensions and ensure vector spaces are compatable
        self.__get_extent_and_check_compatability(list_of_vector_spaces)

        # Check that a valid set of variable orderings have been passed in
        self.variable_ordering = self.__check_valid_variable_ordering(variable_ordering)

        # Construct basis as a list of local bases
        self.__construct_compact_basis(list_of_vector_spaces)

    def get_dimension(self):
        return self.__extent[2]

    def extents(self) -> np.ndarray:
        return self.__extent

    def get_shift_vector(self) -> np.ndarray:
        return self.__construct_global_shift_vector()

    def get_basis(self) -> np.ndarray:
        return self.__construct_full_basis(self.__compact_basis, self.variable_ordering)

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
        assert np.allclose(np.diff(dims), np.zeros(dims.size-1)), " Error constructing composite vector space, not all spaces have the same spatial dimension "

    def __check_valid_variable_ordering(self, variable_ordering):
        # Checks that a valid variable ordering has been passed
        # and updates variable_ordering accordingly

        n_vars = self.__extent[0]
        if variable_ordering is None:
            variable_ordering = np.arange(0, n_vars, dtype='int')
        else:
            # Check that we have a valid variable ordering
            variables_left = np.arange(0, n_vars, dtype='int')
            assert len(variable_ordering) == n_vars
            for variable in variable_ordering:
                assert np.isin(variable, variables_left)
                variables_left = np.delete(variables_left, np.where(variable == variables_left))
        return variable_ordering

    def __construct_global_shift_vector(self):
        # Constructs the shift vector for the composite vector space
        shift_vector = self.__compact_shift_vector[0]
        for local_shift_vector in self.__compact_shift_vector[1:]:
            shift_vector = np.append(shift_vector, local_shift_vector, axis=0)

        # Re-order according to variable ordering
        shift_vector = shift_vector[self.variable_ordering]
        return shift_vector

    def __construct_full_basis(self, list_of_vector_spaces, variable_ordering):
        # Constructs a dense basis for the composite vector space
        basis = np.zeros((self.__extent[0], self.__extent[1], self.__extent[2]))
        start_var_index = 0
        start_basis_index = 0
        for local_vector_space in list_of_vector_spaces:
            local_basis = local_vector_space.get_basis()
            dim = local_basis.shape
            basis[start_var_index:start_var_index+dim[0], :, start_basis_index:start_basis_index+dim[2]] = local_basis
            start_var_index += dim[0]
            start_basis_index += dim[2]
        return basis[variable_ordering]

    def __construct_compact_basis(self, list_of_vector_spaces):
        # Constructs a list of bases.
        # This is much more efficient in terms of memory
        self.__compact_basis = [space.get_basis() for space in list_of_vector_spaces]
        self.__compact_shift_vector = [space.get_shift_vector() for space in list_of_vector_spaces]
