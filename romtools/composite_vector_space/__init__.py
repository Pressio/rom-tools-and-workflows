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

from romtools.vector_space import *
from romtools.vector_space.utils import *

import numpy as np 
class CompositeVectorSpace:
    '''
    Constructs a composite vector space out of a list of vector spaces
    Different vector spaces need to have the same number of spatial DOFs
    '''

    def __init__(self,list_of_vector_spaces : list[VectorSpace], variable_ordering=None, compact_representation=False):
        '''
        Inputs: list_of_vector_spaces: list[VectorSpace] containing the list of vector spaces to combine
                variable_ordering: Either None or np.ndarray of integers. Specifies the desired variable ordering.
                compact_representation: bool. If True, we compute a global basis and shift. If false, we only store a list of bases and shifts
        '''
        if compact_representation == True and (variable_ordering is None) == False:
            print('Warning, compact representations do not reflect variable ordering')

        self.__compact_representation = compact_representation
        self.list_of_vector_spaces_ = list_of_vector_spaces

        ## Computed dimensions and ensure vector spaces are compatable 
        dims = np.zeros(len(list_of_vector_spaces))
        self.n_vector_spaces_ = len(list_of_vector_spaces)
        n_vars = 0
        total_number_of_bases = 0
        for i in range(0,self.n_vector_spaces_):
            local_vector_space = list_of_vector_spaces[i]
            local_vector_space_dimensions = local_vector_space.get_basis().shape#get_dimension()
            n_vars += local_vector_space_dimensions[0]
            dims[i] = local_vector_space_dimensions[1]
            total_number_of_bases += local_vector_space_dimensions[2]

        nx = int(dims[0])
        self.__n_vars = int(n_vars)
        self.__nx = int(dims[0])
        self.__total_number_of_bases = int(total_number_of_bases)
        self.__extent = np.array([self.__n_vars,self.__nx,self.__total_number_of_bases],dtype='int')
        assert np.allclose( np.diff(dims),np.zeros(dims.size-1)) , " Error constructing composite vector space, not all spaces have the same spatial dimension "

        if variable_ordering is None:
            variable_ordering = np.arange(0,n_vars,dtype='int')
        else:
            ## Check that we have a valid variable ordering
            variables_left = np.arange(0,n_vars,dtype='int')
            assert(len(variable_ordering) == n_vars)
            for variable in variable_ordering:
                assert np.isin(variable,variables_left)
                variables_left = np.delete(variables_left,np.where(variable==variables_left))

        ## Construct a global shift vector
        self.__shift_vector = list_of_vector_spaces[0].get_shift_vector()
        for i in range(1,self.n_vector_spaces_):
          local_shift_vector = list_of_vector_spaces[i].get_shift_vector()
          self.__shift_vector = np.append(self.__shift_vector,local_shift_vector,axis=0)

        # Re-order according to variable ordering
        self.__shift_vector = self.__shift_vector[variable_ordering]

        ## Compute basis
        if compact_representation == False:
            # If False, return a global array
            self.__basis = np.zeros((n_vars,nx,total_number_of_bases))
            start_var_index = 0
            start_basis_index = 0
            for i in range(0,self.n_vector_spaces_):
                local_vector_space = self.list_of_vector_spaces_[i]
                local_vector_space_dimensions = local_vector_space.get_dimension()
                local_basis = local_vector_space.get_basis()
                dim = local_basis.shape
                self.__basis[start_var_index:start_var_index+dim[0],:,start_basis_index:start_basis_index+dim[2]] = local_basis
                start_var_index += dim[0]
                start_basis_index += dim[2]
            self.__basis = self.__basis[variable_ordering]

        else:
            # If true, return a list of the bases.
            # This is much more efficient in terms of memory 
            self.__compact_basis = []
            self.__compact_shift_vector = []

            for i in range(0,self.n_vector_spaces_):
                self.__compact_basis.append(self.list_of_vector_spaces_[i].get_basis())
                self.__compact_shift_vector.append(self.list_of_vector_spaces_[i].get_shift_vector())

 

    def extents(self) -> np.ndarray:
        return self.__extent

    def get_shift_vector(self) -> np.ndarray:
        return self.__shift_vector


    def get_basis(self) -> np.ndarray:
        assert self.__compact_representation == False, "Error, CompositeVectorSpace was constructed with a compact representation and does not have a direct global basis. Use get_compact_basis"
        return self.__basis

    def get_compact_basis(self) -> list[np.ndarray]:
        assert self.__compact_representation == True , 'Error, CompositeVectorSpace was constructed with a full basis representation and does not have a compact basis representation. Use get_basis'
        return self.__compact_basis

    def get_compact_shift_vector(self) -> list[np.ndarray]:
        assert self.__compact_representation == True , 'Error, CompositeVectorSpace was constructed with a full basis representation and does not have a compact basis representation. Use get_basis'
        return self.__compact_shift_vector



if __name__=="__main__":
    ## Basic check 
    #=======================================
    snapshots = np.random.normal(size=(5,8,4))
    v1 = VectorSpaceFromPOD(snapshots[0:2])
    v2 = VectorSpaceFromPOD(snapshots[2:4])
    v3 = VectorSpaceFromPOD(snapshots[4::])

    V = CompositeVectorSpace([v1,v2,v3])
    global_basis = V.get_basis()
    global_shift = V.get_shift_vector()

    v1b = v1.get_basis()
    v2b = v2.get_basis()
    v3b = v3.get_basis()
    v1v = v1.get_shift_vector()
    v2v = v2.get_shift_vector()
    v3v = v3.get_shift_vector()
    

    global_basis_2 = np.zeros((5,8,12))
    global_basis_2[0:2,:,0:4] = v1b
    global_basis_2[2:4,:,4:8] = v2b
    global_basis_2[4::,:,8::] = v3b

    global_shift_2 = np.append(v1v,v2v,axis=0)
    global_shift_2 = np.append(global_shift_2,v3v,axis=0)

    assert np.allclose(global_basis,global_basis_2)
    assert np.allclose(global_shift,global_shift_2)
    assert np.allclose(global_basis.shape, V.extents()) 
    ## Check with non-uniform sized spaces 
    #=======================================
    snapshots = np.random.normal(size=(5,8,4))
    my_truncater = BasisSizeTruncater(3) 
    v1 = VectorSpaceFromPOD(snapshots[0:2],my_truncater)
    my_truncater = BasisSizeTruncater(2) 
    v2 = VectorSpaceFromPOD(snapshots[2:4],my_truncater)
    my_truncater = BasisSizeTruncater(4) 
    v3 = VectorSpaceFromPOD(snapshots[4::],my_truncater)

    V = CompositeVectorSpace([v1,v2,v3])
    global_basis = V.get_basis()
    global_shift = V.get_shift_vector()

    v1b = v1.get_basis()
    v2b = v2.get_basis()
    v3b = v3.get_basis()
    v1v = v1.get_shift_vector()
    v2v = v2.get_shift_vector()
    v3v = v3.get_shift_vector()

    global_basis_2 = np.zeros((5,8,9))
    global_basis_2[0:2,:,0:3] = v1b
    global_basis_2[2:4,:,3:5] = v2b
    global_basis_2[4::,:,5::] = v3b

    global_shift_2 = np.append(v1v,v2v,axis=0)
    global_shift_2 = np.append(global_shift_2,v3v,axis=0)

    assert np.allclose(global_basis,global_basis_2)
    assert np.allclose(global_shift,global_shift_2)

    ## Check with non-uniform orderings
    #=========================================
    snapshots = np.random.normal(size=(5,8,4))
    my_truncater = BasisSizeTruncater(3) 
    v1 = VectorSpaceFromPOD(snapshots[0:2],my_truncater)
    my_truncater = BasisSizeTruncater(2) 
    v2 = VectorSpaceFromPOD(snapshots[2:4],my_truncater)
    my_truncater = BasisSizeTruncater(4) 
    v3 = VectorSpaceFromPOD(snapshots[4::],my_truncater)

    variable_ordering = np.array([3,2,1,0,4])
    V = CompositeVectorSpace([v1,v2,v3],variable_ordering = variable_ordering)
    global_basis = V.get_basis()
    global_shift = V.get_shift_vector()

    v1b = v1.get_basis()
    v2b = v2.get_basis()
    v3b = v3.get_basis()
    v1v = v1.get_shift_vector()
    v2v = v2.get_shift_vector()
    v3v = v3.get_shift_vector()

    global_basis_2 = np.zeros((5,8,9))
    global_basis_2[0:2,:,0:3] = v1b
    global_basis_2[2:4,:,3:5] = v2b
    global_basis_2[4::,:,5::] = v3b
    global_basis_2 = global_basis_2[variable_ordering]

    global_shift_2 = np.append(v1v,v2v,axis=0)
    global_shift_2 = np.append(global_shift_2,v3v,axis=0)
    global_shift_2 = global_shift_2[variable_ordering]

    assert np.allclose(global_basis,global_basis_2)
    assert np.allclose(global_shift,global_shift_2)

    # Test compact representation
    #=============================================
    snapshots = np.random.normal(size=(5,8,4))
    my_truncater = BasisSizeTruncater(3) 
    v1 = VectorSpaceFromPOD(snapshots[0:2],my_truncater)
    my_truncater = BasisSizeTruncater(2) 
    v2 = VectorSpaceFromPOD(snapshots[2:4],my_truncater)
    my_truncater = BasisSizeTruncater(4) 
    v3 = VectorSpaceFromPOD(snapshots[4::],my_truncater)

    print('ok')
    V = CompositeVectorSpace([v1,v2,v3],variable_ordering,compact_representation=True)
    try: 
        global_basis = V.get_basis()
        Failed = False
    except:
        Failed = True
    assert Failed == True, "Shouldn't have been able to construct a basis from a compact representation"
    global_compact_basis = V.get_compact_basis()
    global_shift = V.get_shift_vector()
    global_compact_shift = V.get_compact_shift_vector()

    v1b = v1.get_basis()
    v2b = v2.get_basis()
    v3b = v3.get_basis()
    v1v = v1.get_shift_vector()
    v2v = v2.get_shift_vector()
    v3v = v3.get_shift_vector()

    global_shift_2 = np.append(v1v,v2v,axis=0)
    global_shift_2 = np.append(global_shift_2,v3v,axis=0)
    vbs = [v1b,v2b,v3b]
    shifts = [v1v,v2v,v3v]
    for i in range(0,3):
      assert np.allclose(global_compact_basis[i],vbs[i])
      assert np.allclose(global_compact_shift[i],shifts[i])
    assert(np.allclose(global_shift,global_shift_2))

    ## Check that an invalid ordering fails
    #===============================================
    variable_ordering = np.array([3,2,1,0])
    try: 
        V = CompositeVectorSpace([v1,v2,v3],variable_ordering = variable_ordering)
        Failed = False
    except:
        Failed = True
    assert Failed == True, "Shouldn't have been able to construct the vector space"

    variable_ordering = np.array([3,2,1,0,4,6])
    try: 
        V = CompositeVectorSpace([v1,v2,v3],variable_ordering = variable_ordering)
        Failed = False
    except:
        Failed = True
    assert Failed == True, "Shouldn't have been able to construct the vector space"

