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
import romtools
import numpy as np
from romtools.vector_space import VectorSpace
def optimal_l2_projection(input_tensor : np.ndarray , vector_space : romtools.VectorSpace , weighting_matrix : np.ndarray = None, return_full_state=False):

    '''
    Compute L2 projection in the weighted inner product.

    .. math::

       \\arg\\min \\| ( \\Phi \\hat{x} + x_{ref}) - x \\|_M^2

    Solution satisfies the linear system

    .. math::

       \\Phi^T M \\Phi \\hat{x} = \\Phi^T M ( x - x_{ref} )

    Args:
        input_tensor (np.ndarray): 2d or 3d data array of size (n_vars, nx) or (n_vars, nx, n_snaps)
        vector_space (romtools.VectorSpace): vector space class containing basis and affine offset
        weighting_matrix (np.ndarray): 2d weighting matrix of size :math:`nvars nx \\times nvars nx`
    '''

    basis = vector_space.get_basis()
    shift_vector = vector_space.get_shift_vector()

    nvars,nx,k = basis.shape
    basis = np.reshape(basis,(nvars*nx,k),'C')

    # If input_tensor is just a single snapshot
    if len(input_tensor.shape) == 2:
        right_hand_side = (input_tensor - shift_vector).flatten()

    # If input_tensor is many snapshots
    elif len(input_tensor.shape) == 3:
        right_hand_side = input_tensor - shift_vector[...,None]
        right_hand_side = np.reshape(right_hand_side,(nvars*nx,input_tensor.shape[-1]))


    if weighting_matrix is None:
        left_hand_side = basis.transpose() @  basis 
        right_hand_side = basis.transpose() @ right_hand_side

    else:
        left_hand_side = basis.transpose() @ ( weighting_matrix @ basis )
        right_hand_side = basis.transpose() @ ( weighting_matrix  @ right_hand_side )

    reduced_state = np.linalg.solve(left_hand_side,right_hand_side)
    basis = np.reshape(basis,(nvars,nx,k),'C')
  
    if return_full_state == False:
        return reduced_state 

    if len(input_tensor.shape) == 2:
        full_state = np.einsum('nik,k...->ni...',basis,reduced_state) + shift_vector
    else:
        full_state = np.einsum('nik,k...->ni...',basis,reduced_state) + shift_vector[...,None]
    return reduced_state,full_state


