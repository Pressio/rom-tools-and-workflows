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

'''Implementation of DEIM technique for hyper-reduction'''

import numpy as np


def deim_get_approximation_matrix(function_basis, sample_indices):
    '''
    Given a function basis $\\mathbf{U}$ and sample indices defining $\\mathbf{P}$, we compute
    $$ \\mathbf{U} \\mathrm{pinv}( \\mathbf{P}^T \\mathbf{U})$$
    which comprises the matrix need for the DEIM approximation to $\\mathbf{f}$
    '''
    sampled_function_basis = function_basis[sample_indices]
    PU_pinv = np.linalg.pinv(sampled_function_basis)
    result = function_basis @ PU_pinv
    return result


def deim_get_test_basis(test_basis, function_basis, sample_indices):
    '''
    Given a test basis $\\mathbf{\\Phi}$, a function basis $\\mathbf{U}$, and
    sample indices defining $\\mathbf{P}$, we compute
    $$[ \\mathbf{\Phi}^T \\mathbf{U} \\mathrm{pinv}( \\mathbf{P}^T \\mathbf{U}) ]^T$$
    which comprises the "test basis" for the DEIM approximation for
    $\\mathbf{\Phi}^T \\mathbf{f}$
    '''
    sampled_function_basis = function_basis[sample_indices]
    PU_pinv = np.linalg.pinv(sampled_function_basis)
    result = (test_basis.transpose() @ function_basis) @ PU_pinv
    return result.transpose()

# def vectorDeimGetIndices(U,n_var,variable_ordering='F'):
#   '''
#   Version of DEIM for multi-state systems.
#   We perform DEIM on each state variable, and
#   then return the union of all indices.
#   Repeated indices are removed.
#   '''
#   all_indices = np.zeros(0,dtype=int)
#   for i in range(0,n_var):
#     dataMatrix = __getDataMatrixForIthVar(i,n_var,U,variable_ordering)
#     indices = deim(dataMatrix)
#     all_indices = np.unique(np.append(all_indices,indices))
#   return all_indices


def deim_get_indices(U):
    '''
    Implementation of the discrete empirical method as described in Algorithm 1 of
    S. Chaturantabut and D. C. Sorensen, "Discrete Empirical Interpolation for
    nonlinear model reduction," doi: 10.1109/CDC.2009.5400045.

    Args:
        $\\mathbf{U} \\in \\mathbb{R}^{m \\times n}$, where m is the number of
        DOFs and n the number of samples

    Returns:
        $\\mathrm{indices} \\in \\mathbb{I}^{n}$
    '''

    m = np.shape(U)[1]
    first_index = np.argmax(np.abs(U[:, 0]))
    indices = first_index
    for ell in range(1, m):
        LHS = U[indices, 0:ell]
        RHS = U[indices, ell]
        if ell == 1:
            LHS = np.ones((1, 1))*LHS
            RHS = np.ones(1)*RHS
        C = np.linalg.solve(LHS, RHS)

        residual = U[:, ell] - U[:, 0:ell] @ C
        index_to_add = np.argmax(np.abs(residual))
        indices = np.append(indices, index_to_add)
    return indices
