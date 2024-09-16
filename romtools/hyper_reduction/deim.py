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
import romtools.linalg.linalg as la



def _deim_get_indices_sharedmem(U):
    '''
    Implementation of the discrete empirical method as described in Algorithm 1 of
    S. Chaturantabut and D. C. Sorensen, "Discrete Empirical Interpolation for
    nonlinear model reduction," doi: 10.1109/CDC.2009.5400045.

    Args:
        $\\mathbf{U} \\in \\mathbb{R}^{m \\times n}$, where
            $m$ is the number of DOFs and
            $n$ is the number of samples.
            Function basis in matrix format

    Returns:
        $\\mathrm{indices} \\in \\mathbb{I}^{n}$: sample mesh indices
    '''

    m = np.shape(U)[1]
    first_index = la.argmax(np.abs(U[:, 0]))
    indices = first_index
    for ell in range(1, m):
        LHS = U[indices, 0:ell]
        RHS = U[indices, ell]
        if ell == 1:
            LHS = np.ones((1, 1))*LHS
            RHS = np.ones(1)*RHS
        C = np.linalg.solve(LHS, RHS)

        residual = U[:, ell] - U[:, 0:ell] @ C
        index_to_add = la.argmax(np.abs(residual))
        indices = np.append(indices, index_to_add)
    return indices


class _DistDeimData:
    def __init__(self, i, r):
        self.local_indices = np.array([int(i)])
        self.owning_ranks = np.array([int(r)])

    def append(self, i, r):
        '''Adds the local index and rank to self.local_indices and self.owning_ranks.'''
        self.local_indices = np.append(self.local_indices , int(i))
        self.owning_ranks = np.append(self.owning_ranks, int(r))


def _deim_get_indices_distributed(U, comm):
    m = np.shape(U)[1]
    local_index, found_rank = la.argmax(np.abs(U[:, 0]), comm)
    result = _DistDeimData(local_index, found_rank)
    if m == 1:
        return result.local_indices, result.owning_ranks

    my_rank = comm.Get_rank()
    LHS, RHS, C = np.array([]), np.array([]), np.array([])
    for ell in range(1, m):
        indices = result.local_indices[result.owning_ranks==my_rank]
        LHS = np.array([]) if indices.size == 0 else U[indices, 0:ell]
        RHS = np.array([]) if indices.size == 0 else U[indices, ell]

        A, b = la.move_distributed_linear_system_to_rank_zero(LHS, RHS, comm)
        if my_rank == 0:
            C = np.linalg.solve(A, b)
        C = comm.bcast(C, root=0)

        residual = U[:, ell] - U[:, 0:ell] @ C
        local_index, found_rank = la.argmax(np.abs(residual), comm)
        result.append(local_index, found_rank)

    return result.local_indices, result.owning_ranks


def deim_get_indices(U, comm=None):
    '''
    Implementation of the discrete empirical method as described in Algorithm 1 of
    S. Chaturantabut and D. C. Sorensen, "Discrete Empirical Interpolation for
    nonlinear model reduction," doi: 10.1109/CDC.2009.5400045.

    Args:
        $\\mathbf{U} \\in \\mathbb{R}^{m \\times n}$, where
            $m$ is the number of DOFs and
            $n$ is the number of samples.
            Function basis in matrix format.
        comm: Optional communicator object. If none, algorithm assumes shared-memory data.

    Returns:
         if comm==None:
            $\\mathrm{indices} \\in \\mathbb{I}^{n}$: sample mesh indices

         else:
            sample mesh local indices and the corresponding owing ranks
    '''

    assert np.shape(U)[1] >= 1, "deim requires a basis matrix with at least one basis vector (one column)"

    if comm:
        return _deim_get_indices_distributed(U, comm)

    return _deim_get_indices_sharedmem(U)


def _deim_multi_state_get_indices_sharedmem(U):
    all_indices = np.zeros(0, dtype=int)
    n_var = U.shape[0]
    for i in range(0, n_var):
        data_matrix = U[i]
        indices = deim_get_indices(data_matrix)
        all_indices = np.unique(np.append(all_indices, indices))
    return all_indices


def _deim_multi_state_get_indices_distributed(U, comm):
    all_local_indices = np.zeros(0, dtype=int)
    all_ranks = np.zeros(0, dtype=int)
    n_var = U.shape[0]
    for i in range(0, n_var):
        data_matrix = U[i]
        indices, ranks = deim_get_indices(data_matrix, comm)
        # print(my_rank, ":::", indices, ranks)
        all_local_indices = np.append(all_local_indices, indices)
        all_ranks = np.append(all_ranks, ranks)

    # from here we need to operate on ranks and indices together
    # so stack them for convenience
    M = np.vstack((all_local_indices, all_ranks))

    # first, sort based on rankID
    M = np.unique(M, axis=1)
    inds = M[1,:].argsort()
    M = M[:, inds]

    # once we are here, M is sorted based on the ranks, but could look like this:

    # M = [ 0 14 13  5 15  4  2  6  8 10  4  4 14  7  8  3  6]
    #     [ 0  0  0  0  0  0  0  1  1  1  1  2  2  2  2  2  2]

    # we do an additional step where we sort based on the local index within each rank

    rank_ids = np.unique(M[1,:])
    for rank in rank_ids:
        locs = np.where(M[1,:] == rank)
        # only need to sort the 0-th row since we are sorting locally for each rank with same rank id
        M[0, locs] = np.sort(M[0, locs])

    # return local indices and ranks separately
    return M[0,:], M[1,:]


def multi_state_deim_get_indices(U, comm=None):
    '''
    Version of DEIM for multi-state systems.

    We perform DEIM on each state variable, and
    then return the union of all indices.
    Repeated indices are removed.


    Args:
         $\\mathbf{U} \\in \\mathbb{R}^{l \\times m \\times n}$, where
            $l$ is the number of variables,
            $m$ is the number of DOFs and
            $n$ the number of samples.
            Multi-dimensional function basis in tensor format.

    Returns:
         if comm==None:
            $\\mathrm{indices} \\in \\mathbb{I}^{n}$: sample mesh indices

         else:
            sample mesh local indices and the corresponding owing ranks
    '''
    shape = np.shape(U)
    assert len(shape) == 3
    assert shape[2] >= 1, "deim requires a basis matrix with at least one basis vector (one column)"

    if comm:
        return _deim_multi_state_get_indices_distributed(U, comm)

    return _deim_multi_state_get_indices_sharedmem(U)



def deim_get_approximation_matrix(function_basis, sample_indices):
    '''
    Given a function basis $\\mathbf{U}$ and sample indices defining $\\mathbf{P}$, we compute
    $$ \\mathbf{U} \\mathrm{pinv}( \\mathbf{P}^T \\mathbf{U})$$
    which comprises the matrix needed for the DEIM approximation to $\\mathbf{f}$

    Args:
        function_basis: ($m$, $n$) array, where $m$ is the number of DOFs and $n$ the number of basis functions.
            Basis for function to be approximated.
        sample_indices: ($n_s$, ) array, where $n_s$ is the number of sample points. Sampling points.

    Returns:
        deim_matrix: ($n$,$n_s$) array. DEIM approximation basis
    '''
    sampled_function_basis = function_basis[sample_indices]
    PU_pinv = np.linalg.pinv(sampled_function_basis)
    deim_matrix = function_basis @ PU_pinv
    return deim_matrix


def multi_state_deim_get_test_basis(test_basis, function_basis, sample_indices):
    '''
    For multistate systems. Constructs an independent DEIM basis for each state variable using uniform sample indices
    Args:
        test_basis: ($n_{var}$, $m$, $k$) array, where
            $n_{var}$ is the number of state variables,
            $m$ is the number of DOFs and
            $k$ is the number of basis functions. Test basis in projection scheme
        function_basis: ($n_{var}$, m, n) array, where
            $n_{var}$ is the number of state variables,
            $m$ is the number of DOFs and
            $n$ is the number of basis functions.
            Basis for function to be approximated.
        sample_indices: ($n_s$, ) array, where $n_s$ is the number of sample points. Sampling points.

    Returns:
        deim_test_basis: ($n_{var}$, $n_s$, $k$) array, where
            $n_{var}$ is the number of state variables,
            $n_s$ is the number of sample points, and
            $k$ the number of basis functions.
            DEIM test basis matrix.

    '''
    n_var = function_basis.shape[0]
    deim_test_basis = deim_get_test_basis(
        test_basis[0], function_basis[0], sample_indices
    )
    deim_test_basis = deim_test_basis[None]
    for i in range(1, n_var):
        deim_test_basis_i = deim_get_test_basis(
            test_basis[i], function_basis[i], sample_indices
        )
        deim_test_basis = np.append(deim_test_basis, deim_test_basis_i[None], axis=0)
    return deim_test_basis


def deim_get_test_basis(test_basis, function_basis, sample_indices):
    '''
    Given a test basis $\\mathbf{\\Phi}$, a function basis $\\mathbf{U}$, and
    sample indices defining $\\mathbf{P}$, we compute
    $$[ \\mathbf{\\Phi}^T \\mathbf{U} \\mathrm{pinv}( \\mathbf{P}^T \\mathbf{U}) ]^T$$
    which comprises the "test basis" for the DEIM approximation for
    $\\mathbf{\\Phi}^T \\mathbf{f}$

    Args:
        test_basis: ($m$, $k$) array, where
            $m$ is the number of DOFs and
            $k$ is the number of basis functions. Test basis in projection scheme
        function_basis: ($m$, $n$) array, where
            $m$ is the number of DOFs and
            $n$ is the number of basis functions. Basis for function to be approximated.
        sample_indices: ($n_s$, ) array, where $n_s$ is the number of sample points. Sampling points.

    Returns:
        deim_test_basis: ($n_s$, $k$) array, where
            $n_s$ is the number of sample points and
            $k$ the number of basis functions. DEIM test basis matrix.

    '''
    sampled_function_basis = function_basis[sample_indices]
    PU_pinv = np.linalg.pinv(sampled_function_basis)
    deim_test_basis = (test_basis.transpose() @ function_basis) @ PU_pinv
    return deim_test_basis.transpose()
