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
import scipy.linalg as scipy_linalg
import romtools.linalg.linalg as la
from romtools.vector_space import VectorSpaceFromPOD
from romtools.vector_space.utils.truncater import NoOpTruncater


def _validate_basis(function_basis):
    '''Validate a matrix-form basis used by the legacy DEIM helpers.'''
    basis = np.asarray(function_basis, dtype=float)
    if basis.ndim != 2:
        raise ValueError("function_basis must be a rank-2 array")
    if basis.shape[0] == 0 or basis.shape[1] == 0:
        raise ValueError("function_basis must have at least one row and one column")
    if basis.shape[0] < basis.shape[1]:
        raise ValueError(
            "function_basis must have at least as many rows as columns"
        )
    if not np.all(np.isfinite(basis)):
        raise ValueError("function_basis must contain only finite values")
    if np.linalg.matrix_rank(basis) != basis.shape[1]:
        raise ValueError("function_basis must have linearly independent columns")
    return np.array(basis, copy=True)


def _tensor_to_matrix(tensor):
    '''Collapse the romtools ``(n_vars, n_dofs, n_cols)`` tensor in C order.'''
    array = np.asarray(tensor)
    return np.reshape(
        array,
        (array.shape[0] * array.shape[1], array.shape[2]),
        order="C",
    )


def _validate_basis_tensor(function_basis):
    basis = np.asarray(function_basis, dtype=float)
    if basis.ndim != 3:
        raise ValueError(
            "function_basis must be a rank-3 array with shape "
            "(n_vars, n_dofs, n_basis)"
        )
    if any(extent == 0 for extent in basis.shape):
        raise ValueError("function_basis must have no empty extents")
    _validate_basis(_tensor_to_matrix(basis))
    return np.array(basis, copy=True)


def _validate_snapshot_tensor(function_snapshots):
    snapshots = np.asarray(function_snapshots, dtype=float)
    if snapshots.ndim != 3:
        raise ValueError(
            "function_snapshots must be a rank-3 array with shape "
            "(n_vars, n_dofs, n_snapshots)"
        )
    if any(extent == 0 for extent in snapshots.shape):
        raise ValueError("function_snapshots must have no empty extents")
    if not np.all(np.isfinite(snapshots)):
        raise ValueError("function_snapshots must contain only finite values")
    return np.array(snapshots, copy=True)


def _unique_preserve_order(indices):
    indices = np.asarray(indices, dtype=int)
    _, first_occurrences = np.unique(indices, return_index=True)
    return indices[np.sort(first_occurrences)]


def _expand_spatial_sample_indices(sample_indices, n_vars, n_dofs):
    '''Map shared spatial samples to flattened state-major C-order indices.'''
    indices = np.asarray(sample_indices, dtype=int)
    expanded = (
        np.arange(n_vars, dtype=int)[:, None] * n_dofs
        + indices[None, :]
    )
    return np.reshape(expanded, (-1,), order="C")


def _validate_class_sample_indices(
        sample_indices, function_basis, sample_all_states):
    indices = np.asarray(sample_indices)
    if indices.ndim != 1:
        raise ValueError("sample_indices must be a rank-1 array")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("sample_indices must contain integers")
    indices = np.array(indices, dtype=int, copy=True)
    if indices.size == 0:
        raise ValueError("sample_indices must contain at least one index")
    if np.unique(indices).size != indices.size:
        raise ValueError("sample_indices must not contain duplicates")

    n_vars, n_dofs, n_basis = function_basis.shape
    function_basis_matrix = _tensor_to_matrix(function_basis)
    if sample_all_states:
        if np.any(indices < 0) or np.any(indices >= n_dofs):
            raise ValueError("sample_indices contains an out-of-bounds spatial index")
        flat_indices = _expand_spatial_sample_indices(
            indices, n_vars, n_dofs
        )
    else:
        if np.any(indices < 0) or np.any(indices >= n_vars * n_dofs):
            raise ValueError("sample_indices contains an out-of-bounds flattened index")
        flat_indices = indices.copy()

    if flat_indices.size < n_basis:
        raise ValueError(
            "sample_indices do not provide enough sampled rows for the basis"
        )
    sampled_basis = function_basis_matrix[flat_indices, :]
    if np.linalg.matrix_rank(sampled_basis) != n_basis:
        raise ValueError(
            "the sampled function basis must have linearly independent columns"
        )
    return indices, flat_indices


def qdeim_get_indices(function_basis):
    '''Select interpolation indices using QDEIM for a matrix-form basis.

    QDEIM applies a column-pivoted QR factorization to the transpose of the
    function basis. The first ``n_basis`` pivot indices define the
    interpolation points.

    Args:
        function_basis: ``(n_dofs, n_basis)`` function basis.

    Returns:
        numpy.ndarray: Rank-1 integer array of interpolation indices.
    '''
    basis = _validate_basis(function_basis)
    _, _, pivots = scipy_linalg.qr(
        basis.transpose(), mode="economic", pivoting=True
    )
    return np.asarray(pivots[:basis.shape[1]], dtype=int)


class DEIM:
    '''Serial discrete empirical interpolation operator.

    The public class API follows the romtools tensor convention. Function
    bases have shape ``(n_vars, n_dofs, n_basis)`` and function snapshots have
    shape ``(n_vars, n_dofs, n_snapshots)``. Internally, tensors are collapsed
    to matrices using explicit C ordering, matching the rest of romtools.

    By default, DEIM selects scalar state-space rows and promotes the selected
    locations to shared spatial sample points: if any state is selected at a
    spatial point, every state is sampled there. Set ``sample_all_states=False``
    to retain independent flattened state-DOF sampling.

    Notes:
        Function snapshots are snapshots of the quantity being approximated,
        such as a nonlinear term or residual. They are not necessarily the
        state snapshots used to construct a ROM trial basis.
    '''

    def __init__(self, function_basis, sample_indices, rcond=None,
                 sample_all_states=True):
        self.__function_basis = _validate_basis_tensor(function_basis)
        if not isinstance(sample_all_states, (bool, np.bool_)):
            raise TypeError("sample_all_states must be a bool")
        self.__sample_all_states = bool(sample_all_states)
        (
            self.__sample_indices,
            self.__flat_sample_indices,
        ) = _validate_class_sample_indices(
            sample_indices,
            self.__function_basis,
            self.__sample_all_states,
        )
        if rcond is not None and (not np.isfinite(rcond) or rcond < 0):
            raise ValueError("rcond must be finite and nonnegative")
        self.__rcond = rcond

        function_basis_matrix = _tensor_to_matrix(self.__function_basis)
        sampled_basis = function_basis_matrix[self.__flat_sample_indices, :]
        if self.__rcond is None:
            sampled_basis_pinv = np.linalg.pinv(sampled_basis)
        else:
            sampled_basis_pinv = np.linalg.pinv(
                sampled_basis, rcond=self.__rcond
            )
        self.__reconstruction_matrix = (
            function_basis_matrix @ sampled_basis_pinv
        )

    @staticmethod
    def _select_sample_indices(function_basis):
        return _deim_get_indices_sharedmem(function_basis)

    @classmethod
    def from_basis(cls, function_basis, sample_indices=None, rcond=None,
                   sample_all_states=True):
        '''Construct a serial DEIM operator from a tensor-form function basis.

        Args:
            function_basis: ``(n_vars, n_dofs, n_basis)`` function basis.
            sample_indices: Optional rank-1 integer array. With
                ``sample_all_states=True`` these are spatial indices. With
                ``sample_all_states=False`` these are flattened state-DOF
                indices in C order. If omitted, DEIM selects the samples.
            rcond: Relative cutoff passed to :func:`numpy.linalg.pinv`.
            sample_all_states: If ``True`` (default), selecting any state at a
                spatial point samples every state at that point.

        Returns:
            DEIM: Constructed interpolation operator.
        '''
        basis = _validate_basis_tensor(function_basis)
        if not isinstance(sample_all_states, (bool, np.bool_)):
            raise TypeError("sample_all_states must be a bool")

        if sample_indices is None:
            n_dofs = basis.shape[1]
            matrix_basis = _tensor_to_matrix(basis)
            raw_indices = np.atleast_1d(
                cls._select_sample_indices(matrix_basis)
            ).astype(int, copy=False)
            if sample_all_states:
                sample_indices = _unique_preserve_order(raw_indices % n_dofs)
            else:
                sample_indices = raw_indices

        return cls(
            basis,
            sample_indices,
            rcond=rcond,
            sample_all_states=sample_all_states,
        )

    @classmethod
    def from_snapshots(cls, function_snapshots, truncater=None,
                       sample_indices=None, rcond=None,
                       sample_all_states=True):
        '''Construct a serial DEIM operator from tensor-form function snapshots.

        Args:
            function_snapshots: ``(n_vars, n_dofs, n_snapshots)`` snapshots of
                the function to be approximated.
            truncater: Optional implementation of the
                ``LeftSingularVectorTruncater`` protocol. If omitted, all
                available POD modes are retained.
            sample_indices: Optional rank-1 integer array. With
                ``sample_all_states=True`` these are spatial indices. With
                ``sample_all_states=False`` these are flattened state-DOF
                indices in C order.
            rcond: Relative cutoff passed to :func:`numpy.linalg.pinv`.
            sample_all_states: If ``True`` (default), selecting any state at a
                spatial point samples every state at that point.

        Returns:
            DEIM: Constructed interpolation operator.
        '''
        snapshots = _validate_snapshot_tensor(function_snapshots)
        if truncater is None:
            truncater = NoOpTruncater()
        function_space = VectorSpaceFromPOD(
            snapshots, truncater=truncater
        )
        function_basis = function_space.get_basis()
        return cls.from_basis(
            function_basis,
            sample_indices=sample_indices,
            rcond=rcond,
            sample_all_states=sample_all_states,
        )

    @property
    def function_basis(self):
        '''Return a copy of the tensor-form function basis.'''
        return self.__function_basis.copy()

    @property
    def sample_indices(self):
        '''Return the sample indices.

        These are spatial indices when ``sample_all_states=True`` and flattened
        C-order state-DOF indices otherwise.
        '''
        return self.__sample_indices.copy()

    @property
    def sample_all_states(self):
        '''Whether all states are sampled at every selected spatial point.'''
        return self.__sample_all_states

    def reconstruction_matrix(self):
        '''Return the flattened DEIM reconstruction matrix.

        The row ordering follows the romtools C-order state-major convention.
        When ``sample_all_states=True``, columns are ordered by state first and
        then by the spatial sample ordering returned by :attr:`sample_indices`.
        '''
        return self.__reconstruction_matrix.copy()

    def reconstruct(self, sampled_values):
        '''Reconstruct full-order tensor values from sampled values.

        With ``sample_all_states=True``, ``sampled_values`` must have shape
        ``(n_vars, n_sample_points)`` or
        ``(n_vars, n_sample_points, n_snapshots)``. The return value has shape
        ``(n_vars, n_dofs)`` or ``(n_vars, n_dofs, n_snapshots)``.

        With ``sample_all_states=False``, sampled values have shape
        ``(n_samples,)`` or ``(n_samples, n_snapshots)`` because the samples
        are arbitrary flattened state-DOF entries. The reconstructed result is
        still returned in romtools tensor form.
        '''
        values = np.asarray(sampled_values)
        n_vars, n_dofs, _ = self.__function_basis.shape

        if self.__sample_all_states:
            n_samples = self.__sample_indices.size
            if values.ndim not in (2, 3):
                raise ValueError(
                    "sampled_values must be rank 2 or 3 when "
                    "sample_all_states=True"
                )
            if values.shape[:2] != (n_vars, n_samples):
                raise ValueError(
                    "sampled_values must have leading shape "
                    "(n_vars, n_sample_points)"
                )
            if values.ndim == 2:
                flat_values = np.reshape(
                    values, (n_vars * n_samples,), order="C"
                )
                reconstructed = self.__reconstruction_matrix @ flat_values
                return np.reshape(
                    reconstructed, (n_vars, n_dofs), order="C"
                )

            flat_values = np.reshape(
                values,
                (n_vars * n_samples, values.shape[2]),
                order="C",
            )
            reconstructed = self.__reconstruction_matrix @ flat_values
            return np.reshape(
                reconstructed,
                (n_vars, n_dofs, values.shape[2]),
                order="C",
            )

        if values.ndim not in (1, 2):
            raise ValueError(
                "sampled_values must be rank 1 or 2 when "
                "sample_all_states=False"
            )
        if values.shape[0] != self.__sample_indices.size:
            raise ValueError(
                "the first extent of sampled_values must equal the number "
                "of flattened sample indices"
            )
        reconstructed = self.__reconstruction_matrix @ values
        if values.ndim == 1:
            return np.reshape(
                reconstructed, (n_vars, n_dofs), order="C"
            )
        return np.reshape(
            reconstructed,
            (n_vars, n_dofs, values.shape[1]),
            order="C",
        )

    def project_test_basis(self, test_basis):
        '''Construct the sampled test basis used for hyper-reduction.

        ``test_basis`` must have romtools tensor shape
        ``(n_vars, n_dofs, n_test_basis)``. With shared-state sampling the
        result has shape ``(n_vars, n_sample_points, n_test_basis)``. With
        independent flattened sampling it has shape
        ``(n_samples, n_test_basis)``.
        '''
        basis = np.asarray(test_basis)
        if basis.ndim != 3:
            raise ValueError(
                "test_basis must be a rank-3 array with shape "
                "(n_vars, n_dofs, n_test_basis)"
            )
        n_vars, n_dofs, _ = self.__function_basis.shape
        if basis.shape[:2] != (n_vars, n_dofs):
            raise ValueError(
                "test_basis and function_basis must have matching "
                "(n_vars, n_dofs) extents"
            )
        basis_matrix = _tensor_to_matrix(basis)
        projected = (
            basis_matrix.transpose() @ self.__reconstruction_matrix
        ).transpose()
        if not self.__sample_all_states:
            return projected
        return np.reshape(
            projected,
            (n_vars, self.__sample_indices.size, basis.shape[2]),
            order="C",
        )


class QDEIM(DEIM):
    '''Tensor-native DEIM operator using pivoted-QR point selection.

    Reconstruction, tensor handling, and shared-state sampling are inherited
    from :class:`DEIM`; only the initial scalar interpolation-point selection
    differs.
    '''

    @staticmethod
    def _select_sample_indices(function_basis):
        return qdeim_get_indices(function_basis)


def _deim_get_indices_sharedmem(U):
    '''
    Implementation of the discrete empirical method as described in Algorithm 1 of
    S. Chaturantabut and D. C. Sorensen, "Discrete Empirical Interpolation for
    nonlinear model reduction," doi: 10.1109/CDC.2009.5400045.

    Args:
        :math:`\\mathbf{U} \\in \\mathbb{R}^{m \\times n}`, where
            :math:`m` is the number of DOFs and
            :math:`n` is the number of samples.
            Function basis in matrix format

    Returns:
        :math:`\\mathrm{indices} \\in \\mathbb{I}^{n}`: sample mesh indices
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


class _dist_deim_data:
    def __init__(self, i, r):
        self.local_indices = np.array([int(i)])
        self.owning_ranks = np.array([int(r)])

    def append(self, i, r):
        '''Adds the local index and rank to self.local_indices and self.owning_ranks.'''
        self.local_indices = np.append(self.local_indices , int(i))
        self.owning_ranks = np.append(self.owning_ranks, int(r))


def _deim_get_indices_distributed(U, comm):
    m = np.shape(U)[1]
    local_index, foundRank = la.argmax(np.abs(U[:, 0]), comm)
    result = _dist_deim_data(local_index, foundRank)
    if m == 1:
        return result.local_indices, result.owning_ranks

    myRank = comm.Get_rank()
    LHS, RHS, C = np.array([]), np.array([]), np.array([])
    for ell in range(1, m):
        indices = result.local_indices[result.owning_ranks==myRank]
        LHS = np.array([]) if indices.size == 0 else U[indices, 0:ell]
        RHS = np.array([]) if indices.size == 0 else U[indices, ell]

        A, b = la.move_distributed_linear_system_to_rank_zero(LHS, RHS, comm)
        if myRank == 0:
            C = np.linalg.solve(A, b)
        C = comm.bcast(C, root=0)

        residual = U[:, ell] - U[:, 0:ell] @ C
        local_index, foundRank = la.argmax(np.abs(residual), comm)
        result.append(local_index, foundRank)

    return result.local_indices, result.owning_ranks


def deim_get_indices(U, comm=None):
    '''
    Implementation of the discrete empirical method as described in Algorithm 1 of
    S. Chaturantabut and D. C. Sorensen, "Discrete Empirical Interpolation for
    nonlinear model reduction," doi: 10.1109/CDC.2009.5400045.

    Args:
        :math:`\\mathbf{U} \\in \\mathbb{R}^{m \\times n}`, where
            :math:`m` is the number of DOFs and
            :math:`n` is the number of samples.
            Function basis in matrix format.
        comm: Optional communicator object. If none, algorithm assumes shared-memory data.

    Returns:
         if comm==None:
            :math:`\\mathrm{indices} \\in \\mathbb{I}^{n}`: sample mesh indices

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
        all_local_indices = np.append(all_local_indices, indices)
        all_ranks = np.append(all_ranks, ranks)

    M = np.vstack((all_local_indices, all_ranks))
    M = np.unique(M, axis=1)
    inds = M[1,:].argsort()
    M = M[:, inds]

    rank_ids = np.unique(M[1,:])
    for rank in rank_ids:
        locs = np.where(M[1,:] == rank)
        M[0, locs] = np.sort(M[0, locs])

    return M[0,:], M[1,:]


def multi_state_deim_get_indices(U, comm=None):
    '''
    Version of DEIM for multi-state systems.

    We perform DEIM on each state variable, and
    then return the union of all indices.
    Repeated indices are removed.

    Args:
         :math:`\\mathbf{U} \\in \\mathbb{R}^{l \\times m \\times n}`, where
            :math:`l` is the number of variables,
            :math:`m` is the number of DOFs and
            :math:`n` the number of samples.
            Multi-dimensional function basis in tensor format.

    Returns:
         if comm==None:
            :math:`\\mathrm{indices} \\in \\mathbb{I}^{n}`: sample mesh indices

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
    Given a function basis :math:`\\mathbf{U}` and sample indices defining :math:`\\mathbf{P}`, we compute

    .. math::

       \\mathbf{U} \\mathrm{pinv}( \\mathbf{P}^T \\mathbf{U})

    which comprises the matrix needed for the DEIM approximation to :math:`\\mathbf{f}`

    Args:
        function_basis: :math:`(m, n)` array, where :math:`m` is the number of DOFs and :math:`n` the number of basis
            functions.
            Basis for function to be approximated.
        sample_indices: :math:`(n_s, )` array, where :math:`n_s` is the number of sample points. Sampling points.

    Returns:
        deim_matrix: :math:`(n, n_s)` array. DEIM approximation basis
    '''
    sampled_function_basis = function_basis[sample_indices]
    PU_pinv = np.linalg.pinv(sampled_function_basis)
    deim_matrix = function_basis @ PU_pinv
    return deim_matrix


def multi_state_deim_get_test_basis(test_basis, function_basis, sample_indices):
    '''
    For multistate systems. Constructs an independent DEIM basis for each state variable using uniform sample indices
    Args:
        test_basis: :math:`(n_{var}, m, k)` array, where
            :math:`n_{var}` is the number of state variables,
            :math:`m` is the number of DOFs and
            :math:`k` is the number of basis functions. Test basis in projection scheme
        function_basis: :math:`(n_{var}, m, n)` array, where
            :math:`n_{var}` is the number of state variables,
            :math:`m` is the number of DOFs and
            :math:`n` is the number of basis functions.
            Basis for function to be approximated.
        sample_indices: :math:`(n_s, )` array, where :math:`n_s` is the number of sample points. Sampling points.

    Returns:
        deim_test_basis: :math:`(n_{var}, n_s, k)` array, where
            :math:`n_{var}` is the number of state variables,
            :math:`n_s` is the number of sample points, and
            :math:`k` the number of basis functions.
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


def deim_get_test_basis(test_basis, function_basis, sample_indices, comm=None):
    '''
    Given a test basis :math:`\\mathbf{\\Phi}`, a function basis :math:`\\mathbf{U}`, and
    sample indices defining :math:`\\mathbf{P}`, we compute

    .. math::

       [ \\mathbf{\\Phi}^T \\mathbf{U} \\mathrm{pinv}( \\mathbf{P}^T \\mathbf{U}) ]^T

    which comprises the "test basis" for the DEIM approximation for
    :math:`\\mathbf{\\Phi}^T \\mathbf{f}`

    Args:
        test_basis: :math:`(m, k)` array, where
            :math:`m` is the number of DOFs and
            :math:`k` is the number of basis functions. Test basis in projection scheme
        function_basis: :math:`(m, n)` array, where
            :math:`m` is the number of DOFs and
            :math:`n` is the number of basis functions. Basis for function to be approximated.
        sample_indices: :math:`(n_s, )` array, where :math:`n_s` is the number of sample points. Sampling points.
        comm: Optional communicator object. If none, algorithm assumes shared-memory data.

    Returns:
        deim_test_basis: :math:`(n_s, k)` array, where
            :math:`n_s` is the number of sample points and
            :math:`k` the number of basis functions. DEIM test basis matrix.

    '''
    sampled_function_basis = np.empty((0, function_basis.shape[1]))
    if len(sample_indices) > 0:
        sampled_function_basis = function_basis[sample_indices]

    PU_pinv_transpose = la.pinv(sampled_function_basis, comm=comm)

    phi_U = np.empty((test_basis.shape[1], function_basis.shape[1]))
    la.product("T", "N", 1, test_basis, function_basis, 0, phi_U, comm=comm)

    if comm is not None:
        local_pinvs = comm.allgather(PU_pinv_transpose)
        PU_pinv_transpose = np.vstack(local_pinvs)

    deim_test_basis = np.empty((phi_U.shape[0], PU_pinv_transpose.shape[0]))
    la.product("N", "T", 1, phi_U, PU_pinv_transpose, 0, deim_test_basis)

    return deim_test_basis.transpose()
