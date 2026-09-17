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


_BASIS_MODES = ("per_state", "global")


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


def _validate_per_state_basis_tensor(function_basis):
    basis = np.asarray(function_basis, dtype=float)
    if basis.ndim != 3:
        raise ValueError(
            "function_basis must be a rank-3 array with shape "
            "(n_vars, n_dofs, n_basis)"
        )
    if any(extent == 0 for extent in basis.shape):
        raise ValueError("function_basis must have no empty extents")
    return tuple(_validate_basis(basis[i]) for i in range(basis.shape[0]))


def _validate_state_bases(state_bases):
    if len(state_bases) == 0:
        raise ValueError("at least one state function basis is required")
    bases = tuple(_validate_basis(basis) for basis in state_bases)
    n_dofs = bases[0].shape[0]
    if any(basis.shape[0] != n_dofs for basis in bases):
        raise ValueError("all state function bases must have the same n_dofs")
    return bases


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


def _validate_basis_mode(basis_mode):
    if basis_mode not in _BASIS_MODES:
        raise ValueError(
            f"basis_mode must be one of {_BASIS_MODES}; got {basis_mode!r}"
        )
    return basis_mode


def _validate_rcond(rcond):
    if rcond is not None and (not np.isfinite(rcond) or rcond < 0):
        raise ValueError("rcond must be finite and nonnegative")
    return rcond


def _compute_pinv(matrix, rcond):
    if rcond is None:
        return np.linalg.pinv(matrix)
    return np.linalg.pinv(matrix, rcond=rcond)


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


def _validate_index_array(sample_indices):
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
    return indices


def _validate_spatial_sample_indices(sample_indices, n_dofs):
    indices = _validate_index_array(sample_indices)
    if np.any(indices < 0) or np.any(indices >= n_dofs):
        raise ValueError("sample_indices contains an out-of-bounds spatial index")
    return indices


def _validate_global_sample_indices(
        sample_indices, function_basis, sample_all_states):
    indices = _validate_index_array(sample_indices)
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


def _validate_per_state_sample_indices(sample_indices, state_bases):
    indices = _validate_spatial_sample_indices(
        sample_indices, state_bases[0].shape[0]
    )
    for state_index, basis in enumerate(state_bases):
        if indices.size < basis.shape[1]:
            raise ValueError(
                "sample_indices do not provide enough sampled rows for state "
                f"{state_index}"
            )
        sampled_basis = basis[indices, :]
        if np.linalg.matrix_rank(sampled_basis) != basis.shape[1]:
            raise ValueError(
                "the sampled function basis for state "
                f"{state_index} must have linearly independent columns"
            )
    return indices


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
    snapshots have shape ``(n_vars, n_dofs, n_snapshots)`` and tensor-form
    function bases have shape ``(n_vars, n_dofs, n_basis)``.

    ``basis_mode="per_state"`` is the default. A separate POD/function basis
    and DEIM sampling problem is constructed for each state variable. The
    state-specific sample sets are unioned into a common spatial sample mesh,
    and every state is sampled on that mesh. This avoids mixing quantities with
    different physical units in a single POD/SVD and allows the states to have
    different basis dimensions when constructed from snapshots.

    ``basis_mode="global"`` retains the coupled formulation in which the first
    two tensor axes are collapsed in explicit C order and a single global basis
    is used. Global mode can exploit cross-state correlations but requires the
    state variables to be consistently nondimensionalized or weighted.

    Notes:
        Function snapshots are snapshots of the quantity being approximated,
        such as a nonlinear term or residual. They are not necessarily the
        state snapshots used to construct a ROM trial basis.
    '''

    def __init__(self, function_basis, sample_indices=None, rcond=None,
                 sample_all_states=True, basis_mode="per_state"):
        basis_mode = _validate_basis_mode(basis_mode)
        if not isinstance(sample_all_states, (bool, np.bool_)):
            raise TypeError("sample_all_states must be a bool")
        rcond = _validate_rcond(rcond)

        if basis_mode == "per_state":
            state_bases = _validate_per_state_basis_tensor(function_basis)
            self._initialize_per_state(
                state_bases,
                sample_indices=sample_indices,
                rcond=rcond,
                sample_all_states=sample_all_states,
            )
        else:
            basis = _validate_basis_tensor(function_basis)
            self._initialize_global(
                basis,
                sample_indices=sample_indices,
                rcond=rcond,
                sample_all_states=sample_all_states,
            )

    @staticmethod
    def _select_sample_indices(function_basis):
        return _deim_get_indices_sharedmem(function_basis)

    def _initialize_global(self, function_basis, sample_indices, rcond,
                           sample_all_states):
        self.__basis_mode = "global"
        self.__sample_all_states = bool(sample_all_states)
        self.__function_basis = np.array(function_basis, copy=True)
        self.__state_function_bases = None
        self.__state_sample_indices = None
        self.__n_vars, self.__n_dofs, _ = self.__function_basis.shape
        self.__rcond = rcond

        if sample_indices is None:
            matrix_basis = _tensor_to_matrix(self.__function_basis)
            raw_indices = np.atleast_1d(
                self._select_sample_indices(matrix_basis)
            ).astype(int, copy=False)
            if self.__sample_all_states:
                sample_indices = _unique_preserve_order(
                    raw_indices % self.__n_dofs
                )
            else:
                sample_indices = raw_indices

        (
            self.__sample_indices,
            self.__flat_sample_indices,
        ) = _validate_global_sample_indices(
            sample_indices,
            self.__function_basis,
            self.__sample_all_states,
        )

        function_basis_matrix = _tensor_to_matrix(self.__function_basis)
        sampled_basis = function_basis_matrix[self.__flat_sample_indices, :]
        self.__reconstruction_matrix = (
            function_basis_matrix @ _compute_pinv(sampled_basis, self.__rcond)
        )

    def _initialize_per_state(self, state_bases, sample_indices, rcond,
                              sample_all_states):
        if not sample_all_states:
            raise ValueError(
                "sample_all_states=False is incompatible with "
                "basis_mode='per_state'; use basis_mode='global' for "
                "independent flattened state-DOF sampling"
            )

        state_bases = _validate_state_bases(state_bases)
        self.__basis_mode = "per_state"
        self.__sample_all_states = True
        self.__function_basis = None
        self.__state_function_bases = tuple(
            np.array(basis, copy=True) for basis in state_bases
        )
        self.__n_vars = len(self.__state_function_bases)
        self.__n_dofs = self.__state_function_bases[0].shape[0]
        self.__rcond = rcond
        self.__flat_sample_indices = None

        if sample_indices is None:
            state_sample_indices = tuple(
                np.atleast_1d(self._select_sample_indices(basis)).astype(
                    int, copy=False
                )
                for basis in self.__state_function_bases
            )
            self.__state_sample_indices = tuple(
                np.array(indices, copy=True)
                for indices in state_sample_indices
            )
            sample_indices = _unique_preserve_order(
                np.concatenate(state_sample_indices)
            )
        else:
            self.__state_sample_indices = None

        self.__sample_indices = _validate_per_state_sample_indices(
            sample_indices, self.__state_function_bases
        )

        reconstruction_matrices = []
        for basis in self.__state_function_bases:
            sampled_basis = basis[self.__sample_indices, :]
            reconstruction_matrices.append(
                basis @ _compute_pinv(sampled_basis, self.__rcond)
            )
        self.__reconstruction_matrix = scipy_linalg.block_diag(
            *reconstruction_matrices
        )

    @classmethod
    def _from_state_bases(cls, state_bases, sample_indices=None, rcond=None,
                          sample_all_states=True):
        obj = cls.__new__(cls)
        if not isinstance(sample_all_states, (bool, np.bool_)):
            raise TypeError("sample_all_states must be a bool")
        obj._initialize_per_state(
            state_bases,
            sample_indices=sample_indices,
            rcond=_validate_rcond(rcond),
            sample_all_states=sample_all_states,
        )
        return obj

    @classmethod
    def from_basis(cls, function_basis, sample_indices=None, rcond=None,
                   sample_all_states=True, basis_mode="per_state"):
        '''Construct a serial DEIM operator from a tensor-form function basis.

        Args:
            function_basis: ``(n_vars, n_dofs, n_basis)`` function basis. In
                ``per_state`` mode, each state slice is treated as an
                independent basis. The tensor form therefore requires a common
                basis dimension across states when bases are supplied directly.
            sample_indices: Optional rank-1 integer array. In ``per_state``
                mode and in shared-sample ``global`` mode these are spatial
                indices. With ``basis_mode="global"`` and
                ``sample_all_states=False`` these are flattened state-DOF
                indices in C order.
            rcond: Relative cutoff passed to :func:`numpy.linalg.pinv`.
            sample_all_states: If ``True`` (default), sample every state at each
                selected spatial point. ``False`` is supported only in global
                mode.
            basis_mode: ``"per_state"`` (default) constructs independent DEIM
                operators for each state and unions their sample points.
                ``"global"`` constructs one coupled operator.

        Returns:
            DEIM: Constructed interpolation operator.
        '''
        return cls(
            function_basis,
            sample_indices=sample_indices,
            rcond=rcond,
            sample_all_states=sample_all_states,
            basis_mode=basis_mode,
        )

    @classmethod
    def from_snapshots(cls, function_snapshots, truncater=None,
                       sample_indices=None, rcond=None,
                       sample_all_states=True, basis_mode="per_state"):
        '''Construct a serial DEIM operator from tensor-form function snapshots.

        Args:
            function_snapshots: ``(n_vars, n_dofs, n_snapshots)`` snapshots of
                the function to be approximated.
            truncater: Optional implementation of the
                ``LeftSingularVectorTruncater`` protocol. In ``per_state`` mode
                the truncater is applied independently to each state's SVD, so
                energy-based truncation may produce different basis dimensions
                for different states. If omitted, all available modes are
                retained.
            sample_indices: Optional rank-1 integer array. In ``per_state``
                mode and in shared-sample ``global`` mode these are spatial
                indices. With ``basis_mode="global"`` and
                ``sample_all_states=False`` these are flattened state-DOF
                indices in C order.
            rcond: Relative cutoff passed to :func:`numpy.linalg.pinv`.
            sample_all_states: If ``True`` (default), sample every state at each
                selected spatial point. ``False`` is supported only in global
                mode.
            basis_mode: ``"per_state"`` (default) performs a separate POD and
                DEIM selection for each state before taking the union of sample
                points. ``"global"`` performs one POD on the C-order flattened
                multistate snapshots.

        Returns:
            DEIM: Constructed interpolation operator.
        '''
        snapshots = _validate_snapshot_tensor(function_snapshots)
        basis_mode = _validate_basis_mode(basis_mode)
        if truncater is None:
            truncater = NoOpTruncater()

        if basis_mode == "per_state":
            state_bases = []
            for state_index in range(snapshots.shape[0]):
                function_space = VectorSpaceFromPOD(
                    snapshots[state_index:state_index + 1],
                    truncater=truncater,
                )
                state_bases.append(function_space.get_basis()[0])
            return cls._from_state_bases(
                state_bases,
                sample_indices=sample_indices,
                rcond=rcond,
                sample_all_states=sample_all_states,
            )

        function_space = VectorSpaceFromPOD(
            snapshots, truncater=truncater
        )
        function_basis = function_space.get_basis()
        return cls.from_basis(
            function_basis,
            sample_indices=sample_indices,
            rcond=rcond,
            sample_all_states=sample_all_states,
            basis_mode="global",
        )

    @property
    def basis_mode(self):
        '''Return ``"per_state"`` or ``"global"``.'''
        return self.__basis_mode

    @property
    def basis_sizes(self):
        '''Return the number of function-basis vectors associated with each state.'''
        if self.__basis_mode == "per_state":
            return tuple(
                basis.shape[1] for basis in self.__state_function_bases
            )
        return (self.__function_basis.shape[2],)

    @property
    def function_basis(self):
        '''Return a copy of the function basis or per-state function bases.

        In ``global`` mode this is a rank-3 romtools tensor. In ``per_state``
        mode this is a tuple of rank-2 arrays because the state-specific basis
        dimensions may differ.
        '''
        if self.__basis_mode == "per_state":
            return tuple(
                np.array(basis, copy=True)
                for basis in self.__state_function_bases
            )
        return self.__function_basis.copy()

    @property
    def sample_indices(self):
        '''Return the sample indices.

        These are spatial indices in ``per_state`` mode and whenever
        ``sample_all_states=True``. They are flattened C-order state-DOF indices
        only for global mode with ``sample_all_states=False``.
        '''
        return self.__sample_indices.copy()

    @property
    def state_sample_indices(self):
        '''Return the pre-union DEIM/QDEIM sample points for each state.

        This is a tuple in automatically sampled ``per_state`` mode and
        ``None`` for a global basis or when a sample mesh was supplied by the
        user.
        '''
        if self.__state_sample_indices is None:
            return None
        return tuple(
            np.array(indices, copy=True)
            for indices in self.__state_sample_indices
        )

    @property
    def sample_all_states(self):
        '''Whether all states are sampled at every selected spatial point.'''
        return self.__sample_all_states

    def reconstruction_matrix(self):
        '''Return the flattened DEIM reconstruction matrix.

        The row and column ordering follows the romtools C-order state-major
        convention. In ``per_state`` mode this is a block-diagonal matrix whose
        blocks are ``U_i @ pinv(U_i[sample_indices, :])``. The common union
        sample mesh therefore acts as an oversampled DEIM system for each state.
        '''
        return self.__reconstruction_matrix.copy()

    def reconstruct(self, sampled_values):
        '''Reconstruct full-order tensor values from sampled values.

        With shared-state sampling, ``sampled_values`` must have shape
        ``(n_vars, n_sample_points)`` or
        ``(n_vars, n_sample_points, n_snapshots)``. The return value has shape
        ``(n_vars, n_dofs)`` or ``(n_vars, n_dofs, n_snapshots)``.

        In global mode with ``sample_all_states=False``, sampled values have
        shape ``(n_samples,)`` or ``(n_samples, n_snapshots)`` because the
        samples are arbitrary flattened state-DOF entries. The reconstructed
        result is still returned in romtools tensor form.
        '''
        values = np.asarray(sampled_values)

        if self.__sample_all_states:
            n_samples = self.__sample_indices.size
            if values.ndim not in (2, 3):
                raise ValueError(
                    "sampled_values must be rank 2 or 3 when "
                    "sample_all_states=True"
                )
            if values.shape[:2] != (self.__n_vars, n_samples):
                raise ValueError(
                    "sampled_values must have leading shape "
                    "(n_vars, n_sample_points)"
                )
            if values.ndim == 2:
                flat_values = np.reshape(
                    values, (self.__n_vars * n_samples,), order="C"
                )
                reconstructed = self.__reconstruction_matrix @ flat_values
                return np.reshape(
                    reconstructed,
                    (self.__n_vars, self.__n_dofs),
                    order="C",
                )

            flat_values = np.reshape(
                values,
                (self.__n_vars * n_samples, values.shape[2]),
                order="C",
            )
            reconstructed = self.__reconstruction_matrix @ flat_values
            return np.reshape(
                reconstructed,
                (self.__n_vars, self.__n_dofs, values.shape[2]),
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
                reconstructed,
                (self.__n_vars, self.__n_dofs),
                order="C",
            )
        return np.reshape(
            reconstructed,
            (self.__n_vars, self.__n_dofs, values.shape[1]),
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
        if basis.shape[:2] != (self.__n_vars, self.__n_dofs):
            raise ValueError(
                "test_basis and function basis must have matching "
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
            (self.__n_vars, self.__sample_indices.size, basis.shape[2]),
            order="C",
        )


class QDEIM(DEIM):
    '''Tensor-native DEIM operator using pivoted-QR point selection.

    Basis construction, tensor handling, and shared-state sampling are inherited
    from :class:`DEIM`; only the DEIM selection performed on each applicable
    basis differs.
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
    """
    Implementation of the discrete empirical method as described in Algorithm 1 of
    S. Chaturantabut and D. C. Sorensen, "Discrete Empirical Interpolation for
    nonlinear model reduction," doi: 10.1109/CDC.2009.5400045.

    Args:
        U:
            :math:`\\mathbf{U} \\in \\mathbb{R}^{m \\times n}`, where
            :math:`m` is the number of DOFs and
            :math:`n` is the number of samples.
            Function basis in matrix format.
        comm:
            Optional communicator object. If none, algorithm assumes shared-memory data.

    Returns:
        If ``comm == None``:

            :math:`\\mathrm{indices} \\in \\mathbb{I}^{n}`: sample mesh indices.

        Otherwise:

            Sample mesh local indices and the corresponding owing ranks.
    """

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
    """
    Version of DEIM for multi-state systems.

    We perform DEIM on each state variable, and
    then return the union of all indices.
    Repeated indices are removed.

    Args:
        U:
            :math:`\\mathbf{U} \\in \\mathbb{R}^{l \\times m \\times n}`, where
            :math:`l` is the number of variables,
            :math:`m` is the number of DOFs and
            :math:`n` the number of samples.
            Multi-dimensional function basis in tensor format.

    Returns:
        If ``comm == None``:

            :math:`\\mathrm{indices} \\in \\mathbb{I}^{n}`: sample mesh indices.

        Otherwise:

            Sample mesh local indices and the corresponding owing ranks.
    """
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
