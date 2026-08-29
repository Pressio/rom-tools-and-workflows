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
Notes
-----
The scaler class is used to performed scaled POD.
Scaling is applied to tensors of shape :math:`\\mathbb{R}^{ N_{\\mathrm{vars}} \\times N_{\\mathrm{x}} \\times N_s}`.
These tensors are then reshaped into matrices when performing SVD.

Theory
------

*What is scaled POD, and why would I do it?*

Standard POD computes a basis that minimizes the projection error in a standard Euclidean :math:`\\ell^2` inner product,
i.e., for a snapshot matrix :math:`\\mathbf{S} \\in \\mathbb{R}^{  N_{\\mathrm{vars}} N_{\\mathrm{x}} \\times N_s}`,
POD computes the basis by solving the minimization problem (assuming no affine offset)

.. math::

   \\boldsymbol \\Phi = \\underset{ \\boldsymbol \\Phi_{\\ast} \\in \\mathbb{R}^{ N_{\\mathrm{vars}} N_{\\mathrm{x}}
   \\times K} | \\boldsymbol \\Phi_{\\ast}^T \\boldsymbol \\Phi_{\\ast} = \\mathbf{I}}{ \\mathrm{arg \\; min} }
   \\| \\Phi_{\\ast} \\Phi_{\\ast}^T \\mathbf{S} - \\mathbf{S} \\|_2.

In this minimization problem, errors are measured in a standard :math:`\\ell^2` norm.
For most practical applications, where our snapshot matrix involves variables of different scales,
this norm does not make sense (both intuitively, and on dimensional grounds).
As a practical example, consider fluid dynamics where the total energy is orders of magnitude larger than the density.

One of the most common approaches for mitigating this issue is to perform scaled POD.
In scaled POD, we solve a minimization problem on a scaled snapshot matrix.
Defining :math:`\\mathbf{S}_{\\ast} = \\mathbf{W}^{-1} \\mathbf{S}`, where :math:`\\mathbf{W}` is a weighting matrix
(e.g., a diagonal matrix containing the max absolute value of each state variable),
we compute the basis as the solution to the minimization problem

.. math::

   \\boldsymbol \\Phi = \\mathbf{W} \\underset{ \\boldsymbol \\Phi_{\\ast} \\in \\mathbb{R}^{N_{\\mathrm{vars}} N_{\\mathrm{x}}
   \\times K} |\\boldsymbol \\Phi_{\\ast}^T \\boldsymbol \\Phi_{\\ast} = \\mathbf{I}}{ \\mathrm{arg \\; min} }
   \\| \\Phi_{\\ast} \\Phi_{\\ast}^T \\mathbf{S}_{\\ast} - \\mathbf{S}_{\\ast} \\|_2.

The Scaler encapsulates this information.

API
---
'''

from typing import Protocol
import numpy as np
import romtools.linalg.linalg as la
from romtools.vector_space.utils.snapshot_loader import SnapshotLoader


class Scaler(Protocol):
    '''
    Interface for the Scaler class.
    '''

    def pre_scale(self, data_tensor: np.ndarray) -> None:
        '''
        Scales the snapshot matrix in place before performing SVD
        '''
        ...

    def post_scale(self, data_tensor: np.ndarray) -> None:
        '''
        Scales the left singular vectors in place after performing SVD
        '''
        ...


class StreamingScaler(Scaler, Protocol):
    '''
    Scaler interface required by streaming POD vector spaces.
    '''

    def initialize_scalings_from_loader(
            self,
            snapshot_loader: SnapshotLoader,
            block_size: int,
            n_snapshots: int,
            comm=None) -> None:
        '''
        Initialize scaling data from snapshot blocks.

        This method is only required for streaming POD. Fixed scalers may
        implement it as a no-op.
        '''
        ...


class NoOpScaler:
    '''
    No op implementation

    This class conforms to the :class:`Scaler` protocol.
    '''

    def __init__(self) -> None:
        pass

    def initialize_scalings_from_loader(
            self,
            snapshot_loader: SnapshotLoader,
            block_size: int,
            n_snapshots: int,
            comm=None) -> None:
        # This method is only required for streaming POD.
        _ = snapshot_loader, block_size, n_snapshots, comm

    def pre_scale(self, data_tensor: np.ndarray):
        '''Does not alter the input data matrix.'''
        pass

    def post_scale(self, data_tensor):
        '''Does not alter the input data matrix.'''
        pass


class VectorScaler:
    '''
    Concrete implementation designed to scale snapshot matrices by a vector.
    For a snapshot tensor :math:`\\mathbf{S} \\in \\mathbb{R}^{N_{\\mathrm{u}} \\times N \\times K}`, the VectorScaler
    accepts in a scaling vector :math:`\\mathbf{v} \\in \\mathbb{R}^{N}`, and scales by

    .. math::

       \\mathbf{S}^* = \\mathrm{diag}(\\mathbf{v})^{-1} \\mathbf{S}

    before performing POD (i.e., POD is performed on :math:`\\mathbf{S}^*`). After POD is performed, the bases
    are post-scaled by

    .. math::

       \\boldsymbol \\Phi = \\mathrm{diag}(\\mathbf{v}) \\mathbf{U}

    **Note that scaling can cause bases to not be orthonormal; we do not
    recommend using scalers with the NoOpOrthogonalizer.**

    This class conforms to the :class:`Scaler` protocol.
    '''

    def __init__(self, scaling_vector) -> None:
        '''
        Constructor for the VectorScaler.

        Args:
            scaling_vector: Array containing the scaling vector for each row
                in the snapshot matrix.

        This constructor initializes the VectorScaler with the specified
        scaling vector.
        '''
        self.__scaling_vector_matrix = scaling_vector
        self.__scaling_vector_matrix_inv = 1.0 / scaling_vector

    def initialize_scalings_from_loader(
            self,
            snapshot_loader: SnapshotLoader,
            block_size: int,
            n_snapshots: int,
            comm=None) -> None:
        # This method is only required for streaming POD.
        _ = snapshot_loader, block_size, n_snapshots, comm

    def pre_scale(self, data_tensor: np.ndarray) -> None:
        '''
        Scales the input data matrix in place using the inverse of the scaling vector.

        Args:
            data_tensor (np.ndarray): The input data matrix to be scaled.
        '''
        data_tensor *= self.__scaling_vector_matrix_inv[None, :, None]

    def post_scale(self, data_tensor: np.ndarray) -> None:
        '''
        Scales the input data matrix in place using the scaling vector.

        Args:
            data_tensor (np.ndarray): The input data matrix to be scaled.
        '''
        data_tensor *= self.__scaling_vector_matrix[None, :, None]


class ScalarScaler:
    '''
    Applies a scalar scale factor

    This class conforms to the :class:`Scaler` protocol.
    '''

    def __init__(self, factor: float = 1.0) -> None:
        self._factor = factor

    def initialize_scalings_from_loader(
            self,
            snapshot_loader: SnapshotLoader,
            block_size: int,
            n_snapshots: int,
            comm=None) -> None:
        # This method is only required for streaming POD.
        _ = snapshot_loader, block_size, n_snapshots, comm

    def pre_scale(self, data_tensor: np.ndarray) -> np.ndarray:
        '''
        Scales the input data matrix in place using the reciprocal of the input factor.

        Args:
            data_tensor (np.ndarray): The input data matrix to be scaled.
        '''
        data_tensor /= self._factor

    def post_scale(self, data_tensor: np.ndarray) -> np.ndarray:
        '''
        Scales the input data matrix in place using the input factor.

        Args:
            data_tensor (np.ndarray): The input data matrix to be scaled.
        '''
        data_tensor *= self._factor


class VariableScaler:
    '''
    Concrete implementation designed for snapshot matrices involving multiple
    state variables.

    This class is designed to scale a data matrix comprising multiple states
    (e.g., for the Navier--Stokes, rho, rho u, rhoE)

    The available scaling options are:

    - ``"max_abs"``: for state variable :math:`u_i`, compute
      :math:`s_i = \\max\\left(\\lvert S_i \\rvert\\right)`.
    - ``"mean_abs"``: for state variable :math:`u_i`, compute
      :math:`s_i = \\operatorname{mean}\\left(\\lvert S_i \\rvert\\right)`.
    - ``"variance"``: for state variable :math:`u_i`, compute
      :math:`s_i = \\operatorname{std}\\left(S_i\\right)`.

    Here, :math:`S_i` denotes the snapshot matrix for state variable
    :math:`u_i`.

    This class conforms to the :class:`Scaler` protocol.
    '''

    def __init__(self, scaling_type) -> None:
        '''
        Constructor for the VariableScaler.

        Args:
            scaling_type (str): The scaling method to use ('max_abs',
            'mean_abs', or 'variance').

        This constructor initializes the VariableScaler with the specified
        scaling type, variable ordering, and number of variables.
        '''
        self.__scaling_type = scaling_type
        self.have_scales_been_initialized = False
        self.var_scales_ = None

    def initialize_scalings(self, data_tensor: np.ndarray) -> None:
        '''
        Initializes the scaling factors for each state variable based on the
        specified method.

        Args:
            data_tensor (np.ndarray): The input data matrix.
        '''
        n_var = data_tensor.shape[0]
        self.var_scales_ = np.ones(n_var)
        for i in range(n_var):
            ith_var = data_tensor[i]
            if self.__scaling_type == "max_abs":
                var_scale = la.max(abs(ith_var))
            elif self.__scaling_type == "mean_abs":
                var_scale = la.mean(abs(ith_var))
            elif self.__scaling_type == "variance":
                var_scale = la.std(ith_var)

            # in case of a zero field (e.g., 2D)
            if var_scale < 1e-10:
                var_scale = 1.0
            self.var_scales_[i] = var_scale
        self.have_scales_been_initialized = True

    def initialize_scalings_from_loader(
            self,
            snapshot_loader: SnapshotLoader,
            block_size: int,
            n_snapshots: int,
            comm=None) -> None:
        '''
        Initialize variable scales from all snapshot blocks.

        This method is only required for streaming POD.
        '''
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if n_snapshots <= 0:
            raise ValueError("n_snapshots must be positive")

        state_shape = None
        running_count = 0
        running_max = None
        running_abs_sum = None
        running_mean = None
        running_m2 = None

        for start in range(0, n_snapshots, block_size):
            end = min(start + block_size, n_snapshots)
            block = np.asarray(snapshot_loader(start, end))
            if block.ndim != 3:
                raise ValueError("snapshot loader must return three-dimensional blocks")
            if block.shape[-1] != end - start:
                raise ValueError("snapshot loader returned an incorrect number of snapshots")
            if state_shape is None:
                state_shape = block.shape[:-1]
                n_var = state_shape[0]
                running_max = np.zeros(n_var)
                running_abs_sum = np.zeros(n_var)
                running_mean = np.zeros(n_var)
                running_m2 = np.zeros(n_var)
                if comm is not None and comm.Get_size() > 1:
                    from mpi4py import MPI
                    variable_counts = comm.allgather(n_var)
                    if any(count != variable_counts[0]
                           for count in variable_counts):
                        raise ValueError(
                            "variable count must match across MPI ranks"
                        )
                    minimum_local_dofs = comm.allreduce(
                        state_shape[1], op=MPI.MIN
                    )
                    if minimum_local_dofs <= 0:
                        raise ValueError(
                            "each MPI rank must own at least one spatial DOF"
                        )
            elif block.shape[:-1] != state_shape:
                raise ValueError("snapshot loader returned inconsistent state dimensions")

            flattened_block = block.reshape(block.shape[0], -1)
            block_count = flattened_block.shape[1]
            running_max = np.maximum(
                running_max, np.max(np.abs(flattened_block), axis=1)
            )
            running_abs_sum += np.sum(np.abs(flattened_block), axis=1)

            block_mean = np.mean(flattened_block, axis=1)
            block_m2 = np.sum(
                (flattened_block - block_mean[:, None])**2, axis=1
            )
            combined_count = running_count + block_count
            delta = block_mean - running_mean
            running_m2 += (
                block_m2
                + delta**2 * running_count * block_count / combined_count
            )
            running_mean += delta * block_count / combined_count
            running_count = combined_count

        if comm is not None and comm.Get_size() > 1:
            gathered_statistics = comm.allgather(
                (running_count, running_max, running_abs_sum,
                 running_mean, running_m2)
            )
            global_count = 0
            global_max = np.zeros_like(running_max)
            global_abs_sum = np.zeros_like(running_abs_sum)
            global_mean = np.zeros_like(running_mean)
            global_m2 = np.zeros_like(running_m2)
            for (local_count, local_max, local_abs_sum,
                 local_mean, local_m2) in gathered_statistics:
                global_max = np.maximum(global_max, local_max)
                global_abs_sum += local_abs_sum
                combined_count = global_count + local_count
                delta = local_mean - global_mean
                global_m2 += (
                    local_m2
                    + delta**2 * global_count * local_count / combined_count
                )
                global_mean += delta * local_count / combined_count
                global_count = combined_count
            running_count = global_count
            running_max = global_max
            running_abs_sum = global_abs_sum
            running_mean = global_mean
            running_m2 = global_m2

        if self.__scaling_type == "max_abs":
            scales = running_max
        elif self.__scaling_type == "mean_abs":
            scales = running_abs_sum / running_count
        elif self.__scaling_type == "variance":
            scales = np.sqrt(running_m2 / running_count)
        else:
            raise ValueError(
                f"Unknown variable scaling type: {self.__scaling_type}"
            )

        self.var_scales_ = np.where(scales < 1e-10, 1.0, scales)
        self.have_scales_been_initialized = True

    # These are all inplace operations
    def pre_scale(self, data_tensor: np.ndarray) -> None:
        '''
        Scales the input data matrix in place before processing, taking into account
        the previously initialized scaling factors.

        Args:
            data_tensor (np.ndarray): The input data matrix to be scaled.
        '''
        n_var = data_tensor.shape[0]
        if self.have_scales_been_initialized:
            pass
        else:
            self.initialize_scalings(data_tensor)
        # scale each field (variable scaling)
        for i in range(n_var):
            data_tensor[i] /= self.var_scales_[i]

    def post_scale(self, data_tensor: np.ndarray) -> None:
        '''
        Scales the input data matrix in place using the scaling vector.

        Args:
            data_tensor (np.ndarray): The input data matrix to be scaled.
        '''
        assert self.have_scales_been_initialized, "Scales in VariableScaler have not been initialized"
        # scale each field
        n_var = data_tensor.shape[0]
        for i in range(n_var):
            data_tensor[i] *= self.var_scales_[i]

class VariableAndVectorScaler:
    '''
    Concrete implementation designed to scale snapshot matrices involving
    multiple state variables by both the variable magnitudes and an additional
    vector.  This is particularly useful when wishing to perform POD for,
    e.g., a finite volume method where we want to scale by the cell volumes as
    well as the variable magnitudes. This implementation combines the
    VectorScaler and VariableScaler classes.

    This class conforms to the :class:`Scaler` protocol.
    '''

    def __init__(self, scaling_vector, scaling_type) -> None:
        '''
        Constructor for the VariableAndVectorScaler.

        Args:
            scaling_vector: Array containing the scaling vector for each row
            in the snapshot matrix.
            scaling_type: Scaling method ('max_abs',
            'mean_abs', or 'variance') for variable magnitudes.

        This constructor initializes the :class:`VariableAndVectorScaler` with the
        specified parameters.
        '''
        self.__my_variable_scaler = VariableScaler(scaling_type)
        self.__my_vector_scaler = VectorScaler(scaling_vector)

    def initialize_scalings_from_loader(
            self,
            snapshot_loader: SnapshotLoader,
            block_size: int,
            n_snapshots: int,
            comm=None) -> None:
        # This method is only required for streaming POD.
        self.__my_variable_scaler.initialize_scalings_from_loader(
            snapshot_loader, block_size, n_snapshots, comm
        )

    def pre_scale(self, data_tensor: np.ndarray) -> None:
        '''
        Scales the input data matrix in place before processing, first using the
        :class:`VariableScaler` and then the :class:`VectorScaler`.

        Args:
            data_tensor (np.ndarray): The input data matrix to be scaled.
        '''
        self.__my_variable_scaler.pre_scale(data_tensor)
        self.__my_vector_scaler.pre_scale(data_tensor)

    def post_scale(self, data_tensor: np.ndarray) -> None:
        '''
        Scales the input data matrix in place after processing, first using the
        :class:`VectorScaler` and then the :class:`VariableScaler`.

        Args:
            data_tensor (np.ndarray): The input data matrix to be scaled.
        '''
        self.__my_vector_scaler.post_scale(data_tensor)
        self.__my_variable_scaler.post_scale(data_tensor)
