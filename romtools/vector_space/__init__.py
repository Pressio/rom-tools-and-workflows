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
This module defines the API to work with a vector subspace.
A vector subspace is foundational to reduced-order models.
In a ROM, a high-dimensional state is restricted to live within a low-dimensional vector space, known as a trial space.
Mathematically, given a "FOM" vector $\\mathbf{u} \\in \\mathbb{R}^{N_{\\mathrm{vars}} N_{\\mathrm{x}}}$, we can write
$$\\mathbf{u} \\approx \\tilde{\\mathbf{u}} \\in \\mathcal{V} + \\mathbf{u}_{\\mathrm{shift}}$$
where
- $\\mathcal{V}$ with $\\text{dim}(\\mathcal{V}) = K \\le N_{\\mathrm{vars}}  N_{\\mathrm{x}}$ is the trial space
- $N_{\\mathrm{vars}}$ is the number of PDE variables (e.g., 5 for the compressible Navier-Stokes equations in 3D)
- $N_{\\mathrm{x}}$ is the number of spatial DOFs

Formally, we can describe this low-dimensional representation with a basis and an affine offset,
$$\\tilde{\\mathbf{u}}  = \\boldsymbol \\Phi \\hat{\\mathbf{u}} + \\mathbf{u}_{\\mathrm{shift}}$$
where $\\boldsymbol \\Phi \\in \\mathbb{R}^{ N_{\\mathrm{vars}}  N_{\\mathrm{x}} \\times K}$ is the basis matrix
($K$ is the number of basis), $\\hat{\\mathbf{u}} \\in \\mathbb{R}^{K}$ are the reduced, or generalized coordinates,
$\\mathbf{u}_{\\mathrm{shift}} \\in \\mathbb{R}^{ N_{\\mathrm{vars}}  N_{\\mathrm{x}}}$ is the shift vector (or affine offset),
and, by definition, $\\mathcal{V} \\equiv \\mathrm{range}(\\boldsymbol \\Phi)$.

The `VectorSpace` abstract class defined below encapsulates the information of an affine vector space, $\\mathcal{V}$,
by virtue of providing access to a basis matrix, a shift vector, and the dimensionality of the vector space,
while decoupling this representation from *how* it is computed.

####We rely on a tensor representation!

Our representation of the basis and the affine offset for a vector space is based on tensors
$$\\mathcal{\Phi} \\in \mathbb{R}^{ N_{\\mathrm{vars}} \\times N_{\\mathrm{x}} \\times K},$$
$$\\mathcal{u}_{\\mathrm{shift}} \\in \mathbb{R}^{ N_{\\mathrm{vars}} \\times N_{\\mathrm{x}}}.$$
Internally, we remark that all tensors are reshaped into 2D matrices, e.g., when performing SVD.

###**Content**

We currently provide the following concrete classes:

- `DictionaryVectorSpace`: construct a vector space from a matrix without truncation.

- `VectorSpaceFromPOD`: construct a vector subspace computed via SVD.

which derive from the abstract class `VectorSpace`.

---
##**API**
'''

from typing import Tuple, Protocol, Callable
import numpy as np
from romtools.vector_space.utils.truncater import LeftSingularVectorTruncater, NoOpTruncater
from romtools.vector_space.utils.shifter import Shifter, create_noop_shifter
from romtools.vector_space.utils.scaler import Scaler, NoOpScaler
from romtools.vector_space.utils.orthogonalizer import Orthogonalizer, NoOpOrthogonalizer


class VectorSpace(Protocol):
    '''
    Abstract base class for vector space implementations.

    Methods:
    '''

    def extents(self) -> Tuple[int, int, int]:
        '''
        Retrieves the dimension of the vector space

        Returns:
            A Tuple with the the dimensions of the vector space (n_var,nx,K).
        '''
        ...

    def get_shift_vector(self) -> np.ndarray:
        '''
        Retrieves the shift vector of the vector space.

        Returns:
            `np.ndarray`: The shift vector in tensorm form.
        '''
        ...

    def get_basis(self) -> np.ndarray:
        '''
        Retrieves the basis vectors of the vector space.

        Returns:
            `np.ndarray`: The basis of the vector space in tensor form.
        '''
        ...


class DictionaryVectorSpace():
    '''
    Reduced basis vector space (no truncation).

    This class conforms to `VectorSpace` protocol.

    Given a snapshot matrix $\\mathbf{S}$, we set the basis to be

    $$\\boldsymbol \\Phi = \\mathrm{orthogonalize}(\\mathbf{S} - \\mathbf{u}_{\\mathrm{shift}})$$

    where the orthogonalization and shifts are defined by their
    respective classes
    '''

    def __init__(self,
                 snapshots,
                 shifter:        Shifter        = None,
                 orthogonalizer: Orthogonalizer = NoOpOrthogonalizer()) -> None:
        '''
        Constructor.

        Args:
            snapshots (np.ndarray): Snapshot data in tensor form
                $\in \mathbb{R}^{ N_{\\mathrm{vars}} \\times N_{\\mathrm{x}} \\times N_{samples}}$
            shifter: Class that shifts the basis.
            orthogonalizer: Class that orthogonalizes the basis.

        This constructor initializes a vector space by performing basis
        manipulation operations on the provided snapshot data.
        '''
        # Create noop shifter if not provided
        if shifter is None:
            shifter = create_noop_shifter(snapshots)

        # compute basis
        n_var = snapshots.shape[0]
        shifter.apply_shift(snapshots)
        snapshot_matrix = _tensor_to_matrix(snapshots)
        self.__basis = orthogonalizer.orthogonalize(snapshot_matrix)
        self.__basis = _matrix_to_tensor(n_var, self.__basis)
        self.__shift_vector = shifter.get_shift_vector()

    def get_shift_vector(self) -> np.ndarray:
        '''
        Concrete implementation of `VectorSpace.get_shift_vector()`
        '''
        return self.__shift_vector

    def get_basis(self) -> np.ndarray:
        '''
        Concrete implementation of `VectorSpace.get_basis()`
        '''
        return self.__basis

    def extents(self) -> Tuple[int, int, int]:
        return self.__basis.shape


class VectorSpaceFromPOD():
    '''
    POD vector space (constructed via SVD).

    This class conforms to `VectorSpace` protocol.

    Given a snapshot matrix $\\mathbf{S}$, we compute the basis $\\boldsymbol \\Phi$ as


    $$\\boldsymbol U = \\mathrm{SVD}(\\mathrm{prescale}(\\mathbf{S} - \\mathbf{u}_{\\mathrm{shift}}))$$
    $$\\boldsymbol \\Phi = \\mathrm{orthogonalize}(\\mathrm{postscale}(\\mathrm{truncate}( \\boldsymbol U )))$$

    where $\\boldsymbol U$ are the left singular vectors and the
    orthogonalization, truncation, scaling, and shifts are defined by their respective classes.

    For truncation, we enable truncation based on a fixed dimension or the
    decay of singular values; please refer to the documentation for the truncater.
    '''

    def __init__(self,
                 snapshots,
                 truncater:      LeftSingularVectorTruncater   = NoOpTruncater(),
                 shifter:        Shifter        = None,
                 orthogonalizer: Orthogonalizer = NoOpOrthogonalizer(),
                 scaler:         Scaler         = NoOpScaler(),
                 svdFnc:         Callable       = None) -> None:
        '''
        Constructor.

        Args:
            snapshots (np.ndarray): Snapshot data in tensor form
                $\in \mathbb{R}^{ N_{\\mathrm{vars}} \\times N_{\\mathrm{x}} \\times N_{samples}}$
            truncater (Truncater): Concrete implementation for truncating the basis.
            shifter (Shifter): Concrete implementation responsible for shifting the basis.
            orthogonalizer (Orthogonalizer): Concrete implementation that orthogonalizes the basis.
            scaler: Concrete implementation that scales the basis.
            svdFnc: a callable to use for computing the SVD on the snapshots data.
                IMPORTANT: must conform to the API of
                [np.linalg.svd](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html#numpy-linalg-svd).
                If `None`, internally we use `np.linalg.svd`.
                Note: this is useful when you want to use a custom svd, for
                    example when your snapshots are distributed with MPI, or
                    maybe you have a fancy svd function that you can use.

        This constructor initializes a POD vector space by performing SVD on the provided snapshot
        data and applying various basis manipulation operations, including truncation, shifting, scaling,
        and orthogonalization.
        '''
        if shifter is None:
            shifter = create_noop_shifter(snapshots)
        n_var = snapshots.shape[0]
        shifter.apply_shift(snapshots)
        scaled_shifted_snapshots = scaler.pre_scale(snapshots)
        snapshot_matrix = _tensor_to_matrix(scaled_shifted_snapshots)
        svd_picked = np.linalg.svd if svdFnc is None else svdFnc
        lsv, svals, _ = svd_picked(snapshot_matrix, full_matrices=False,
                                   compute_uv=True, hermitian=False)

        self.__basis = truncater.truncate(lsv, svals)
        self.__basis = _matrix_to_tensor(n_var, self.__basis)
        self.__basis = scaler.post_scale(self.__basis)
        self.__basis = _tensor_to_matrix(self.__basis)
        self.__basis = orthogonalizer.orthogonalize(self.__basis)
        self.__basis = _matrix_to_tensor(n_var, self.__basis)
        self.__shift_vector = shifter.get_shift_vector()

    def get_shift_vector(self) -> np.ndarray:
        '''
        Concrete implementation of `VectorSpace.get_shift_vector()`
        '''
        return self.__shift_vector

    def get_basis(self) -> np.ndarray:
        '''
        Concrete implementation of `VectorSpace.get_basis()`
        '''
        return self.__basis

    def extents(self) -> Tuple[int, int, int]:
        return self.__basis.shape


def _tensor_to_matrix(tensor_input: np.ndarray) -> np.ndarray:
    '''
    Converts a tensor with shape $[N, M, P]$ to a matrix representation
    in which the first two dimension are collapsed $[N M, P]$.
    '''
    output_tensor = tensor_input.reshape(tensor_input.shape[0]*tensor_input.shape[1],
                                         tensor_input.shape[2])
    return output_tensor


def _matrix_to_tensor(n_var: int, matrix_input: np.ndarray) -> np.ndarray:
    '''
    Inverse operation of `_tensor_to_matrix`
    '''
    d1 = int(matrix_input.shape[0] / n_var)
    d2 = matrix_input.shape[1]
    output_matrix = matrix_input.reshape(n_var, d1, d2)
    return output_matrix
