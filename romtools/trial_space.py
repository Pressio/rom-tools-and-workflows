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
#Trial space overview

A trial space is foundational to reduced-order models.
In a ROM, we restrict a high-dimensional state to live within a low-dimensional trial space.
Mathematically, for a "FOM" vector $\\mathbf{u} \\in \\mathbb{R}^N$, we represent this as
$$\\mathbf{u} \\approx \\tilde{\\mathbf{u}} \\in \\mathcal{V}$$
where $\\mathcal{V}$ with
$\\text{dim}(\\mathcal{V}) = K \\le N$
is the trial space. Formally, we can describe this low-dimensional representation with a basis and an affine offset,
$$\\tilde{\\mathbf{u}}  = \\boldsymbol \\Phi \\hat{\\mathbf{u}} + \\mathbf{u}_{\\mathrm{shift}}$$
where $\\boldsymbol \\Phi \\in \\mathbb{R}^{N \\times K}$ is the basis matrix,
$\\hat{\\mathbf{u}} \\in \\mathbb{R}^{K}$ are the reduced, or generalized coordinates,
$\\mathbf{u}_{\\mathrm{shift}} \\in \\mathbb{R}^N$ is the shift vector (or affine offset), and, by definition,
$\\mathcal{V} \\equiv \\mathrm{range}(\\boldsymbol \\Phi) + \\mathbf{u}_{\\mathrm{shift}}$.

The trial_space class encapsulates the information of an affine trial space, $\\mathcal{V}$,
by virtue of providing access to a basis matrix, a shift vector, and the dimensionality of the trial space.
'''

import abc
import numpy as np
from romtools.trial_space_utils.truncater import Truncater, NoOpTruncater
from romtools.trial_space_utils.shifter import Shifter, NoOpShifter
from romtools.trial_space_utils.scaler import Scaler
from romtools.trial_space_utils.splitter import Splitter, NoOpSplitter
from romtools.trial_space_utils.orthogonalizer import Orthogonalizer, NoOpOrthogonalizer


class AbstractTrialSpace(abc.ABC):
    '''
    Abstract base class for trial space implementations.

    This abstract class defines the interface for a trial space.

    Methods:
    '''

    @abc.abstractmethod
    def get_dimension(self):
        '''
        Retrieves the dimension of the trial space

        Returns:
            int: The dimension of the trial space.

        Concrete subclasses should implement this method to return the
        appropriate dimension for their specific trial space implementation.
        '''
        pass

    @abc.abstractmethod
    def get_shift_vector(self):
        '''
        Retrieves the shift vector of the trial space.

        Returns:
            np.ndarray: The shift vector.

        Concrete subclasses should implement this method to return the shift
        vector specific to their trial space implementation.
        '''
        pass

    @abc.abstractmethod
    def get_basis(self):
        '''
        Retrieves the basis vectors of the trial space.

        Returns:
            np.ndarray: The basis of the trial space.

        Concrete subclasses should implement this method to return the basis
        vectors specific to their trial space implementation.
        '''
        pass


def tensor_to_matrix(tensor_input):
    '''
    Converts a tensor of snapshots (N_vars x N_space x N_samples) to a matrix
    representation in which the first two dimension are collapsed
    (N_vars*N_space x N_samples).
    '''
    output_tensor = tensor_input.reshape(tensor_input.shape[0]*tensor_input.shape[1],
                                         tensor_input.shape[2])
    return output_tensor


def matrix_to_tensor(n_var, matrix_input):
    '''
    Inverse operation of `tensor_to_matrix`
    '''
    d1 = int(matrix_input.shape[0] / n_var)
    d2 = matrix_input.shape[1]
    output_matrix = matrix_input.reshape(n_var, d1, d2)
    return output_matrix


class DictionaryTrialSpace(AbstractTrialSpace):
    '''
    ##Reduced basis trial space (no truncation).

    Given a snapshot matrix $\\mathbf{S}$, we set the basis to be

    $$\\boldsymbol \\Phi = \\mathrm{orthogonalize}(\\mathrm{split}(\\mathbf{S} - \\mathbf{u}_{\\mathrm{shift}}))$$

    where the orthogonalization, splitting, and shifts are defined by their
    respective classes
    '''
    def __init__(self, snapshots, shifter, splitter, orthogonalizer):
        '''
        Constructor for the reduced basis trial space without truncation.

        Args:
            snapshots: Snapshot tensor containing solution data 
            shifter: Class that shifts the basis.
            splitter: Class that splitts the basis.
            orthogonalizer: Class that orthogonalizes the basis.

        This constructor initializes a trial space by performing basis
        manipulation operations on the provided snapshot data.
        '''

        # compute basis
        n_var = snapshots.shape[0]
        shifted_snapshots, self.__shift_vector = shifter(snapshots)
        snapshot_matrix = tensor_to_matrix(shifted_snapshots)
        self.__basis = splitter(snapshot_matrix)
        self.__basis = orthogonalizer(self.__basis)
        self.__basis = matrix_to_tensor(n_var, self.__basis)
        self.__dimension = self.__basis.shape[2]

    def get_dimension(self):
        '''
        Retrieves the dimension of trial space

        Returns:
            int: The dimension of the trial space.
        '''
        return self.__dimension

    def get_shift_vector(self):
        '''
        Retrieves the shift vector

        Returns:
            np.ndarray: The shift vector.

        '''
        return self.__shift_vector

    def get_basis(self):
        '''
        Retrieves the basis of the trial space

        Returns:
            np.ndarray: The basis of the trial space.
        '''
        return self.__basis


class TrialSpaceFromPOD(AbstractTrialSpace):
    '''
    ##POD trial space (constructed via SVD).

    Given a snapshot matrix $\\mathbf{S}$, we compute the basis $\\boldsymbol \\Phi$ as

    $$\\boldsymbol U = \\mathrm{SVD}(\\mathrm{split}(\\mathbf{S} - \\mathbf{u}_{\\mathrm{shift}})))$$
    $$\\boldsymbol \\Phi = \\mathrm{orthogonalize}(\\mathrm{truncate}( \\boldsymbol U ))$$

    where $\\boldsymbol U$ are the left singular vectors and the
    orthogonalization, truncation, splitting, and shifts are defined
    by their respective classes.

    For truncation, we enable truncation based on a fixed dimension or the
    decay of singular values; please refer to the documentation for the
    truncater.
    '''

    def __init__(self,
                 snapshot_tensor,
                 truncater:      Truncater      = NoOpTruncater(),
                 shifter:        Shifter        = NoOpShifter(),
                 splitter:       Splitter       = NoOpSplitter(),
                 orthogonalizer: Orthogonalizer = NoOpOrthogonalizer(),
                 svdFnc = None):
        '''
        Constructor for the POD trial space.

        Args:
            snapshot_tensor (np.ndarray): Snapshot data tensor
            truncater (Truncater): Class that truncates the basis.
            shifter (Shifter): Class that shifts the basis.
            splitter (Splitter): Class that splits the basis.
            orthogonalizer (Orthogonalizer): Class that orthogonalizes
                the basis.
            svdFnc: a callable to use for computing the SVD on the snapshots data.
                IMPORTANT: must conform to the API of [np.linalg.svd](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html#numpy-linalg-svd).
                If `None`, internally we use `np.linalg.svd`.
                Note: this is useful when you want to use a custom svd, for
                    example when your snapshots are distributed with MPI, or
                    maybe you have a fancy svd function that you can use.

        This constructor initializes a POD trial space by performing SVD on
        the provided snapshot data and applying various basis manipulation
        operations, including truncation, shifting, splitting, and
        orthogonalization.
        '''

        n_var = snapshot_tensor.shape[0]
        shifted_snapshot_tensor, self.__shift_vector = shifter(snapshot_tensor)
        snapshot_matrix = tensor_to_matrix(shifted_snapshot_tensor)
        shifted_split_snapshots = splitter(snapshot_matrix)

        svd_picked = np.linalg.svd if svdFnc is None else svdFnc
        lsv, svals, _ = svd_picked(shifted_split_snapshots, full_matrices=False,
                                   compute_uv=True, hermitian=False)

        self.__basis = truncater(lsv, svals)
        self.__basis = orthogonalizer(self.__basis)
        self.__basis = matrix_to_tensor(n_var, self.__basis)
        self.__dimension = self.__basis.shape[2]

    def get_dimension(self):
        '''
        Retrieves the dimension of the trial space

        Returns:
            int: The dimension of the trial space.
        '''
        return self.__dimension

    def get_shift_vector(self):
        '''
        Retrieves the shift vector

        Returns:
            np.ndarray: The shift vector.

        '''
        return self.__shift_vector

    def get_basis(self):
        '''
        Retrieves the basis of the trial space

        Returns:
            np.ndarray: The basis of the trial space.
        '''
        return self.__basis


class TrialSpaceFromScaledPOD(AbstractTrialSpace):
    '''
    ##POD trial space (constructed via scaled SVD).

    Given a snapshot matrix $\\mathbf{S}$, we set the basis to be

    $$\\boldsymbol U = \\mathrm{SVD}(\\mathrm{split}(\\mathrm{prescale}(\\mathbf{S} - \\mathbf{u}_{\\mathrm{shift}})))$$
    $$\\boldsymbol \\Phi = \\mathrm{orthogonalize}(\\mathrm{postscale}(\\mathrm{truncate}( \\boldsymbol U )))$$

    where $\\boldsymbol U$ are the left singular vectors and the
    orthogonalization, truncation, splitting, and shifts are defined by their
    respective classes.

    For truncation, we enable truncation based on a fixed dimension or the
    decay of singular values; please refer to the documentation for the
    truncater.
    '''

    def __init__(self, snapshot_tensor,
                 truncater: Truncater,
                 shifter: Shifter,
                 scaler: Scaler,
                 splitter: Splitter,
                 orthogonalizer: Orthogonalizer):
        '''
        Constructor for the POD trial space constructed via scaled SVD.

        Args:
            snapshot_tensor: np.ndarray snapshot tensor 
            truncater: Class that truncates the basis.
            shifter: Class that shifts the basis.
            scaler: Class that scales the basis.
            splitter: Class that splits the basis.
            orthogonalizer: Class that orthogonalizes the basis.

        This constructor initializes a POD trial space by performing SVD on
        the provided snapshot data and applying various basis manipulation
        operations, including scaling, shifting, truncation, splitting, and
        orthogonalization.
        '''

        # compute basis
        n_var = snapshot_tensor.shape[0]
        shifted_snapshot_tensor, self.__shift_vector = shifter(snapshot_tensor)
        scaled_shifted_snapshot_tensor = scaler.pre_scaling(shifted_snapshot_tensor)
        snapshot_matrix = tensor_to_matrix(scaled_shifted_snapshot_tensor)
        snapshot_matrix = splitter(snapshot_matrix)

        lsv, svals, _ = np.linalg.svd(snapshot_matrix, full_matrices=False)
        self.__basis = truncater(lsv, svals)
        self.__basis = matrix_to_tensor(n_var, self.__basis)
        self.__basis = scaler.post_scaling(self.__basis)
        self.__basis = tensor_to_matrix(self.__basis)
        self.__basis = orthogonalizer(self.__basis)
        self.__basis = matrix_to_tensor(n_var, self.__basis)
        self.__dimension = self.__basis.shape[2]

    def get_dimension(self):
        '''
        Retrieves the dimension of the trial space

        Returns:
            int: The dimension of the trial space.
        '''
        return self.__dimension

    def get_shift_vector(self):
        '''
        Retrieves the shift vector

        Returns:
            np.ndarray: The shift vector.

        '''
        return self.__shift_vector

    def get_basis(self):
        '''
        Retrieves the basis of the trial space

        Returns:
            np.ndarray: The basis of the trial space.
        '''
        return self.__basis
