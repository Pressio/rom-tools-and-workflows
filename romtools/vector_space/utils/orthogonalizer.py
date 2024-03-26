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
The OrthogonalizerClass is used to orthogonalize a basis at the end of the
construction of a vector space.  Specifically, given a basis
$$\\boldsymbol \\Phi \\in \\mathbb{R}^{N \\times K},$$
the orthogonalizer will compute a new, orthogonalized basis $\\boldsymbol \\Phi_{\\*}$
where
$$\\boldsymbol \\Phi_{\\*}^T \\mathbf{W} \\boldsymbol \\Phi_{\\*} = \\mathbf{I}.$$
In the above, $\\mathbf{W}$ is a weighting matrix (typically the cell volumes).
'''

from typing import Protocol
import numpy as np
import scipy.sparse


class Orthogonalizer(Protocol):
    '''
    Interface for the Orthogonalizer class.
    '''
    def orthogonalize(self, my_array: np.ndarray) -> np.ndarray:
        ...


class NoOpOrthogonalizer:
    '''
    No op class (doesn't do anything)
    '''
    def __init__(self) -> None:
        pass

    def orthogonalize(self, my_array: np.ndarray) -> np.ndarray:
        return my_array


class EuclideanL2Orthogonalizer:
    '''
    Orthogonalizes the basis in the standard Euclidean L2 inner product, i.e.,
    the output basis will satisfy
    $$\\boldsymbol \\Phi_{\\*}^T \\boldsymbol \\Phi_{\\*} = \\mathbf{I}.$$
    '''
    def __init__(self, qrFnc=None) -> None:
        '''
        Constructor
        Args:

            qrFnc: a callable to use for computing the QR decomposition.
                    IMPORTANT: must conform to the API of [np.linalg.qr](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html).
                    If `None`, internally we use `np.linalg.qr`.
                    Note: this is useful when you want to use a custom qr, for example when your snapshots are
                    distributed with MPI, or maybe you have a fancy qr function that you can use.

        '''
        self.__qr_picked = np.linalg.qr if qrFnc is None else qrFnc

    def orthogonalize(self, my_array: np.ndarray) -> np.ndarray:
        my_array, _ = self.__qr_picked(my_array, mode='reduced')
        return my_array


class EuclideanVectorWeightedL2Orthogonalizer:
    '''
    Orthogonalizes the basis in vector-weighted Euclidean L2 inner product,
    i.e., the output basis will satisfy
    $$\\boldsymbol \\Phi_{\\*}^T \\mathrm{diag}(\\mathbf{w})\\boldsymbol \\Phi_{\\*} = \\mathbf{I},$$
    where $\\mathbf{w}$ is the weighting vector. Typically, this inner product
    is used for orthogonalizing with respect to cell volumes
    '''
    def __init__(self, weighting_vector: np.ndarray, qrFnc=None) -> None:
        '''
        Constructor for the EuclideanVectorWeightedL2Orthogonalizer that
        initializes the orthogonalizer with the provided weighting vector and
        an optional custom QR decomposition function.
        Args:
            weighting_vector (np.ndarray): a 1-D NumPy array that the matrix will be orthogonalized against. The
                length of the array must match the number of rows in the matrix that will be orthogonalized.
            qrFnc: a callable to use for computing the QR decomposition.
                    IMPORTANT: must conform to the API of [np.linalg.qr](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html).
                    If `None`, internally we use `np.linalg.qr`.
                    Note: this is useful when you want to use a custom qr, for example when your snapshots are
                    distributed with MPI, or maybe you have a fancy qr function that you can use.
        '''

        self.__weighting_vector = weighting_vector
        self.__qr_picked = np.linalg.qr if qrFnc is None else qrFnc

    def orthogonalize(self, my_array: np.ndarray) -> np.ndarray:
        '''
        Orthogonalizes the input matrix using the specified weighting vector and returns the orthogonalized matrix.

        Args:
            my_array (np.ndarray): The input matrix to be orthogonalized.

        Returns:
            np.ndarray: The orthogonalized matrix.
        '''
        assert my_array.shape[0] == self.__weighting_vector.size, "Weighting vector does not match basis size"
        tmp = scipy.sparse.diags(np.sqrt(self.__weighting_vector)) @ my_array
        my_array, _ = self.__qr_picked(tmp, mode='reduced')
        my_array = scipy.sparse.diags(np.sqrt(1./self.__weighting_vector)) @ my_array
        return my_array
