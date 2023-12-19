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
---
##**Notes**
The scaler class is used to performed scaled POD. Scaling is applied to tensors of shape $\mathbb{R}^{ N_{\\mathrm{vars}} \\times N_{\\mathrm{x}} \\times K}$. These tensors are then reshaped into matrices when performing SVD.

___
##**Theory**

*What is scaled POD, and why would I do it?*

Standard POD computes a basis that minimizes the projection error in a standard Euclidean $\\ell^2$ inner product,
i.e., for a snapshot matrix $\\mathbf{S}$, POD computes the basis by solving the minimization problem
(assuming no affine offset)
$$ \\boldsymbol \\Phi = \\underset{ \\boldsymbol \\Phi_{\\*} \\in \\mathbb{R}^{N \\times K} | \\boldsymbol
\\Phi_{\\*}^T \\boldsymbol \\Phi_{\\*} = \\mathbf{I}}{ \\mathrm{arg \\; min} } \\| \\Phi_{\\*} \\Phi_{\\*}^T
\\mathbf{S} - \\mathbf{S} \\|_2.$$
In this minimization problem, errors are measured in a standard $\\ell^2$ norm.
For most practical applications, where our snapshot matrix involves variables of different scales,
this norm does not make sense (both intuitively, and on dimensional grounds).
As a practical example, consider fluid dynamics where the total energy is orders of magnitude larger than the density.

One of the most common approaches for mitigating this issue is to perform scaled POD.
In scaled POD, we solve a minimization problem on a scaled snapshot matrix.
Defining $\\mathbf{S}_{\\*} = \\mathbf{W}^{-1} \\mathbf{S}$, where $\\mathbf{W}$ is a weighting matrix
(e.g., a diagonal matrix containing the max absolute value of each state variable),
we compute the basis as the solution to the minimization problem
$$ \\boldsymbol \\Phi = \\mathbf{W} \\underset{ \\boldsymbol \\Phi_{\\*} \\in \\mathbb{R}^{N \\times K} |\\boldsymbol
\\Phi_{\\*}^T \\boldsymbol \\Phi_{\\*} = \\mathbf{I}}{ \\mathrm{arg \\; min} } \\| \\Phi_{\\*} \\Phi_{\\*}^T
\\mathbf{S}_{\\*} - \\mathbf{S}_{\\*} \\|_2.$$

The Scaler encapsulates this information

___
___
##**API**
'''

import abc
import numpy as np


class AbstractScaler(abc.ABC):
    '''
    Abstract base class
    '''

    @abc.abstractmethod
    def pre_scaling(self, data_tensor: np.ndarray) -> np.ndarray:
        '''
        Scales the snapshot matrix before performing SVD

        Args:
          $(N_{\\mathrm{vars}} ,N_{\\mathrm{x}} , K)$ np.ndarray: data matrix in tensor form
          
        Returns:
          $(N_{\\mathrm{vars}} ,N_{\\mathrm{x}} , K)$ np.ndarray: scaled data matrix in tensor form

        '''
        pass

    @abc.abstractmethod
    def post_scaling(self, data_tensor: np.ndarray) -> np.ndarray:
        '''
        Scales the left singular vectors after performing SVD

        Args:
          $(N_{\\mathrm{vars}} ,N_{\\mathrm{x}} , K)$ np.ndarray: left singular vectors 

        Returns:
          $(N_{\\mathrm{vars}} ,N_{\\mathrm{x}} , K)$ np.ndarray: unscaled left singular vectors 

        '''
        pass


class NoOpScaler(AbstractScaler):
    '''
    No op implementation
    '''
    def __init__(self):
        pass

    def pre_scaling(self, data_tensor):
        return data_tensor

    def post_scaling(self, data_tensor):
        return data_tensor


class VectorScaler(AbstractScaler):
    '''
    Concrete implementation designed to scale snapshot matrices by a vector.
    For a snapshot tensor $\\mathcal{S} \\in \\mathbb{R}^{N_{\\mathrm{vars}} \\times N_{\\mathrm{x}} \\times K}$, the VectorScaler
    accepts in a scaling vector $\\mathbf{v} \\in \\mathbb{R}^{N_{\\mathrm{x}}}$, and scales by
    $$\\mathcal{S}_i^* = \\mathrm{diag}(\\mathbf{v})^{-1} \\mathcal{S}_i,$$
    for $i=1,\\ldots, N_{\\mathrm{vars}}$,
    before performing POD (i.e., POD is performed on matrix form of $\\mathcal{S}^*$). After POD is performed, the tensorized bases
    are post-scaled by $$\\mathcal{ \\Phi}_i = \\mathrm{diag}(\\mathbf{v}) \\mathcal{U}_i$$
    for $i=1,\\ldots, N_{\\mathrm{vars}}$ where $\\mathcal{U}_i$ are the tensorized left singular vectors.

    **Note that scaling can cause bases to not be orthonormal; we do not
    recommend using scalers with the NoOpOrthonormalizer**
    '''
    def __init__(self, scaling_vector):
        '''
        Constructor for the VectorScaler.

        Args:
            scaling_vector, ($N_{\\mathrm{x}}$) np.ndarray: Array containing the scaling vector

        This constructor initializes the VectorScaler with the specified
        scaling vector.
        '''
        self.__scaling_vector_matrix = scaling_vector
        self.__scaling_vector_matrix_inv = 1./scaling_vector

    def pre_scaling(self, data_tensor):
        return self.__scaling_vector_matrix_inv[None, :, None] * data_tensor

    def post_scaling(self, data_tensor):
        return self.__scaling_vector_matrix[None, :, None] * data_tensor


class VariableScaler(AbstractScaler):
    '''
    Concrete implementation designed for snapshot matrices involving multiple
    state variables.

    This class is designed to scale a data matrix comprising multiple states
    (e.g., for the Navier--Stokes, rho, rho u, rhoE)

    This scaler will scale each variable based on
      - max-abs scaling: for the $i$th state variable $u_i$, we will compute the scaling as
        $s_i = \\mathrm{max}( \\mathrm{abs}( \\mathcal{S}_i ) )$, where $\\mathcal{S}_i$ denotes the snapshot matrix of the $i$th variable.
      - mean abs: for the $i$th state variable $u_i$, we will compute the scaling as
        $s_i = \\mathrm{mean}( \\mathrm{abs}( \\mathcal{S}_i ) )$, where $\\mathcal{S}_i$ denotes the snapshot matrix of the $i$th variable.
      - variance: for the $i$th state variable $u_i$, we will compute the scaling as
        $s_i = \\mathrm{std}( \\mathcal{S}_i ) $, where $\\mathcal{S}_i$ denotes the snapshot matrix of the $i$th variable.
    '''
    def __init__(self, scaling_type):
        '''
        Constructor for the VariableScaler.

        Args:
            scaling_type (str): The scaling method to use ('max_abs','mean_abs', or 'variance').

        This constructor initializes the VariableScaler with the specified
        scaling type, variable ordering, and number of variables.
        '''
        self.__scaling_type = scaling_type
        self.have_scales_been_initialized = False
        self.var_scales_ = None

    def initialize_scalings(self, data_tensor):
        '''
        Initializes the scaling factors for each state variable based on the
        specified method.

        Args:
            data_tensor, $(N_{\\mathrm{vars}} ,N_{\\mathrm{x}} , K)$ np.ndarray: data matrix in tensor form
        '''
        n_var = data_tensor.shape[0]
        self.var_scales_ = np.ones(n_var)
        for i in range(n_var):
            ith_var = data_tensor[i]
            if self.__scaling_type == 'max_abs':
                var_scale = np.max(abs(ith_var))
            elif self.__scaling_type == 'mean_abs':
                var_scale = np.mean(abs(ith_var))
            elif self.__scaling_type == 'variance':
                var_scale = np.sqrt(np.var(ith_var))

            # in case of a zero field (e.g., 2D)
            if var_scale < 1e-10:
                var_scale = 1.
            self.var_scales_[i] = var_scale
        self.have_scales_been_initialized = True

    # These are all inplace operations
    def pre_scaling(self, data_tensor):
        n_var = data_tensor.shape[0]
        if self.have_scales_been_initialized:
            pass
        else:
            self.initialize_scalings(data_tensor)
        # scale each field (variable scaling)
        for i in range(n_var):
            data_tensor[i] = data_tensor[i] / self.var_scales_[i]
        return data_tensor

    def post_scaling(self, data_tensor):
        assert self.have_scales_been_initialized, "Scales in VariableScaler have not been initialized"
        # scale each field
        n_var = data_tensor.shape[0]
        for i in range(n_var):
            data_tensor[i] = data_tensor[i]*self.var_scales_[i]
        return data_tensor


class VariableAndVectorScaler(AbstractScaler):
    '''
    Concrete implementation designed to scale snapshot matrices involving
    multiple state variables by both the variable magnitudes and an additional
    vector.  This is particularly useful when wishing to perform POD for,
    e.g., a finite volume method where we want to scale by the cell volumes as
    well as the variable magnitudes. This implementation combines the
    VectorScaler and VariableScaler classes.
    '''

    def __init__(self, scaling_vector, scaling_type):
        '''
        Constructor for the VariableAndVectorScaler.

        Args:
            scaling_vector: Array containing the scaling vector for each row in the snapshot matrix.
            scaling_type: Scaling method ('max_abs','mean_abs', or 'variance') for variable magnitudes.

        This constructor initializes the `VariableAndVectorScaler` with the
        specified parameters.
        '''
        self.__my_variable_scaler = VariableScaler(scaling_type)
        self.__my_vector_scaler = VectorScaler(scaling_vector)

    def pre_scaling(self, data_tensor):
        '''
        Scales the input data matrix before processing, first using the
        `VariableScaler` and then the `VectorScaler`.

        Args:
            data_tensor (np.ndarray): The input data matrix to be scaled.

        Returns:
            np.ndarray: The scaled data matrix.
        '''
        data_tensor = self.__my_variable_scaler.pre_scaling(data_tensor)
        return self.__my_vector_scaler.pre_scaling(data_tensor)

    def post_scaling(self, data_tensor):
        '''
        Scales the input data matrix after processing, first using the
        `VectorScaler` and then the `VariableScaler`.

        Args:
            data_tensor (np.ndarray): The input data matrix to be scaled.

        Returns:
            np.ndarray: The scaled data matrix.
        '''
        data_tensor = self.__my_vector_scaler.post_scaling(data_tensor)
        return self.__my_variable_scaler.post_scaling(data_tensor)
