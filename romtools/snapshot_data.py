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
Most ROM formulations require access to so-called snapshot data to construct a reduced trial space.
A snapshot is typically a solution to a full-order model.
As an example, consider a (discretized) parameterized PDE defined by

$$\\boldsymbol r( \\mathbf{u}(\\boldsymbol \\mu);\\boldsymbol \\mu)$$

where $\\boldsymbol r$ is the residual operator, $\\mathbf{u}$ is the state,
and $\\boldsymbol \\mu$ are system parameters.
Suppose we have solved the PDE for a set of $M$ training parameters to obtain
the so-called snapshot matrix

$$\\mathbf{S} = \\begin{bmatrix}
\\mathbf{u} (\\boldsymbol \\mu_1 ) &
\\cdots &
\\mathbf{u}(\\boldsymbol \\mu_M)
\\end{bmatrix}
$$


The SnapshotData class encapsulates the information contained in set of
snapshots, and is the main class used in the construction of trial spaces
'''

import abc
from typing import Iterable
import numpy as np


class AbstractSnapshotData(abc.ABC):
    '''
    An abstract base class for representing snapshot data.

    This class defines the common interface for classes that store and provide access to snapshot data
    as part of a simulation or data processing system. Implementations of this class are expected to
    define the initialization method and various methods for accessing and manipulating the data.

    Methods:
    '''

    @abc.abstractmethod
    def get_snapshot_tensor(self) -> np.ndarray:
        '''
        Returns numpy tensor of shape N_vars x N_space x N_samples
        (assuming each snapshot corresponds to a column vector)
        '''
        pass

    @abc.abstractmethod
    def get_mesh_gids(self):
        '''
        Retrieves global ids associated with mesh points (used for hyper-reduction)

        Returns:
            None or specific data type: The mesh global identifiers, or None if not applicable.

        Note:
        Subclasses must implement this method to provide access to mesh global identifiers if relevant.
        '''
        pass

    def get_snapshot_matrix(self) -> np.ndarray:
        '''
        Returns numpy matrix of shape N_vars N_space x N_samples
        (assuming each snapshot corresponds to a column vector)
        '''
        variable_ordering = 'C'
        snapshot_tensor = self.get_snapshot_tensor()
        matrix_shape = (snapshot_tensor.shape[0]*snapshot_tensor.shape[1],
                        snapshot_tensor.shape[2])
        snapshot_matrix = np.reshape(snapshot_tensor,
                                     matrix_shape,
                                     variable_ordering)
        return snapshot_matrix
