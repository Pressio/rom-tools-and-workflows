"""
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


The SnapshotData class encapsulates the information contained in set of snapshots,
and is the main class used in the construction of trial spaces
"""

import numpy as np
import abc
from typing import Iterable

class AbstractSnapshotData(abc.ABC):
    """
    Abstract base class for snapshot data
    """
    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def getSnapshotTensor(self) -> np.ndarray:
        """
        Returns numpy tensor of shape N_vars x N_space x N_samples 
        (assuming each snapshot corresponds to a column vector)
        """
        pass 

    @abc.abstractmethod
    def getMeshGids(self):
        """
        Returns global ids associated with mesh points (used for hyper-reduction)
        """
        pass

    def getSnapshotMatrix(self) -> np.ndarray:
        """
        Returns numpy matrix of shape N_vars N_space x N_samples 
        (assuming each snapshot corresponds to a column vector)
        """
        variable_ordering = 'C'
        snapshot_tensor = self.getSnapshotTensor()
        snapshot_matrix = np.reshape(snapshot_tensor,(snapshot_tensor.shape[0]*snapshot_tensor.shape[1],snapshot_tensor.shape[2]),variable_ordering)
        return snapshot_matrix 
       
