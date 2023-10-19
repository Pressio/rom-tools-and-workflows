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


The SnapshotData class encapsulates the information contained in set of snapshots,
and is the main class used in the construction of trial spaces
'''

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
    def getSnapshotsAsListOfArrays(self) -> Iterable[np.ndarray]:
        """
        Return snapshot matrix as list of arrays
        (e.g., each element in the list could be its own snapshot matrix)
        """
        pass

    @abc.abstractmethod
    def getMeshGids(self):
        """
        Returns global ids associated with mesh points (used for hyper-reduction)
        """
        pass

    def getSnapshotsAsArray(self) -> np.ndarray:
        snapshot_array = listOfSnapshotsToArray(self.getSnapshotsAsListOfArrays())
        return snapshot_array

    def getVariableNames(self) -> list:
        """
        Returns the names of different state variables
        """
        return self.var_names

    def getNumVars(self) -> int:
        """
        Returns the number of state variables
        (e.g., 5 for the compressible Navier--Stokes equations in 3 dimensions)
        """
        return len(self.get_variable_names())



def listOfSnapshotsToArray(list_of_snapshots: Iterable[np.ndarray]) -> np.ndarray:
  '''
  Helper function to move snapshot list into a matrix
  '''
  return np.hstack([ar.reshape(ar.shape[0],-1) for ar in list_of_snapshots])
