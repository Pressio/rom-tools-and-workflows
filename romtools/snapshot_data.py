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
    An abstract base class for representing snapshot data.

    This class defines the common interface for classes that store and provide access to snapshot data
    as part of a simulation or data processing system. Implementations of this class are expected to
    define the initialization method and various methods for accessing and manipulating the data.

    Attributes:
        var_names (list): A list of variable names associated with the snapshot data.

    Methods:
    """

    @abc.abstractmethod
    def __init__(self, **kwargs):
        """
        Initializes an instance of the AbstractSnapshotData class. Subclasses should implement
        this method to set up the necessary data structures or connections to data sources.

        Args:
            **kwargs: Additional keyword arguments that subclasses may accept for configuration.

        Note:
        Subclasses must call this base class constructor and set the `var_names` attribute to
        define the variable names associated with the snapshot data.
        """
        pass

    @abc.abstractmethod
    def getSnapshotsAsListOfArrays(self) -> Iterable[np.ndarray]:
        """
        Retrieves the snapshots as a list of NumPy arrays. Each array represents a single snapshot.

        Returns:
            Iterable[np.ndarray]: An iterable of NumPy arrays representing the snapshots.

        Note:
        Subclasses must implement this method to provide access to the actual snapshot data.
        """
        pass

    @abc.abstractmethod
    def getMeshGids(self):
        """
        Retrieves global ids associated with mesh points (used for hyper-reduction)

        Returns:
            None or specific data type: The mesh global identifiers, or None if not applicable.

        Note:
        Subclasses must implement this method to provide access to mesh global identifiers if relevant.
        """
        pass

    def getSnapshotsAsArray(self) -> np.ndarray:
        """
        Retrieves the snapshots as a single NumPy array by converting the list of snapshots into an array.

        Returns:
            np.ndarray: A NumPy array containing all the snapshots.

        Note:
        This method provides a convenient way to access the snapshot data as a single array.
        Subclasses can use the `getSnapshotsAsListOfArrays()` method to implement this.
        """
        snapshot_array = listOfSnapshotsToArray(self.getSnapshotsAsListOfArrays())
        return snapshot_array

    def getVariableNames(self) -> list:
        """
        Retrieves the names of different state variables associated with the snapshot data.

        Returns:
            list: A list of variable names.

        Note:
        Subclasses should ensure that this list is properly defined and set in the constructor.
        """
        return self.var_names

    def getNumVars(self) -> int:
        """
        Returns the number of state variables in the snapshot data
        (e.g., 5 for the compressible Navier--Stokes equations in 3 dimensions)

        Returns:
            int: The number of variables.

        Note:
        Subclasses should make sure that this method returns the correct number of variables
        associated with the snapshot data.
        """
        return len(self.get_variable_names())



def listOfSnapshotsToArray(list_of_snapshots: Iterable[np.ndarray]) -> np.ndarray:
  '''
  Helper function to move snapshot list into a matrix
  '''
  return np.hstack([ar.reshape(ar.shape[0],-1) for ar in list_of_snapshots])
