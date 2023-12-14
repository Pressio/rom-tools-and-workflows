'''
Constructing a basis via POD typically entails computing the SVD of a snapshot matrix,
$$ \\mathbf{U} ,\\mathbf{\\Sigma} = \\mathrm{svd}(\\mathbf{S})$$
and then selecting the first $K$ left singular vectors (i.e., the first $K$
columns of $\\mathbf{U}$). Typically, $K$ is determined through the decay of
the singular values.

The truncater class is desined to truncate a basis.
We provide concrete implementations that truncate based on a specified number
of basis vectors and the decay of the singular values
'''

import abc
import numpy as np


class AbstractTruncater(abc.ABC):
    '''
    Abstract implementation
    '''
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, basis: np.ndarray,  singular_values: np.ndarray) -> np.ndarray:
        '''
        Truncate left singular vectors
        '''
        pass


class NoOpTruncater(AbstractTruncater):
    '''
    No op implementation
    '''
    def __init__(self) -> None:
        pass

    def __call__(self, basis: np.ndarray,  singular_values: np.ndarray) -> np.ndarray:
        return basis


class BasisSizeTruncater(AbstractTruncater):
    '''
    Truncates to a specified number of singular vectors, as specified in the constructor
    '''
    def __init__(self, basis_dimension: int) -> None:
        self.__basis_dimension = basis_dimension

    def __call__(self, basis: np.ndarray, singular_values: np.ndarray) -> np.ndarray:
        return basis[:, :self.__basis_dimension]


class EnergyTruncater(AbstractTruncater):
    '''
    Truncates based on the decay of singular values, i.e., will define $K$ to
    be the number of singular values such that the cumulative energy retained
    is greater than some threshold.
    '''
    def __init__(self, threshold: float) -> None:
        self.energy_threshold_ = threshold

    def __call__(self, basis: np.ndarray, singular_values: np.ndarray) -> np.ndarray:
        energy = np.cumsum(singular_values**2)/np.sum(singular_values**2)
        basis_dimension = np.argmax(energy > self.energy_threshold_) + 1
        return basis[:, 0:basis_dimension]
