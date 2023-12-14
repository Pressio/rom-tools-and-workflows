'''
The Shifter class is used to create an affine offset in a trial space.

*What is a shift vector, and why would I use it?* In ROMs, we restrict a state to belong to a low-dimensional affine trial space,
$$\\mathbf{u} \\approx \\tilde{\\mathbf{u}} \\in \\mathcal{V}$$
where
$\\mathcal{V} \\equiv \\mathrm{range}(\\boldsymbol \\Phi) + \\mathbf{u}_{\\mathrm{shift}}$
is the trial space. Here $\\mathbf{u}_{\\mathrm{shift}}$ defines an affine offset.
Affine offsets can be useful for a variety of reasons, including satisfying boundary conditions, and satisfying initial conditions.

The Shifter class encapsulates the affine offset.
'''

from typing import Tuple
import abc
import numpy as np


class AbstractShifter(abc.ABC):
    '''
    Abstract implmentation
    '''
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, my_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Overload to apply shift
        '''
        pass


class NoOpShifter(AbstractShifter):
    '''
    No op implementation
    '''
    def __init__(self) -> None:
        pass

    def __call__(self, my_array: np.ndarray):
        shift_vector = np.zeros((my_array.shape[0],my_array.shape[1]))
        return my_array, shift_vector


class ConstantShifter(AbstractShifter):
    '''
    Shifts by a constant value
    '''
    def __init__(self, shift_value: np.ndarray) -> None:
        self.__shift_value = shift_value

    def __call__(self, my_array: np.ndarray):
        shift_vector = np.empty((my_array.shape[0], my_array.shape[1],))
        assert my_array.shape[0] == self.__shift_value.size
        for i in range(0, my_array.shape[0]):
            shift_vector[i] = self.__shift_value[i]
        return my_array-shift_vector[:, :, None], shift_vector


class VectorShifter(AbstractShifter):
    '''
    Shifts by a user-input vector
    '''
    def __init__(self, shift_vector: np.ndarray) -> None:
        self.__shift_vector = shift_vector

    def __call__(self, my_array: np.ndarray):
        return my_array-self.__shift_vector[..., None], self.__shift_vector


class AverageShifter(AbstractShifter):
    '''
    Shifts by the average of a data matrix
    '''
    def __init__(self) -> None:
        pass

    def __call__(self, my_array: np.ndarray):
        shift_vector = np.mean(my_array, axis=2)
        return my_array-shift_vector[:, :, None], shift_vector


class FirstVecShifter(AbstractShifter):
    '''
    Shifts by the first vector of a data matrix
    '''
    def __init__(self) -> None:
        pass

    def __call__(self, my_array: np.ndarray):
        shift_vector = my_array[:, :, 0]
        return my_array[:, :, 1::]-shift_vector[:, :, None], shift_vector
