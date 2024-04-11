import abc
import numpy as np
from scipy.stats import qmc

class Parameter(abc.ABC):
    '''Abstract implementation'''

    @abc.abstractmethod
    def get_name(self) -> str:
        '''
        Returns parameter name
        '''

    @abc.abstractmethod
    def get_dimensionality(self) -> int:
        '''
        Returns dimensionality of parameter for vector quantities.
        Returns 1 for scalar parameters
        '''

    @abc.abstractmethod
    def scale_samples(self, uniform_dist_samples) -> np.array:
        '''
        Generates samples from the desired distribution given a set of samples
        from a uniform distribution on (0,1)

        uniform_dist_samples should be of shape
        (number_of_samples, self.get_dimensionality())

        Returns np.array of the same shape
        '''

##########################################
# Concrete Parameter Classes
##########################################


class UniformParameter(Parameter):
    '''
    Uniformly distributed floating point
    '''
    def __init__(self, parameter_name: str,
                 lower_bound: float = 0,
                 upper_bound: float = 1):
        self._parameter_name = parameter_name

        try:
            assert len(lower_bound) == len(upper_bound)
            self._dimension = len(lower_bound)
        except TypeError:
            self._dimension = 1
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def get_name(self) -> str:
        return self._parameter_name

    def get_dimensionality(self) -> int:
        return self._dimension

    def scale_samples(self, uniform_dist_samples: np.array) -> np.array:
        assert uniform_dist_samples.shape[1] == self.get_dimensionality()
        return qmc.scale(uniform_dist_samples,
                         self._lower_bound,
                         self._upper_bound)


class StringParameter(Parameter):
    '''
    Constant string-valued parameter
    '''
    def __init__(self, parameter_name: str, value):
        self._parameter_name = parameter_name
        self._parameter_value = value

    def get_name(self) -> str:
        return self._parameter_name

    def get_dimensionality(self) -> int:
        return 1

    def scale_samples(self, uniform_dist_samples: np.array) -> np.array:
        assert uniform_dist_samples.shape[1] == self.get_dimensionality()
        number_of_samples = uniform_dist_samples.shape[0]
        return np.array([[self._parameter_value]] * number_of_samples)
