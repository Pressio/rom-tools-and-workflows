import abc
import numpy as np
from scipy.stats import qmc
from scipy.stats import norm
from scipy.stats import triang
from scipy.stats import rv_continuous

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


class GaussianParameter(Parameter):
    '''
    Normally distributed parameter
    '''
    def __init__(self, parameter_name: str,
                 mean: float = 0,
                 std: float = 1):
        self._parameter_name = parameter_name
        try:
            assert len(mean) == len(std)
            self._dimension = len(mean)
        except TypeError:
            self._dimension = 1
        self._mean = mean
        self._std = std

    def get_name(self) -> str:
        return self._parameter_name

    def get_dimensionality(self) -> int:
        return self._dimension

    def scale_samples(self, uniform_dist_samples: np.array) -> np.array:
        assert uniform_dist_samples.shape[1] == self.get_dimensionality()
        return norm.ppf(uniform_dist_samples, loc=self._mean, scale=self._std)


class TriangularParameter(Parameter):
    '''
    Random parameter with a triangular distribution
    '''
    def __init__(self, parameter_name: str,
                 lower_bound: float = -1,
                 peak: float = 0,
                 upper_bound: float = 1):
        self._parameter_name = parameter_name
        try:
            assert len(lower_bound) == len(peak)
            assert len(lower_bound) == len(upper_bound)
            self._dimension = len(lower_bound)
        except TypeError:
            self._dimension = 1
        self._loc = lower_bound
        self._scale = np.array(upper_bound) - np.array(lower_bound)
        self._c = (np.array(peak) - np.array(lower_bound))/(self._scale)

    def get_name(self) -> str:
        return self._parameter_name

    def get_dimensionality(self) -> int:
        return self._dimension

    def scale_samples(self, uniform_dist_samples: np.array) -> np.array:
        assert uniform_dist_samples.shape[1] == self.get_dimensionality()
        return triang.ppf(uniform_dist_samples, loc=self._loc, scale=self._scale, c=self._c)


class ScipyDistributionParameter(Parameter):
    '''
    Random parameter with distribution described by a scipy.stats.rv_continuous object
    '''

    def __init__(self, parameter_name: str, distribution: rv_continuous, **kwargs):
        self._parameter_name = parameter_name
        self._dist_obj = distribution
        self._kwargs = kwargs

        try:
            self._dimension = len(kwargs['loc'])
        except TypeError:
            self._dimension = 1

    def get_name(self) -> str:
        return self._parameter_name

    def get_dimensionality(self) -> int:
        return self._dimension

    def scale_samples(self, uniform_dist_samples: np.array) -> np.array:
        assert uniform_dist_samples.shape[1] == self.get_dimensionality()
        return self._dist_obj.ppf(uniform_dist_samples, **self._kwargs)
