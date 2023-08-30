"""
Model reduction is often focused on parameterized PDEs, where $\\boldsymbol \\mu$ is the parameter set.
The ParameterSpace class encapsulates the notion of the parameter space.
"""
import abc
import numpy as np
"""The abstract parameter space"""
class AbstractParameterSpace(abc.ABC):

    """ Abstract implementation"""
    @abc.abstractmethod
    def getNames() -> list:
        """
        return a list of parameter names
        # e.g., ['sigma','beta',...]
        """
        pass

    @abc.abstractmethod
    def getDimensionality() -> int:
        """
        returns an integer for the size
        of the parameter domain
        """
        pass

    @abc.abstractmethod
    def generateSamples(self,number_of_samples):
        """
        generates and returns number of parameter samples
        """
        pass


class UniformParameterSpace(AbstractParameterSpace):
    """Concrete implementation for a uniform parameter space with random sampling"""

    def __init__(self,parameter_names,lower_bounds,upper_bounds):
        self.__parameter_names = parameter_names
        self.__lower_bounds = lower_bounds
        self.__upper_bounds = upper_bounds
        self.__n_params = len(self.__lower_bounds)

    def getNames():
        return self.__parameter_names

    def getDimensionality(self):
        return self.__n_params

    def generateSamples(self,number_of_samples):
        samples = np.random.uniform(self.__lower_bounds,self.__upper_bounds,\
                                    size=(number_of_samples,self.__n_params))
        return samples
