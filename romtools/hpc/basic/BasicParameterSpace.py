import numpy as np
import romtools.workflows.parameter_spaces

class BasicParameterSpace(romtools.workflows.parameter_spaces.ParameterSpace):
    def __init__(self):
        self._names = ['x', 'y']
        self._mins = [0.0, 0.0]
        self._maxes = [1.0, 1.0]

    def get_names(self):
        return list(self._names)

    def get_dimensionality(self):
        return len(self._names)

    def generate_samples(self, number_of_samples: int) -> np.ndarray:
        samples = np.random.uniform(low=self._mins, high=self._maxes, size=(number_of_samples, self.get_dimensionality()))
        return samples
