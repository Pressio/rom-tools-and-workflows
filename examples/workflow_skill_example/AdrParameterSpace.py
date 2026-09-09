import numpy as np

import romtools.workflows.parameter_spaces


class AdrParameterSpace(romtools.workflows.parameter_spaces.ParameterSpace):
    def __init__(self):
        self._names = ['c', 'nu']
        self._c_min = 0.5
        self._c_max = 5.0
        self._nu_log_min = np.log10(1e-3)
        self._nu_log_max = np.log10(1e-1)

    def get_names(self):
        return list(self._names)

    def get_dimensionality(self):
        return len(self._names)

    def generate_samples(self, number_of_samples: int) -> np.ndarray:
        c_samples = np.random.uniform(self._c_min, self._c_max, size=number_of_samples)
        nu_samples = 10.0 ** np.random.uniform(self._nu_log_min, self._nu_log_max, size=number_of_samples)
        return np.column_stack([c_samples, nu_samples])
