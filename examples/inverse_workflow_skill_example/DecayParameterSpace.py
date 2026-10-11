import numpy as np

from romtools.workflows.parameter_spaces import ParameterSpace


class DecayParameterSpace(ParameterSpace):
    lower_bounds = np.array([0.5, 0.1])
    upper_bounds = np.array([3.0, 1.5])

    def get_names(self):
        return ["amplitude", "rate"]

    def get_dimensionality(self):
        return 2

    def generate_samples(self, number_of_samples: int, seed=None) -> np.ndarray:
        # EKI seeds NumPy globally and calls without seed; EGO passes seed.
        rng = np.random if seed is None else np.random.default_rng(seed)
        return rng.uniform(self.lower_bounds, self.upper_bounds, size=(number_of_samples, 2))
