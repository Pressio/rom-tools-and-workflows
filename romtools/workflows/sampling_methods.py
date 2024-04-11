from typing import Protocol
import numpy as np
from scipy.stats import qmc

class Sampler(Protocol):
    '''
    Generate UIID samples

    Returns np.ndarray of shape (number_of_samples, dimensionality)
    '''
    def __call__(self, number_of_samples: int, dimensionality: int=1, seed=None) -> np.ndarray:
        pass


##########################################
# Sampling Methods
##########################################

def MonteCarloSampler(number_of_samples: int, dimensionality: int=1, seed=None) -> np.ndarray:
    '''
    Generate UIID Monte Carlo samples
    '''
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(size=(number_of_samples,
                                   dimensionality))


def LatinHypercubeSampler(number_of_samples: int, dimensionality: int=1, seed=None) -> np.ndarray:
    '''
    Generate UIID LHS samples
    '''
    sampler = qmc.LatinHypercube(dimensionality, seed=seed)
    return sampler.random(n=number_of_samples)
