import numpy as np

from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.sampling_methods import MonteCarloSampler
from romtools.workflows.sampling_methods import LatinHypercubeSampler


def test_monte_carlo_sample():
    param_space = UniformParameterSpace(['p1', 'p2'], [-1, 0], [1, 3],
                                        sampler=MonteCarloSampler)
    s = param_space.generate_samples(4, seed=12)
    assert s.shape == (4, 2)

    gold = [[-0.69167432, 2.22014909],
            [-0.47336997, 1.60121818],
            [-0.97085008, 2.75624102],
            [ 0.80142971, 0.10026428]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)


def test_latin_hypercube_sample():
    np.random.seed(12)
    param_space = UniformParameterSpace(['p1', 'p2'], [-1, 0], [1, 3],
                                        sampler=LatinHypercubeSampler)
    s = param_space.generate_samples(4, seed=12)
    assert s.shape == (4, 2)

    gold = [[-0.12541223, 0.78993529],
            [ 0.40533981, 2.86553144],
            [ 0.82505538, 2.07709407],
            [-0.83522287, 0.66369046]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)