import numpy as np

from romtools.workflows.parameters import UniformParameter
from romtools.workflows.parameters import StringParameter

from romtools.workflows.parameter_spaces import EmptyParameterSpace
from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.parameter_spaces import GaussianParameterSpace
from romtools.workflows.parameter_spaces import ConstParameterSpace
from romtools.workflows.parameter_spaces import HeterogeneousParameterSpace

from romtools.workflows.sampling_methods import MonteCarloSampler


def test_empty_param_space():
    param_space = EmptyParameterSpace()
    assert param_space.get_names() == []
    assert param_space.get_dimensionality() == 0
    s = param_space.generate_samples(3)
    assert s.shape == (3, 0)


def test_uniform_param_space():
    param_space = UniformParameterSpace(['p1', 'p2'], [-1, 0], [1, 3],
                                        sampler=MonteCarloSampler)
    assert param_space.get_names() == ['p1', 'p2']
    assert param_space.get_dimensionality() == 2

    s = param_space.generate_samples(3, seed=1)
    assert s.shape == (3, 2)
    gold = [[-0.16595599, 2.16097348],
            [-0.99977125, 0.90699772],
            [-0.70648822, 0.27701578]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)


def test_gaussian_param_space():
    param_space = GaussianParameterSpace(['p1', 'p2'], [-1, 0], [1, 2],
                                         sampler=MonteCarloSampler)
    assert param_space.get_names() == ['p1', 'p2']
    assert param_space.get_dimensionality() == 2

    s = param_space.generate_samples(3, seed=1)
    assert s.shape == (3, 2)
    gold = [[-1.209518,  1.167611],
            [-4.684948, -1.035407],
            [-2.050449, -2.652982]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)


def test_const_param_space():
    param_space = ConstParameterSpace(['p1', 'p2', 'p3'], [1, 3, 'p3val'])
    assert param_space.get_names() == ['p1', 'p2', 'p3']
    assert param_space.get_dimensionality() == 3

    s = param_space.generate_samples(4)
    assert s.shape == (4, 3)
    assert (s == [['1', '3', 'p3val'],
                  ['1', '3', 'p3val'],
                  ['1', '3', 'p3val'],
                  ['1', '3', 'p3val']]).all()


def test_hetero_param_space():
    param1 = UniformParameter('p1', -1, 1)
    param2 = UniformParameter('p2', 0, 1)
    param3 = StringParameter('p3', 'p3val')
    param_space = HeterogeneousParameterSpace((param1, param2, param3))

    assert param_space.get_names() == ['p1', 'p2', 'p3']
    assert param_space.get_dimensionality() == 3

    s = param_space.generate_samples(4, seed=1)
    assert s.shape == (4, 3)

    np.testing.assert_allclose(s[:, 0].astype(float),
                               [-0.16595599, -0.39533485, -0.62747958, 0.07763347],
                               rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(s[:, 1].astype(float),
                               [0.72032449, 0.14675589, 0.34556073, 0.41919451],
                               rtol=1e-5, atol=1e-8)
    assert (s[:, 2] == ['p3val', 'p3val', 'p3val', 'p3val']).all()



