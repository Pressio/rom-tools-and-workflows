import numpy as np
from romtools.workflows.parameter_spaces import UniformParameter
from romtools.workflows.parameter_spaces import StringParameter
from romtools.workflows.parameter_spaces import EmptyParameterSpace
from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.parameter_spaces import ConstParameterSpace
from romtools.workflows.parameter_spaces import HeterogeneousParameterSpace

from romtools.workflows.parameter_spaces import monte_carlo_sample
from romtools.workflows.parameter_spaces import latin_hypercube_sample


def test_uniform_parameter():
    param = UniformParameter('p1', -1, 1)
    assert param.get_name() == 'p1'
    assert param.get_dimensionality() == 1

    germ = np.array([[0.1], [0.5], [0.7]])
    s = param.generate_samples(germ)
    assert s.shape == germ.shape
    gold = [[-0.8],
            [ 0.0],
            [ 0.4]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)


def test_vector_parameter():
    param = UniformParameter('p1', [-1, 0], [1, 3])
    assert param.get_name() == 'p1'
    assert param.get_dimensionality() == 2

    germ = np.array([[0.1, 0.2], [0.5, 0.6], [0.7, 0.5]])
    s = param.generate_samples(germ)
    assert s.shape == (3, 2)
    gold = [[-0.8, 0.6],
            [ 0.0, 1.8],
            [ 0.4, 1.5]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)


def test_string_parameter():
    param = StringParameter('p1', 'p1val')
    assert param.get_name() == 'p1'
    assert param.get_dimensionality() == 1

    germ = np.array([[0.1], [0.5], [0.7]])
    s = param.generate_samples(germ)
    assert s.shape == germ.shape
    assert (s == [['p1val', 'p1val', 'p1val']]).all()


def test_empty_param_space():
    param_space = EmptyParameterSpace()
    assert param_space.get_names() == []
    assert param_space.get_dimensionality() == 0
    germ = np.random.uniform(size=(3, 0))
    s = param_space.generate_samples(germ)
    print(s.shape)
    assert s.shape == (3, 0)


def test_uniform_param_space():
    param_space = UniformParameterSpace(['p1', 'p2'], [-1, 0], [1, 3])
    assert param_space.get_names() == ['p1', 'p2']
    assert param_space.get_dimensionality() == 2

    germ = np.array([[0.1, 0.2], [0.5, 0.6], [0.7, 0.5]])
    s = param_space.generate_samples(germ)
    assert s.shape == (3, 2)
    gold = [[-0.8, 0.6],
            [ 0.0, 1.8],
            [ 0.4, 1.5]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)


def test_const_param_space():
    param_space = ConstParameterSpace(['p1', 'p2', 'p3'], [1, 3, 'p3val'])
    assert param_space.get_names() == ['p1', 'p2', 'p3']
    assert param_space.get_dimensionality() == 3
    germ = np.array([[0.1, 0.2, 0.3],
                     [0.4, 0.5, 0.6],
                     [0.7, 0.8, 0.9],
                     [0.0, 1.0, 0.5]])
    s = param_space.generate_samples(germ)
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

    germ = np.array([[0.1, 0.2, 0.3],
                     [0.4, 0.5, 0.6],
                     [0.7, 0.8, 0.9],
                     [0.0, 1.0, 0.5]])
    s = param_space.generate_samples(germ)
    assert s.shape == (4, 3)
    np.testing.assert_allclose(s[:, 0].astype(float), [-0.8, -0.2, 0.4, -1.0],
                               rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(s[:, 1].astype(float), [0.2, 0.5, 0.8, 1.0],
                               rtol=1e-5, atol=1e-8)
    assert (s[:, 2] == ['p3val', 'p3val', 'p3val', 'p3val']).all()


def test_monte_carlo_sample():
    param_space = UniformParameterSpace(['p1', 'p2'], [-1, 0], [1, 3])
    s = monte_carlo_sample(param_space, 4, seed=12)
    assert s.shape == (4, 2)

    gold = [[-0.69167432, 2.22014909],
            [-0.47336997, 1.60121818],
            [-0.97085008, 2.75624102],
            [ 0.80142971, 0.10026428]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)


def test_latin_hypercube_sample():
    np.random.seed(12)
    param_space = UniformParameterSpace(['p1', 'p2'], [-1, 0], [1, 3])
    s = latin_hypercube_sample(param_space, 4, seed=12)
    assert s.shape == (4, 2)

    gold = [[-0.12541223, 0.78993529],
            [ 0.40533981, 2.86553144],
            [ 0.82505538, 2.07709407],
            [-0.83522287, 0.66369046]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)
