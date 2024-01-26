import numpy as np
from romtools.workflows.parameter_spaces import UniformParameter
from romtools.workflows.parameter_spaces import StringParameter
from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.parameter_spaces import ConstParameterSpace
from romtools.workflows.parameter_spaces import HeterogeneousParameterSpace


def test_uniform_parameter():
    np.random.seed(12)
    param = UniformParameter('p1', -1, 1)
    assert param.get_name() == 'p1'
    assert param.get_dimensionality() == 1
    s = param.generate_samples(3)
    assert s.shape == (3, 1)
    gold = [[-0.69167432],
            [ 0.48009939],
            [-0.47336997]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)


def test_string_parameter():
    param = StringParameter('p1', 'p1val')
    assert param.get_name() == 'p1'
    assert param.get_dimensionality() == 1
    s = param.generate_samples(3)
    assert s.shape == (3, 1)
    assert (s == [['p1val', 'p1val', 'p1val']]).all()


def test_uniform_param_space():
    np.random.seed(12)
    param_space = UniformParameterSpace(['p1', 'p2'], [-1, 0], [1, 3])
    assert param_space.get_names() == ['p1', 'p2']
    assert param_space.get_dimensionality() == 2
    s = param_space.generate_samples(4)
    assert s.shape == (4, 2)
    gold = [[-0.69167432, 0.04372489],
            [ 0.48009939, 2.75624102],
            [-0.47336997, 2.70214456],
            [ 0.06747879, 0.10026428]]
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
    np.random.seed(12)
    param1 = UniformParameter('p1', -1, 1)
    param2 = UniformParameter('p2', 0, 0)
    param3 = StringParameter('p3', 'p3val')
    param_space = HeterogeneousParameterSpace((param1, param2, param3))

    assert param_space.get_names() == ['p1', 'p2', 'p3']
    assert param_space.get_dimensionality() == 3
    s = param_space.generate_samples(4)
    assert s.shape == (4, 3)
    print(s)
    assert (s == [['-0.6916743152406553', '0.0', 'p3val'],
                  ['0.4800993930308095', '0.0', 'p3val'],
                  ['-0.47336996962973066', '0.0', 'p3val'],
                  ['0.06747878676059549', '0.0', 'p3val']]).all()
