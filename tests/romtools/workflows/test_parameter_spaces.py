import numpy as np
from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.parameter_spaces import ConstParamSpace


def test_uniform_param_space():
    np.random.seed(12)
    param_space = UniformParameterSpace(['p1', 'p2'], [-1, 0], [1, 3])
    assert param_space.get_names() == ['p1', 'p2']
    assert param_space.get_dimensionality() == 2
    s = param_space.generate_samples(4)
    assert s.shape == (4, 2)
    gold = [[-0.69167432, 2.22014909],
            [-0.47336997, 1.60121818],
            [-0.97085008, 2.75624102],
            [0.80142971,  0.10026428]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)


def test_const_param_space():
    param_space = ConstParamSpace(['p1', 'p2', 'p3'], [1, 3, 'p3val'])
    assert param_space.get_names() == ['p1', 'p2', 'p3']
    assert param_space.get_dimensionality() == 3
    s = param_space.generate_samples(4)
    assert s.shape == (4, 3)
    assert (s == [['1', '3', 'p3val'],
                  ['1', '3', 'p3val'],
                  ['1', '3', 'p3val'],
                  ['1', '3', 'p3val']]).all()
