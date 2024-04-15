import numpy as np
from scipy.stats import norm

from romtools.workflows.parameters import UniformParameter
from romtools.workflows.parameters import StringParameter
from romtools.workflows.parameters import GaussianParameter
from romtools.workflows.parameters import TriangularParameter
from romtools.workflows.parameters import ScipyDistributionParameter


def test_uniform_parameter():
    param = UniformParameter('p1', -1, 1)
    assert param.get_name() == 'p1'
    assert param.get_dimensionality() == 1

    germ = np.array([[0.1], [0.5], [0.7]])
    s = param.scale_samples(germ)
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
    s = param.scale_samples(germ)
    assert s.shape == (3, 2)
    gold = [[-0.8, 0.6], [0.0, 1.8], [0.4, 1.5]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)


def test_string_parameter():
    param = StringParameter('p1', 'p1val')
    assert param.get_name() == 'p1'
    assert param.get_dimensionality() == 1

    germ = np.array([[0.1], [0.5], [0.7]])
    s = param.scale_samples(germ)
    assert s.shape == germ.shape
    assert (s == [['p1val', 'p1val', 'p1val']]).all()


def test_gaussian_parameter():
    param = GaussianParameter('p1', 0, 1)
    assert param.get_name() == 'p1'
    assert param.get_dimensionality() == 1

    germ = np.array([[0.1], [0.5], [0.7]])
    s = param.scale_samples(germ)
    assert s.shape == germ.shape
    gold = [[-1.281552], [0.0], [0.524401]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)


def test_multidimensional_gaussian_parameter():
    param = GaussianParameter('p1', [0, 1, 0], [1, 1, 2])
    assert param.get_name() == 'p1'
    assert param.get_dimensionality() == 3

    germ = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7]])
    s = param.scale_samples(germ)
    assert s.shape == germ.shape
    gold = [[-1.281552, -0.281552, -2.563104],
            [0.0, 1.0, 0.0],
            [0.524401, 1.524401, 1.048802]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)


def test_triangular_parameter():
    param = TriangularParameter('p1', 0, 1, 2)
    assert param.get_name() == 'p1'
    assert param.get_dimensionality() == 1

    germ = np.array([[0.1], [0.5], [0.7]])
    s = param.scale_samples(germ)
    assert s.shape == germ.shape
    gold = [[0.447214], [1.0], [1.225403]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)


def test_multidimensional_triangular_parameter():
    param = TriangularParameter('p1', [0, 1, 0], [1, 2, 2], [2, 3, 4])
    assert param.get_name() == 'p1'
    assert param.get_dimensionality() == 3

    germ = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7]])
    s = param.scale_samples(germ)
    assert s.shape == germ.shape
    gold = [[0.447214, 1.447214, 0.894427],
            [1.0, 2.0, 2.0],
            [1.225403, 2.225403, 2.450807]]
    np.testing.assert_allclose(s, gold, rtol=1e-5, atol=1e-8)


def test_scipy_gaussian_parameter():
    param = ScipyDistributionParameter('p1', norm, loc=0, scale=1)
    assert param.get_name() == 'p1'
    assert param.get_dimensionality() == 1

    gold_param = GaussianParameter('p1', 0, 1)
    germ = np.random.random(size=(4, 1))
    s = param.scale_samples(germ)
    assert s.shape == germ.shape
    np.testing.assert_allclose(s, gold_param.scale_samples(germ), rtol=1e-5, atol=1e-8)


def test_scipy_multidimensional_gaussian_parameter():
    param = ScipyDistributionParameter('p1', norm, loc=[0, 1, 0], scale=[1, 1, 2])
    assert param.get_name() == 'p1'
    assert param.get_dimensionality() == 3

    gold_param = GaussianParameter('p1', [0, 1, 0], [1, 1, 2])
    germ = np.random.random(size=(4, param.get_dimensionality()))
    s = param.scale_samples(germ)
    assert s.shape == germ.shape
    np.testing.assert_allclose(s, gold_param.scale_samples(germ), rtol=1e-5, atol=1e-8)
