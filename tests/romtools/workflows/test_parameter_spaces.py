import numpy as np
import pytest

from romtools.workflows.parameters import UniformParameter
from romtools.workflows.parameters import StringParameter

from romtools.workflows.parameter_spaces import ParameterSpace, UnionParameterSpace
from romtools.workflows.parameter_spaces import EmptyParameterSpace
from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.parameter_spaces import GaussianParameterSpace
from romtools.workflows.parameter_spaces import MultivariateGaussianParameterSpace
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


def test_multivariate_gaussian_param_space():
    param_space = MultivariateGaussianParameterSpace(
        ['p1', 'p2'],
        means=[-1, 0.5],
        covariance=[[1.0, 0.2], [0.2, 4.0]],
        sampler=MonteCarloSampler
    )
    assert param_space.get_names() == ['p1', 'p2']
    assert param_space.get_dimensionality() == 2

    s = param_space.generate_samples(3, seed=1)
    assert s.shape == (3, 2)
    gold = [[-1.209518,  1.619855],
            [-4.684948, -1.267207],
            [-2.050449, -2.349773]]
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




def _region(lower, upper, names=None, sampler=MonteCarloSampler):
    lower, upper = np.atleast_1d(lower), np.atleast_1d(upper)
    names = [f'x{i}' for i in range(len(lower))] if names is None else names
    return UniformParameterSpace(names, lower, upper, sampler)


def _check_probability(mask, probability):
    # Six binomial standard errors plus one count for rounding, at large n.
    count = len(mask)
    tolerance = 6 * np.sqrt(probability * (1 - probability) / count) + 1 / count
    assert abs(np.mean(mask) - probability) <= tolerance


def test_uniform_space_bound_copies_and_vector_coordinates():
    space = _region([-1, 0], [1, 3])
    lower, upper = space.get_bounds()
    np.testing.assert_array_equal(lower, [-1, 0])
    np.testing.assert_array_equal(upper, [1, 3])
    lower[:] = -99
    upper[:] = 99
    np.testing.assert_array_equal(space.get_bounds(), [[-1, 0], [1, 3]])
    vector = UniformParameterSpace(['v', 'x'], [[0, 1], 2], [[2, 3], 4], MonteCarloSampler)
    np.testing.assert_array_equal(vector.get_bounds(), [[0, 1, 2], [2, 3, 4]])
    empty = UniformParameterSpace([], [], [], MonteCarloSampler)
    assert all(bound.shape == (0,) for bound in empty.get_bounds())


def test_union_interface_shapes_and_seed():
    union = UnionParameterSpace([_region([0, 0], [2, 2]), _region([1, 1], [3, 3])])
    assert isinstance(union, ParameterSpace)
    assert union.get_names() == ['x0', 'x1']
    assert union.get_dimensionality() == 2
    union.get_names().append('ignored')
    assert union.get_names() == ['x0', 'x1']
    samples = union.generate_samples(1000, seed=42)
    assert samples.shape == (1000, 2) and samples.dtype.kind == 'f'
    assert union.contains(samples).all()
    np.testing.assert_array_equal(samples, union.generate_samples(1000, seed=42))
    assert not np.array_equal(samples, union.generate_samples(1000, seed=43))
    assert union.generate_samples(0, seed=42).shape == (0, 2)
    assert union.generate_samples(1, seed=42).shape == (1, 2)
    assert union.contains(np.empty((0, 2))).shape == (0,)
    assert union.generate_samples(np.int64(2), seed=1).shape == (2, 2)


def test_union_membership_boundaries_holes_and_nonfinite_points():
    union = UnionParameterSpace([_region([0, 0], [1, 1]), _region([2, 2], [3, 3])])
    assert union.contains([0, 0]) is True
    assert union.contains([3, 3]) is True
    assert union.contains([1.5, 1.5]) is False
    np.testing.assert_array_equal(union.contains([[1, 1], [2, 2], [0, 3], [np.nan, 0], [np.inf, 0]]),
                                  [True, True, False, False, False])
    touching = UnionParameterSpace([_region(0, 1), _region(1, 2)])
    assert touching.contains([1]) is True


def test_union_preserves_global_rng_and_ignores_region_samplers():
    def forbidden_sampler(*args):
        raise AssertionError('union must not call a region sampler')
    union = UnionParameterSpace([_region(0, 1, sampler=forbidden_sampler)])
    before = np.random.get_state()
    union.generate_samples(100, seed=2)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[0] == after[0] and before[2:] == after[2:]


def test_union_snapshots_vector_region_bounds():
    lower, upper = [[0, 0]], [[1, 1]]
    region = UniformParameterSpace(['v'], lower, upper, MonteCarloSampler)
    union = UnionParameterSpace([region])
    lower[0][0], upper[0][0] = 10, 11
    assert region.get_bounds()[0][0] == 10
    assert union.get_dimensionality() == 2
    samples = union.generate_samples(30, seed=4)
    assert union.contains(samples).all()
    assert np.all((samples >= 0) & (samples <= 1))


@pytest.mark.parametrize('lower,upper', [([0], [3]), ([0, -2], [3, 2])])
def test_union_single_region_monte_carlo_means(lower, upper):
    union = UnionParameterSpace([_region(lower, upper)])
    samples = union.generate_samples(100000, seed=13)
    assert union.contains(samples).all()
    lower, upper = np.asarray(lower), np.asarray(upper)
    expected = (lower + upper) / 2
    variance = (upper - lower)**2 / 12
    assert np.all(np.abs(samples.mean(axis=0) - expected) <= 6 * np.sqrt(variance / len(samples)))


def test_union_disjoint_region_probabilities_and_mean():
    # Lengths 1 and 3: the right component has probability 3/4.
    union = UnionParameterSpace([_region(0, 1), _region(2, 5)])
    x = union.generate_samples(100000, seed=11)[:, 0]
    _check_probability(x >= 2, .75)
    mean = .25 * .5 + .75 * 3.5
    second_moment = .25 / 3 + .75 * 13
    assert abs(x.mean() - mean) <= 6 * np.sqrt((second_moment - mean**2) / len(x))
    assert union.contains(x[:, None]).all()


def test_union_partial_overlap_one_dimension_statistics_and_independence():
    union = UnionParameterSpace([_region(0, 2), _region(1, 3)])
    x = union.generate_samples(100000, seed=12)[:, 0]
    _check_probability((x >= 1) & (x <= 2), 1/3)
    assert abs(x.mean() - 1.5) <= 6 * np.sqrt(.75 / len(x))
    # Disjoint pairs give independent Bernoulli observations. This also catches
    # grouping output by proposal region instead of preserving draw order.
    _check_probability((x[::2] < 1) & (x[1::2] < 1), 1/9)


def test_union_partial_overlap_two_dimension_statistics():
    # Two area-4 squares overlap in area 1: union area is 7.
    union = UnionParameterSpace([_region([0, 0], [2, 2]), _region([1, 1], [3, 3])])
    samples = union.generate_samples(100000, seed=14)
    intersection = np.all((samples >= 1) & (samples <= 2), axis=1)
    _check_probability(intersection, 1/7)
    assert np.all(np.abs(samples.mean(axis=0) - 1.5) <= 6 * np.sqrt((55/84) / len(samples)))
    assert union.contains(samples).all()


@pytest.mark.parametrize('inner', [(1, 2), (0, 4)])
def test_union_contained_and_identical_regions(inner):
    union = UnionParameterSpace([_region(0, 4), _region(*inner)])
    x = union.generate_samples(100000, seed=15)[:, 0]
    _check_probability((x >= 1) & (x < 2), .25)
    assert abs(x.mean() - 2) <= 6 * np.sqrt((4/3) / len(x))


def test_union_three_overlapping_regions_and_multiplicity():
    union = UnionParameterSpace([_region(0, 2), _region(1, 3), _region(1.5, 4)])
    x = union.generate_samples(100000, seed=16)[:, 0]
    for left, right in [(0, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 4)]:
        _check_probability((x >= left) & (x < right), (right - left) / 4)


@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_union_volume_products_do_not_underflow_or_overflow(scale):
    union = UnionParameterSpace([_region([0, 0], [scale, scale]),
                                 _region([2*scale, 2*scale], [4*scale, 4*scale])])
    samples = union.generate_samples(100000, seed=17)
    _check_probability(samples[:, 0] < scale, .2)
    assert union.contains(samples).all()


def test_union_high_overlap_still_returns_requested_count():
    union = UnionParameterSpace([_region(0, 1) for _ in range(30)])
    samples = union.generate_samples(3000, seed=18)
    assert samples.shape == (3000, 1) and union.contains(samples).all()
    assert abs(samples.mean() - .5) <= 6 * np.sqrt(1 / (12 * len(samples)))


@pytest.mark.parametrize('regions,error', [
    ([], ValueError), ([EmptyParameterSpace()], TypeError), ([object()], TypeError),
    ([_region(0, 1), GaussianParameterSpace(['x0'], [0], [1], MonteCarloSampler)], TypeError),
    ([_region(0, 0)], ValueError), ([_region(1, 0)], ValueError),
    ([_region(np.nan, 1)], ValueError), ([_region(0, np.inf)], ValueError),
    ([_region(-np.inf, 1)], ValueError), ([_region(0, 1j)], ValueError),
    ([_region('0', '1')], ValueError),
    ([_region(-1e308, 1e308)], ValueError),
    ([UniformParameterSpace(['v'], [[[0, 0], [0, 0]]], [[[1, 1], [1, 1]]], MonteCarloSampler)], ValueError),
    ([UniformParameterSpace([], [], [], MonteCarloSampler)], ValueError),
    ([_region([0, 0], [1, 1], ['a', 'b']), _region([0, 0], [1, 1], ['b', 'a'])], ValueError),
    ([_region(0, 1, ['a']), _region(0, 1, ['b'])], ValueError),
    ([_region(0, 1), UniformParameterSpace(['x0'], [[0, 0]], [[1, 1]], MonteCarloSampler)], ValueError),
    ([_region([0, 0], [1e-200, 1e-200]), _region([0, 0], [1e200, 1e200])], ValueError),
])
def test_union_invalid_regions(regions, error):
    with pytest.raises(error):
        UnionParameterSpace(regions)


@pytest.mark.parametrize('count,error', [(-1, ValueError), (True, TypeError), (2.0, TypeError),
                                       (None, TypeError), ('2', TypeError)])
def test_union_invalid_sample_counts(count, error):
    with pytest.raises(error, match='nonnegative integer'):
        UnionParameterSpace([_region(0, 1)]).generate_samples(count)


@pytest.mark.parametrize('samples,error', [(1, ValueError), ([[1, 2]], ValueError),
                                          (np.zeros((1, 1, 1)), ValueError),
                                          (['x'], TypeError), ([1j], TypeError)])
def test_union_invalid_membership_inputs(samples, error):
    with pytest.raises(error):
        UnionParameterSpace([_region(0, 1)]).contains(samples)


def test_union_accepts_generator_of_regions():
    union = UnionParameterSpace(_region(i, i + 1) for i in range(3))
    assert union.contains(union.generate_samples(100, seed=9)).all()
