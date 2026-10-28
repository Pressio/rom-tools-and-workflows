"""Geometry and reproducibility without assumptions about parameter bounds."""

import numpy as np
import pytest

from romtools.workflows.parameter_spaces import (
    ConstParameterSpace, EmptyParameterSpace, GaussianParameterSpace,
    HeterogeneousParameterSpace, MonteCarloSampler, UniformParameterSpace,
)
from romtools.workflows.parameters import StringParameter, UniformParameter
from romtools.workflows.space_filling import select_space_filling_samples as select


class FixedSpace:
    def __init__(self, samples):
        self.samples = np.asarray(samples)

    def get_names(self):
        return [f"p{i}" for i in range(self.samples.shape[1])]

    def get_dimensionality(self):
        return self.samples.shape[1]

    def generate_samples(self, count, seed=None):
        return self.samples.copy()


class LegacySpace(FixedSpace):
    def generate_samples(self, count):
        return np.random.uniform(size=(count, self.get_dimensionality()))


def test_known_one_dimensional_maximin():
    space = FixedSpace([[0], [0.25], [0.5], [0.75], [1]])
    result = select(space, [[0]], 3)
    np.testing.assert_array_equal(result, [[1], [0.5], [0.25]])


def test_two_dimensional_selection_matches_brute_force():
    candidates = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]])
    existing = np.array([[0.1, 0.1], [0.8, 0.7]])
    expected = []
    references = existing.copy()
    remaining = list(range(len(candidates)))
    for _ in range(3):
        distances = np.linalg.norm(candidates[remaining, None] - references, axis=2).min(axis=1)
        index = remaining.pop(int(np.argmax(distances)))
        expected.append(candidates[index])
        references = np.vstack([references, candidates[index]])
    np.testing.assert_array_equal(select(FixedSpace(candidates), existing, 3), expected)


def test_incremental_selection_uses_each_new_point():
    result = select(FixedSpace([[0.1], [0.9], [1.0]]), [[0]], 2)
    # Selecting the two largest distances to existing samples would pick 1,.9.
    np.testing.assert_array_equal(result, [[1.0], [0.1]])


def test_irregular_domain_samples_are_never_projected_to_a_box():
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    candidates = np.c_[np.cos(angles), np.sin(angles)]
    result = select(FixedSpace(candidates), None, 7, random_seed=7)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1)
    assert all(any(np.array_equal(row, candidate) for candidate in candidates) for row in result)


def test_custom_metric_changes_geometry_and_preserves_representations():
    candidates = np.array([["red", "0"], ["blue", "1"], ["red", "2"]])
    metric = lambda a, b: float(a[0] != b[0])
    result = select(FixedSpace(candidates), [["red", "0"]], 2, distance_metric=metric)
    np.testing.assert_array_equal(result, [["blue", "1"]])
    with pytest.raises(ValueError, match="numeric"):
        select(FixedSpace(candidates), None, 2)


def test_custom_euclidean_path_matches_vectorized_path():
    space = FixedSpace(np.random.default_rng(1).normal(size=(30, 3)))
    existing = np.ones((2, 3))
    np.testing.assert_array_equal(
        select(space, existing, 8, random_seed=7),
        select(space, existing, 8, distance_metric=lambda a, b: np.linalg.norm(a-b), random_seed=7),
    )


@pytest.mark.parametrize("space", [
    UniformParameterSpace(["x"], [0], [1], MonteCarloSampler),
    GaussianParameterSpace(["x"], [0], [1], MonteCarloSampler),
    LegacySpace([[0]]),
])
def test_reproducible_and_preserves_global_numpy_state(space):
    np.random.seed(41)
    state = np.random.get_state()
    first = select(space, None, 5, random_seed=12)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    np.testing.assert_array_equal(first, select(space, None, 5, random_seed=12))
    assert not np.array_equal(first, select(space, None, 5, random_seed=13))


def test_duplicates_and_candidate_shortfall():
    space = FixedSpace([[0], [0], [1], [1]])
    np.testing.assert_array_equal(select(space, [[0]], 10, candidate_pool_size=2), [[1]])
    assert select(space, [[0], [1]], 10).shape == (0, 1)
    assert select(FixedSpace(np.empty((0, 1))), None, 5).shape == (0, 1)


def test_degenerate_spaces_and_heterogeneous_builtin_space():
    empty = EmptyParameterSpace()
    assert select(empty, None, 5).shape == (1, 0)
    assert select(empty, np.empty((1, 0)), 5).shape == (0, 0)
    constant = ConstParameterSpace(["x"], ["fixed"])
    assert select(constant, None, 5, distance_metric=lambda a, b: float(a[0] != b[0])).shape == (1, 1)
    heterogeneous = HeterogeneousParameterSpace([
        UniformParameter("x", 0, 1), StringParameter("category", "fixed"),
    ])
    selected = select(heterogeneous, None, 3,
                      distance_metric=lambda a, b: abs(float(a[0])-float(b[0])))
    assert np.all(selected[:, 1] == "fixed")


@pytest.mark.parametrize("value", [np.nan, np.inf, -1, [1], True, 1j])
def test_invalid_distances_fail(value):
    with pytest.raises(ValueError, match="distance"):
        select(FixedSpace([[0], [1]]), None, 2, distance_metric=lambda a, b: value)


def test_sampler_type_error_is_not_retried():
    class BrokenSpace(FixedSpace):
        calls = 0

        def generate_samples(self, count, seed=None):
            self.calls += 1
            raise TypeError("sampler bug")

    space = BrokenSpace([[0]])
    with pytest.raises(TypeError, match="sampler bug"):
        select(space, None, 2)
    assert space.calls == 1


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0}, {"batch_size": True}, {"candidate_pool_size": -1},
    {"distance_metric": 2}, {"existing_samples": [[1, 2]]},
])
def test_invalid_inputs(kwargs):
    args = dict(existing_samples=None, batch_size=2)
    args.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        select(FixedSpace([[0], [1]]), **args)


def test_euclidean_nonfinite_parameters():
    with pytest.raises(ValueError, match="finite"):
        select(FixedSpace([[np.nan]]), None, 1)


def test_scaled_metric_changes_maximin_choice():
    space = FixedSpace([[1, 0], [0, 2]])
    np.testing.assert_array_equal(select(space, [[0, 0]], 1), [[0, 2]])
    np.testing.assert_array_equal(select(
        space, [[0, 0]], 1, distance_metric=lambda a, b: np.linalg.norm((a-b)*[100, 1]),
    ), [[1, 0]])


def test_standalone_selection_supports_vector_valued_parameter_objects():
    from romtools.workflows.parameters import GaussianParameter
    # This built-in space has one name for a two-component parameter. Sampling
    # needs only its dimensionality; model dictionary construction is separate.
    space = HeterogeneousParameterSpace([GaussianParameter("vector", [0, 0], [1, 1])])
    assert select(space, None, 3, random_seed=1).shape == (3, 2)


def test_all_builtin_seed_styles_are_reproducible():
    from romtools.workflows.parameter_spaces import MultivariateGaussianParameterSpace
    from romtools.workflows.sampling_methods import LatinHypercubeSampler, RandomizedQuasiMonteCarloSampler
    for sampler in (LatinHypercubeSampler, RandomizedQuasiMonteCarloSampler):
        space = MultivariateGaussianParameterSpace(["x", "y"], [0, 0], np.eye(2), sampler)
        np.testing.assert_array_equal(select(space, None, 3, candidate_pool_size=32, random_seed=8),
                                      select(space, None, 3, candidate_pool_size=32, random_seed=8))


def test_inputs_are_not_modified_and_large_integer_subtraction_does_not_overflow():
    values = np.array([[-2**62], [2**62]], dtype=np.int64)
    space = FixedSpace(values)
    existing = values[:1].copy()
    np.testing.assert_array_equal(select(space, existing, 1), values[1:])
    np.testing.assert_array_equal(space.samples, values)
    np.testing.assert_array_equal(existing, values[:1])
