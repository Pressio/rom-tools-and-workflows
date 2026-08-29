import numpy as np

from romtools.workflows.batch_greedy import select_batch


def test_batch_size_one_returns_maximum_error_candidate():
    samples = np.array([[0.0], [0.5], [1.0]])
    errors = np.array([0.2, 0.9, 0.4])
    np.testing.assert_array_equal(select_batch(samples, errors, 1), [1])


def test_zero_distance_exponent_returns_top_k_by_error():
    samples = np.array([[0.0], [0.1], [1.0], [0.8]])
    errors = np.array([0.9, 1.0, 0.8, 0.7])
    selected = select_batch(samples, errors, 3, distance_exponent=0.0)
    np.testing.assert_array_equal(selected, [1, 0, 2])


def test_distance_penalty_avoids_clustered_candidates():
    samples = np.array([[0.0], [0.01], [1.0]])
    errors = np.array([1.0, 0.99, 0.8])
    np.testing.assert_array_equal(select_batch(samples, errors, 2), [0, 2])


def test_parameter_scaling_does_not_change_selection():
    samples = np.array([[0.0, 0.0], [0.1, 10.0], [1.0, 100.0], [0.8, 40.0]])
    errors = np.array([1.0, 0.95, 0.8, 0.7])
    selected = select_batch(samples, errors, 3)
    scaled = samples * np.array([1.0e6, 1.0e-3])
    np.testing.assert_array_equal(selected, select_batch(scaled, errors, 3))


def test_ties_are_deterministic():
    samples = np.array([[0.0], [1.0], [0.5]])
    errors = np.ones(3)
    np.testing.assert_array_equal(select_batch(samples, errors, 3), [0, 1, 2])


def test_batch_larger_than_candidate_set_returns_all_candidates():
    samples = np.array([[0.0], [0.5], [1.0]])
    errors = np.array([0.2, 0.9, 0.4])
    selected = select_batch(samples, errors, 10)
    assert sorted(selected.tolist()) == [0, 1, 2]
