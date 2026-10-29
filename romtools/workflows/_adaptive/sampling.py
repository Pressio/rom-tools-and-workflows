"""Incremental maximin sampling using only the public ParameterSpace interface."""

import inspect
from numbers import Integral
from typing import Callable, Optional

import numpy as np

from romtools.workflows.parameter_spaces import ParameterSpace


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _parameter_dimension(parameter_space):
    dimension = parameter_space.get_dimensionality()
    if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension < 0:
        raise ValueError("parameter dimensionality must be a nonnegative integer")
    return dimension


def _parameter_names(parameter_space):
    names = tuple(parameter_space.get_names())
    if _parameter_dimension(parameter_space) != len(names) or len(set(names)) != len(names):
        raise ValueError("parameter names must be unique and match dimensionality")
    return names


def _draw_candidates(parameter_space, count, seed):
    # Some legacy spaces omit seed; do not mistake an internal TypeError for
    # an unsupported keyword and accidentally call a sampler twice.
    method = parameter_space.generate_samples
    signature = inspect.signature(method)
    try:
        signature.bind(count, seed=seed)
    except TypeError:
        signature.bind(count)
        kwargs = {}
    else:
        kwargs = {"seed": seed}
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return np.asarray(method(count, **kwargs))
    finally:
        np.random.set_state(state)


def select_space_filling_samples(
    parameter_space: ParameterSpace,
    existing_samples: Optional[np.ndarray],
    batch_size: int,
    candidate_pool_size: int = 1000,
    distance_metric: Optional[Callable] = None,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Select up to ``batch_size`` maximin-separated candidate rows.

    Candidates come exclusively from ``parameter_space.generate_samples``.
    No bounds or parameter normalization are assumed. With no existing rows,
    the first point is selected randomly; subsequent ties use candidate order.
    A short result means the sampled pool has insufficient distinct points,
    not that the entire domain is exhausted. Zero metric distance identifies
    duplicates, including points coincident with existing training data.

    The default metric is Euclidean and requires finite numeric arrays. A
    custom ``distance_metric(candidate, reference)`` must return a finite,
    nonnegative scalar and may operate on string/object rows. Use it when
    parameter units, scales, or geometry make raw Euclidean distance unsuitable.

    Sampling is reproducible when the space honors its seed argument or uses
    NumPy's legacy RNG. Independent, unseeded RNGs owned by custom spaces cannot
    be controlled by this function. The caller's NumPy RNG state is preserved.
    """
    _positive_integer(batch_size, "batch_size")
    _positive_integer(candidate_pool_size, "candidate_pool_size")
    if distance_metric is not None and not callable(distance_metric):
        raise TypeError("distance_metric must be callable or None")
    dimension = _parameter_dimension(parameter_space)
    rng = np.random.default_rng(random_seed)
    seed = int(rng.integers(0, 2**32))
    candidates = _draw_candidates(parameter_space, candidate_pool_size, seed)
    if candidates.ndim != 2 or candidates.shape[1] != dimension:
        raise ValueError("generated samples must have shape (number_of_samples, dimensionality)")
    return _select_maximin_samples(candidates, existing_samples, batch_size, distance_metric, rng)


def _nearest_distances(candidates, existing_samples, distance_metric):
    """Return nearest distances and an incremental distance-vector evaluator."""
    dimension = candidates.shape[1]
    existing = np.empty((0, dimension)) if existing_samples is None else np.asarray(existing_samples)
    if existing.ndim == 1 and existing.size == 0:
        existing = existing.reshape(0, dimension)
    if existing.ndim != 2 or existing.shape[1] != dimension:
        raise ValueError("existing_samples must be two-dimensional and match dimensionality")

    points, references = candidates, existing
    if distance_metric is None:
        if candidates.dtype.kind not in "biufc" or existing.dtype.kind not in "biufc":
            raise ValueError("Euclidean distance requires numeric parameters; supply distance_metric")
        # Promote before subtraction to avoid integer arithmetic overflow.
        dtype = np.result_type(candidates.dtype, existing.dtype, np.float64)
        points, references = candidates.astype(dtype), existing.astype(dtype)
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(references)):
            raise ValueError("Euclidean distance requires finite parameters")

    def distances(reference):
        if distance_metric is None:
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.linalg.norm(points - np.asarray(reference, dtype=points.dtype), axis=1)
        else:
            values = []
            for candidate in points:
                value = np.asarray(distance_metric(candidate.copy(), reference.copy()))
                if value.ndim != 0 or value.dtype.kind not in "iuf":
                    raise ValueError("distance_metric must return a finite nonnegative scalar")
                values.append(float(value))
            values = np.asarray(values)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("distance_metric must return finite nonnegative distances")
        return values

    nearest = np.full(len(candidates), np.inf)
    for reference in references:
        np.minimum(nearest, distances(reference), out=nearest)
    return nearest, distances


def _validate_samples(samples, dimension, description="generated samples"):
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != dimension:
        raise ValueError(f"{description} must have shape (number_of_samples, dimensionality)")
    return samples


def _select_maximin_samples(candidates, existing_samples, batch_size, distance_metric, rng=None):
    _positive_integer(batch_size, "batch_size")
    if distance_metric is not None and not callable(distance_metric):
        raise TypeError("distance_metric must be callable or None")
    candidates = np.asarray(candidates)
    if candidates.ndim != 2:
        raise ValueError("candidates must be two-dimensional")
    nearest, distances = _nearest_distances(candidates, existing_samples, distance_metric)
    has_references = existing_samples is not None and len(existing_samples) > 0
    selected = []
    for _ in range(min(batch_size, len(candidates))):
        if not selected and not has_references:
            index = 0 if rng is None else int(rng.integers(len(candidates)))
        else:
            index = int(np.argmax(nearest))
            if nearest[index] <= 0:
                break
        selected.append(index)
        np.minimum(nearest, distances(candidates[index]), out=nearest)
        nearest[selected] = -np.inf
    return candidates[selected].copy()


def select_maximin_samples(candidates, existing_samples, batch_size, distance_metric=None):
    """Select up to ``batch_size`` distinct rows from an explicit candidate pool.

    Ties use input order; with no existing samples, start at the first candidate.
    Zero distance excludes equivalent points. Euclidean distances are vectorized
    and updated incrementally; custom metrics may use nonnumeric parameter rows.
    Input arrays are never modified.
    """
    return _select_maximin_samples(candidates, existing_samples, batch_size, distance_metric)


def _unique_samples(samples, distance_metric):
    """Keep first occurrences in sampler order, using metric equivalence."""
    nearest, distances = _nearest_distances(samples, None, distance_metric)
    selected = []
    for index in range(len(samples)):
        if nearest[index] > 0:
            selected.append(index)
            np.minimum(nearest, distances(samples[index]), out=nearest)
    return samples[selected].copy()
