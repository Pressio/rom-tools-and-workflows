"""Selection utilities for batch greedy training."""

import numpy as np


def _normalize_parameter_samples(parameter_samples: np.ndarray) -> np.ndarray:
    """Scale each parameter coordinate to ``[0, 1]`` over the candidate set.

    Constant coordinates are left at zero. Candidate-set scaling keeps the
    distance calculation invariant to simple changes of parameter units while
    avoiding additional requirements on ``ParameterSpace``.
    """
    samples = np.asarray(parameter_samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("parameter_samples must be a two-dimensional array")

    lower = np.min(samples, axis=0)
    span = np.max(samples, axis=0) - lower
    safe_span = np.where(span > 0.0, span, 1.0)
    normalized = (samples - lower) / safe_span
    normalized[:, span == 0.0] = 0.0
    return normalized


def select_batch(
    parameter_samples: np.ndarray,
    error_estimates: np.ndarray,
    batch_size: int,
    distance_exponent: float = 1.0,
) -> np.ndarray:
    """Select a batch using error-greedy sampling with distance penalization.

    The first point maximizes the supplied error estimate. Subsequent points
    maximize ``error * distance**distance_exponent``, where ``distance`` is the
    minimum Euclidean distance in normalized parameter coordinates to a point
    already selected in the current batch. Error estimates remain fixed while
    constructing the batch.

    Parameters
    ----------
    parameter_samples : numpy.ndarray
        Candidate parameters with shape ``(number_of_candidates, dimension)``.
    error_estimates : numpy.ndarray
        Nonnegative scalar error estimates for the candidates.
    batch_size : int
        Requested number of samples. If larger than the candidate set, all
        candidates are returned.
    distance_exponent : float, default=1.0
        Strength of the within-batch diversity penalty. A value of zero gives
        top-k selection by error estimate.

    Returns
    -------
    numpy.ndarray
        Integer indices into ``parameter_samples`` in selection order.
    """
    samples = np.asarray(parameter_samples, dtype=float)
    errors = np.asarray(error_estimates, dtype=float)

    if samples.ndim != 2:
        raise ValueError("parameter_samples must be a two-dimensional array")
    if errors.ndim != 1 or errors.size != samples.shape[0]:
        raise ValueError(
            "error_estimates must be one-dimensional with one value per candidate"
        )
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if distance_exponent < 0.0 or not np.isfinite(distance_exponent):
        raise ValueError("distance_exponent must be finite and nonnegative")
    if samples.shape[0] == 0:
        return np.zeros(0, dtype=int)
    if np.any(~np.isfinite(errors)) or np.any(errors < 0.0):
        raise ValueError("error_estimates must be finite and nonnegative")

    number_to_select = min(batch_size, samples.shape[0])

    # alpha=0 is deliberately a pure top-k error selection. Stable sorting
    # gives deterministic tie breaking in favor of lower candidate indices.
    if distance_exponent == 0.0:
        order = np.argsort(-errors, kind="stable")
        return order[:number_to_select].astype(int)

    normalized_samples = _normalize_parameter_samples(samples)
    selected = [int(np.argmax(errors))]
    available = np.ones(samples.shape[0], dtype=bool)
    available[selected[0]] = False

    while len(selected) < number_to_select:
        remaining = np.flatnonzero(available)
        selected_points = normalized_samples[np.asarray(selected)]
        differences = (
            normalized_samples[remaining, None, :] - selected_points[None, :, :]
        )
        distances = np.linalg.norm(differences, axis=2)
        minimum_distances = np.min(distances, axis=1)
        scores = errors[remaining] * minimum_distances**distance_exponent

        # np.argmax returns the first maximum, so ties remain deterministic.
        next_index = int(remaining[np.argmax(scores)])
        selected.append(next_index)
        available[next_index] = False

    return np.asarray(selected, dtype=int)
