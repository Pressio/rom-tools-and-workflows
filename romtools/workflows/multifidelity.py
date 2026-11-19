"""Fixed-allocation random adaptive training for ordered multifidelity QoI models.

This first workflow intentionally keeps fidelity allocation fixed. All model
evaluations use the existing blocking/nonblocking execution abstraction.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Optional, Sequence

import numpy as np

from romtools.rom.multifidelity import FidelityData
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows._adaptive.execution import _evaluate_batch, _process_state
from romtools.workflows._adaptive.sampling import (
    _parameter_names, _positive_integer,
)
from romtools.workflows._adaptive.validation import _error_value, relative_l2_error


@dataclass(frozen=True)
class FidelityLevel:
    """Executable model and fixed nominal cost, ordered low to high."""

    model: QoiModel
    name: str
    cost: float = 1.0


@dataclass(frozen=True)
class MultiFidelityValidationContext:
    """Inputs to a custom validation strategy; training data are copies."""

    rom_model: QoiModel
    rom_model_builder: object
    training_data: tuple[FidelityData, ...]
    parameter_names: tuple[str, ...]
    iteration: int
    validation_directory: str
    random_seed: int
    error_functor: Callable


@dataclass(frozen=True)
class MultiFidelityKFoldValidation:
    """K-fold CV of the highest fidelity, retaining all lower-level data.

    Lower-fidelity data at held-out high-fidelity locations remain available:
    CV assesses HF prediction when inexpensive evaluations are known.
    The result is the maximum per-sample held-out relative L2 error,
    consistent with ROMtools' existing KFoldValidation.
    """

    num_folds: int = 5

    def __post_init__(self):
        _positive_integer(self.num_folds, "num_folds")
        if self.num_folds < 2:
            raise ValueError("num_folds must be at least two")

    def __call__(self, context: MultiFidelityValidationContext) -> float:
        hf = context.training_data[-1]
        n = len(hf.parameters)
        if n < self.num_folds:
            raise ValueError("highest-fidelity samples must be >= num_folds")
        rng = np.random.default_rng(context.random_seed)
        folds = np.array_split(rng.permutation(n), self.num_folds)
        max_error = 0.0
        for index, holdout in enumerate(folds):
            keep = np.ones(n, dtype=bool)
            keep[holdout] = False
            train_hf = FidelityData(
                parameters=hf.parameters[keep].copy(),
                qois=hf.qois[keep].copy(),
                training_dirs=tuple(np.asarray(hf.training_dirs)[keep])
                if hf.training_dirs else (),
                name=hf.name,
            )
            training = tuple(context.training_data[:-1]) + (train_hf,)
            fold_dir = os.path.join(context.validation_directory, f"fold_{index}")
            offline = os.path.join(fold_dir, "offline_data")
            os.makedirs(offline, exist_ok=False)
            with _process_state(int(rng.integers(0, 2**32))):
                model = context.rom_model_builder.build(offline, training)
            for held in holdout:
                row = hf.parameters[int(held)]
                parameter_sample = dict(zip(context.parameter_names, row))
                # The surrogate is a QoiModel: preserve the workflow's
                # established populate/run/compute execution contract.
                run_dir = os.path.join(fold_dir, f"run_{int(held)}")
                os.makedirs(run_dir, exist_ok=False)
                with _process_state(int(rng.integers(0, 2**32))):
                    model.populate_run_directory(run_dir, parameter_sample)
                    status = model.run_model(run_dir, parameter_sample)
                    if status != 0:
                        raise RuntimeError(f"fold {index} surrogate evaluation failed: {status}")
                    predicted = np.asarray(model.compute_qoi(run_dir, parameter_sample),
                                           dtype=float).reshape(-1)
                reference = np.asarray(hf.qois[int(held)], dtype=float).reshape(-1)
                if predicted.shape != reference.shape:
                    raise ValueError("held-out QoI dimension mismatch")
                error = _error_value(context.error_functor(predicted, reference),
                                     "validation error")
                max_error = max(max_error, error)
        return max_error


@dataclass
class AdaptiveMultiFidelityResult:
    rom_model: QoiModel
    training_data: tuple[FidelityData, ...]
    validation_errors: np.ndarray
    sample_counts: np.ndarray
    total_cost: float
    converged: bool
    termination_reason: str


def _counts(value, n, name):
    if len(value) != n:
        raise ValueError(f"{name} must have one count per fidelity")
    result = tuple(value)
    for i, count in enumerate(result):
        _positive_integer(count, f"{name}[{i}]")
    return result


def _copy_data(data):
    """Do not let a builder or validation strategy change accumulated data."""
    copies = []
    for level in data:
        parameters = level.parameters.copy()
        qois = level.qois.copy()
        parameters.setflags(write=False)
        qois.setflags(write=False)
        copies.append(FidelityData(parameters, qois, level.training_dirs, level.name))
    return tuple(copies)


def _random_batches(parameter_space, names, counts, paired, rng):
    """Draw exactly the requested samples per fidelity, sharing higher points.

    When paired, samples for level l are a superset of level l+1.
    No additional LF evaluation is counted outside the specified allocation.
    """
    n = len(counts)
    result = [None] * n
    higher = np.empty((0, len(names)))
    for level in reversed(range(n)):
        shared = higher if paired else np.empty((0, len(names)))
        additional = counts[level] - len(shared)
        seed = int(rng.integers(0, 2**32))
        drawn = np.asarray(parameter_space.generate_samples(additional, seed=seed))
        if drawn.ndim != 2 or drawn.shape != (additional, len(names)):
            raise ValueError("ParameterSpace returned samples with an unexpected shape")
        if drawn.dtype.kind not in "biuf" or not np.all(np.isfinite(drawn)):
            raise ValueError("multifidelity sampling requires finite real parameters")
        result[level] = np.concatenate((shared, drawn)) if paired else drawn
        higher = result[level]
    return result


def run_adaptive_multifidelity(
    fidelities: Sequence[FidelityLevel],
    surrogate_builder,
    parameter_space: ParameterSpace,
    absolute_work_dir: str,
    tolerance: float,
    initial_samples: Sequence[int],
    batch_sizes: Sequence[int],
    *,
    adaptive_sampling_strategy: str = "random",
    pair_fidelity_samples: bool = True,
    validation_strategy: Optional[Callable] = None,
    error_functor: Callable = relative_l2_error,
    max_iterations: int = 20,
    max_total_cost: Optional[float] = None,
    max_concurrent_fom_runs: int = 1,
    random_seed: int = 1,
) -> AdaptiveMultiFidelityResult:
    """Build a multifidelity surrogate after each randomized sampling round.

    An iteration always schedules the requested counts at *every* fidelity.
    Iteration zero uses initial_samples; subsequent rounds use batch_sizes.
    With pairing, every higher-fidelity location is sampled at all lower
    levels and per-level counts must be nonincreasing. ParameterSpace controls
    the underlying random sampling distribution. Costs are fixed user inputs;
    incomplete rounds are not automatically retried or restarted.
    """
    levels = tuple(fidelities)
    if len(levels) < 2:
        raise ValueError("at least two fidelity levels are required")
    names = _parameter_names(parameter_space)
    if not names:
        raise ValueError("multifidelity sampling requires a nonempty parameter space")
    if not isinstance(pair_fidelity_samples, bool):
        raise TypeError("pair_fidelity_samples must be a bool")
    if adaptive_sampling_strategy != "random":
        raise ValueError("only adaptive_sampling_strategy='random' is supported")
    if not callable(getattr(surrogate_builder, "build", None)):
        raise TypeError("surrogate_builder must provide build()")
    if not callable(error_functor):
        raise TypeError("error_functor must be callable")
    for name, value in (("max_iterations", max_iterations),
                        ("max_concurrent_fom_runs", max_concurrent_fom_runs)):
        _positive_integer(value, name)
    tolerance = _error_value(tolerance, "tolerance")
    initial = _counts(initial_samples, len(levels), "initial_samples")
    batch = _counts(batch_sizes, len(levels), "batch_sizes")
    if pair_fidelity_samples:
        if any(a < b for counts in (initial, batch)
               for a, b in zip(counts[:-1], counts[1:])):
            raise ValueError("paired sampling requires nonincreasing per-level counts")
    if len(set(level.name for level in levels)) != len(levels):
        raise ValueError("fidelity names must be unique")
    for level in levels:
        cost = _error_value(level.cost, "fidelity cost")
        if cost <= 0:
            raise ValueError("fidelity costs must be positive")
        for method in ("populate_run_directory", "run_model", "compute_qoi"):
            if not callable(getattr(level.model, method, None)):
                raise TypeError(f"fidelity {level.name} lacks {method}")
    if max_total_cost is not None:
        max_total_cost = _error_value(max_total_cost, "max_total_cost")
    strategy = MultiFidelityKFoldValidation() if validation_strategy is None else validation_strategy
    if not callable(strategy):
        raise TypeError("validation_strategy must be callable")
    if isinstance(strategy, MultiFidelityKFoldValidation) and initial[-1] < strategy.num_folds:
        raise ValueError("initial highest-fidelity count must be >= num_folds")
    initial_cost = sum(level.cost * n for level, n in zip(levels, initial))
    if max_total_cost is not None and initial_cost > max_total_cost:
        raise ValueError("initial sample allocation exceeds max_total_cost")

    directory = os.fspath(absolute_work_dir)
    if not os.path.isabs(directory):
        raise ValueError("absolute_work_dir must be absolute")
    if os.path.exists(directory) and (not os.path.isdir(directory) or os.listdir(directory)):
        raise ValueError("absolute_work_dir must be absent or empty")
    os.makedirs(directory, exist_ok=True)

    sampling_rng, model_rng, validation_rng, build_rng = map(
        np.random.default_rng, np.random.SeedSequence(random_seed).spawn(4)
    )
    data = [FidelityData(np.empty((0, len(names))), np.empty((0, 1)),
                         name=level.name) for level in levels]
    errors, count_history = [], []
    total_cost = 0.0
    surrogate = None
    reason = "max_iterations"
    with open(os.path.join(directory, "multifidelity_status.log"), "w", encoding="utf8") as log:
        for iteration in range(max_iterations):
            allocations = initial if iteration == 0 else batch
            round_cost = sum(level.cost * n for level, n in zip(levels, allocations))
            if max_total_cost is not None and total_cost + round_cost > max_total_cost + 1e-12:
                reason = "max_total_cost"
                break
            batches = _random_batches(parameter_space, names, allocations,
                                      pair_fidelity_samples, sampling_rng)
            for fidelity, (level, points) in enumerate(zip(levels, batches)):
                old = data[fidelity]
                start = len(old.parameters)
                indices = list(range(start, start + len(points)))
                run_dirs = [
                    os.path.join(directory, f"fidelity_{fidelity}", f"run_{i}")
                    for i in indices
                ]
                seeds = [int(model_rng.integers(0, 2**32)) for _ in indices]
                qois, _ = _evaluate_batch(
                    level.model, names, points, run_dirs, indices, seeds,
                    max_concurrent_fom_runs,
                    None if start == 0 else old.qois.shape[1],
                )
                if start and qois.shape[1] != old.qois.shape[1]:
                    raise ValueError("fidelity QoI dimension changed across rounds")
                data[fidelity] = FidelityData(
                    np.concatenate((old.parameters, points)),
                    np.concatenate((old.qois, qois)) if start else qois,
                    old.training_dirs + tuple(run_dirs), level.name,
                )
            total_cost += round_cost
            round_dir = os.path.join(directory, f"iteration_{iteration}")
            offline = os.path.join(round_dir, "offline_data")
            os.makedirs(offline, exist_ok=False)
            training = _copy_data(data)
            with _process_state(int(build_rng.integers(0, 2**32))):
                surrogate = surrogate_builder.build(offline, training)
            validation_dir = os.path.join(round_dir, "validation")
            os.makedirs(validation_dir, exist_ok=False)
            seed = int(validation_rng.integers(0, 2**32))
            context = MultiFidelityValidationContext(
                surrogate, surrogate_builder, training, tuple(names), iteration,
                validation_dir, seed, error_functor,
            )
            with _process_state(seed):
                error = _error_value(strategy(context), "validation result")
            errors.append(error)
            count_history.append([len(level_data.parameters) for level_data in data])
            log.write(f"Iteration {iteration}: counts={count_history[-1]}, "
                      f"cost={total_cost:.12g}, validation_error={error:.12g}\n")
            log.flush()
            if error <= tolerance:
                reason = "tolerance"
                break

    np.savez(
        os.path.join(directory, "multifidelity_stats.npz"),
        sample_counts=np.asarray(count_history, dtype=int),
        validation_errors=np.asarray(errors),
        total_cost=total_cost,
        termination_reason=reason,
        **{f"parameters_{l}": entry.parameters for l, entry in enumerate(data)},
        **{f"qois_{l}": entry.qois for l, entry in enumerate(data)},
    )
    return AdaptiveMultiFidelityResult(
        surrogate, _copy_data(data), np.asarray(errors),
        np.asarray(count_history, dtype=int).reshape((-1, len(levels))),
        total_cost, reason == "tolerance", reason,
    )
