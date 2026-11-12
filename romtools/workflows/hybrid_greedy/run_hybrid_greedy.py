"""Greedy-directed batches completed by geometric maximin sampling."""

from contextlib import nullcontext
from dataclasses import dataclass
import os
from tempfile import TemporaryDirectory
from typing import Callable, Optional, Tuple

import numpy as np

from romtools.workflows.model_builders import QoiModelBuilderWithTrainingData
from romtools.workflows.models import QoiModel, QoiModelWithErrorEstimate
from romtools.workflows.parameter_spaces import ParameterSpace
from .._adaptive.execution import _evaluate_batch, _process_state
from .._adaptive.sampling import (
    _draw_candidates, _nearest_distances, _parameter_names, _positive_integer,
    _unique_samples, _validate_samples, select_maximin_samples,
)
from .._adaptive.validation import (
    KFoldValidation, SpaceFillingValidationContext, _build_rom, _error_value,
    _readonly_copy, relative_l2_error,
)


@dataclass
class HybridGreedyResult:
    """Final ROM, accumulated FOM data, and completed-round selection history.

    ``greedy_parameters`` and ``greedy_error_indicators`` contain one entry per
    adaptation round, excluding initialization. ``sample_counts`` and
    ``validation_errors`` include initialization.
    """

    rom_model: QoiModelWithErrorEstimate
    training_parameters: np.ndarray
    training_qois: np.ndarray
    training_dirs: Tuple[str, ...]
    validation_errors: np.ndarray
    sample_counts: np.ndarray
    greedy_parameters: np.ndarray
    greedy_error_indicators: np.ndarray
    converged: bool
    termination_reason: str


def _candidate_indicators(rom, names, candidates, directory, rng, create_directories):
    indicators = []
    for index, candidate in enumerate(candidates):
        run_directory = os.path.join(directory, f"run_{index}")
        parameters = dict(zip(names, candidate))
        stage = "directory creation"
        try:
            with _process_state(int(rng.integers(0, 2**32))):
                if create_directories:
                    os.makedirs(run_directory)
                stage = "populate_run_directory"
                rom.populate_run_directory(run_directory, parameters)
                stage = "run_model"
                flag = rom.run_model(run_directory, parameters)
                if flag != 0:
                    raise RuntimeError(f"nonzero model return code: {flag!r}")
                stage = "compute_error_estimate"
                indicators.append(_error_value(
                    rom.compute_error_estimate(run_directory, parameters), "error indicator",
                ))
        except Exception as error:
            raise RuntimeError(
                f"candidate {index}, parameters={parameters!r}, directory={run_directory}: "
                f"failed during {stage}: {error}"
            ) from error
    return np.asarray(indicators)


def _save_stats(directory, names, parameters, qois, training_dirs, errors,
                counts, run_times, greedy_parameters, greedy_indicators, reason):
    temporary = os.path.join(directory, ".hybrid_greedy_stats.tmp.npz")
    np.savez(
        temporary, parameter_names=np.asarray(names), training_parameters=parameters,
        training_qois=qois, training_dirs=np.asarray(training_dirs),
        validation_errors=np.asarray(errors), sample_counts=np.asarray(counts),
        fom_run_times=np.asarray(run_times),
        greedy_parameters=np.asarray(greedy_parameters, dtype=parameters.dtype).reshape(len(greedy_parameters), len(names)),
        greedy_error_indicators=np.asarray(greedy_indicators),
        converged=reason == "tolerance", termination_reason=reason,
    )
    os.replace(temporary, os.path.join(directory, "hybrid_greedy_stats.npz"))


def run_hybrid_greedy(
    fom_model: QoiModel,
    rom_model_builder: QoiModelBuilderWithTrainingData,
    parameter_space: ParameterSpace,
    absolute_work_dir: str,
    tolerance: float,
    initial_sample_size: int = 10,
    batch_size: int = 5,
    candidate_pool_size: int = 1000,
    distance_metric: Optional[Callable] = None,
    max_concurrent_fom_runs: int = 5,
    random_seed: int = 1,
    *,
    error_functor: Optional[Callable] = None,
    validation_strategy: Optional[Callable] = None,
    max_iterations: int = 20,
    max_fom_samples: Optional[int] = None,
    create_rom_run_directories: bool = True,
) -> HybridGreedyResult:
    """Train with one greedy point and up to ``batch_size - 1`` maximin points.

    Initialization draws directly from ``parameter_space.generate_samples``;
    duplicate/metric-equivalent rows are removed in sampler order. Each later
    round draws a fresh pool, excludes training-equivalent points, ranks raw
    error indicators, and completes the batch by maximin selection from the
    same pool. Ties use candidate order. No parameter normalization is applied.
    A short pool or remaining FOM budget may shorten a batch.

    The four-argument builder receives all accumulated FOM directories,
    parameter rows, and QoI rows. Production ROMs must implement QoiModelWithErrorEstimate.
    Indicators rank candidates only; the unchanged space-filling validation
    strategy controls stopping (default: maximum held-out relative L2 error
    over five folds). Initial training is validated before any adaptation.

    Candidate ROM evaluations are serial and use temporary directories removed
    after selection, including on failure. Set ``create_rom_run_directories``
    to False for in-memory models: calls still receive nominal absolute paths,
    which need not exist. Validation retains its existing directory behavior.

    FOM batches use process-isolated execution with independent directories;
    parallel FOM objects must support multiprocessing spawn serialization.
    All tasks finish before any rebuild or failure is reported. Failed batches
    retain FOM files and never enter a ROM build. ``max_iterations`` includes
    initialization; ``max_fom_samples`` caps training runs, not custom validation
    references. Output must be an absent or empty absolute directory. Restart
    and automatic retries are not supported.
    """
    for name, value in (
        ("initial_sample_size", initial_sample_size), ("batch_size", batch_size),
        ("candidate_pool_size", candidate_pool_size),
        ("max_concurrent_fom_runs", max_concurrent_fom_runs),
        ("max_iterations", max_iterations),
    ):
        _positive_integer(value, name)
    if max_fom_samples is not None:
        _positive_integer(max_fom_samples, "max_fom_samples")
    tolerance = _error_value(tolerance, "tolerance")
    for name, value in (("distance_metric", distance_metric),
                        ("error_functor", error_functor),
                        ("validation_strategy", validation_strategy)):
        if value is not None and not callable(value):
            raise TypeError(f"{name} must be callable or None")
    if not isinstance(create_rom_run_directories, bool):
        raise TypeError("create_rom_run_directories must be a bool")
    for name in ("populate_run_directory", "run_model", "compute_qoi"):
        if not callable(getattr(fom_model, name, None)):
            raise TypeError(f"fom_model must implement {name}")
    if not callable(getattr(rom_model_builder, "build_from_training_dirs", None)):
        raise TypeError("rom_model_builder must implement build_from_training_dirs")
    names = _parameter_names(parameter_space)
    strategy = KFoldValidation() if validation_strategy is None else validation_strategy
    error_functor = relative_l2_error if error_functor is None else error_functor
    initial_count = min(initial_sample_size, max_fom_samples or initial_sample_size)
    if isinstance(strategy, KFoldValidation) and initial_count < strategy.num_folds:
        raise ValueError("initial sample count/budget must be at least num_folds")
    streams = np.random.SeedSequence(random_seed).spawn(5)
    sampling_rng, fom_rng, validation_rng, build_rng, candidate_rng = map(np.random.default_rng, streams)
    directory = os.fspath(absolute_work_dir)
    if not os.path.isabs(directory):
        raise ValueError("absolute_work_dir must be absolute")
    if os.path.exists(directory) and (not os.path.isdir(directory) or os.listdir(directory)):
        raise ValueError("absolute_work_dir must be absent or empty; overwrite/restart is unsupported")
    os.makedirs(directory, exist_ok=True)
    parameters = qois = rom = None
    training_dirs, errors, counts, run_times = [], [], [], []
    greedy_parameters, greedy_indicators = [], []
    reason = "max_iterations"
    with open(os.path.join(directory, "hybrid_greedy_status.log"), "w", encoding="utf-8") as log:
        try:
            for iteration in range(max_iterations):
                target = initial_sample_size if iteration == 0 else batch_size
                if max_fom_samples is not None:
                    target = min(target, max_fom_samples - len(training_dirs))
                candidates = _validate_samples(_draw_candidates(
                    parameter_space, target if iteration == 0 else candidate_pool_size,
                    int(sampling_rng.integers(0, 2**32)),
                ), len(names))
                if iteration == 0:
                    batch = _unique_samples(candidates, distance_metric)[:target]
                    if not len(batch):
                        raise ValueError("initial sample design contains no samples")
                    if isinstance(strategy, KFoldValidation) and len(batch) < strategy.num_folds:
                        raise ValueError("initial design has fewer distinct samples than num_folds")
                else:
                    nearest, _ = _nearest_distances(candidates, parameters, distance_metric)
                    candidates = candidates[nearest > 0]
                    if not len(candidates):
                        reason = "candidate_pool_exhausted"
                        break
                    candidate_directory = (
                        TemporaryDirectory(prefix=f"candidates_{iteration}_", dir=directory)
                        if create_rom_run_directories else nullcontext(
                            os.path.join(directory, f"iteration_{iteration}", "candidates")
                        )
                    )
                    with candidate_directory as candidate_dir:
                        indicators = _candidate_indicators(
                            rom, names, candidates, candidate_dir, candidate_rng,
                            create_rom_run_directories,
                        )
                    greedy_index = int(np.argmax(indicators))
                    greedy = candidates[greedy_index:greedy_index + 1].copy()
                    batch = greedy
                    if target > 1:
                        remaining = np.delete(candidates, greedy_index, axis=0)
                        filling = select_maximin_samples(
                            remaining, np.concatenate((parameters, greedy)), target - 1, distance_metric,
                        )
                        batch = np.concatenate((greedy, filling))
                start = len(training_dirs)
                ids = list(range(start, start + len(batch)))
                batch_dirs = [os.path.join(directory, "fom", f"run_{i}") for i in ids]
                seeds = [int(fom_rng.integers(0, 2**32)) for _ in ids]
                log.write(f"Round {iteration}: selected {len(batch)} FOM samples\n")
                log.flush()
                batch_qois, times = _evaluate_batch(
                    fom_model, names, batch, batch_dirs, ids, seeds,
                    max_concurrent_fom_runs, None if qois is None else qois.shape[1],
                )
                parameters = batch if parameters is None else np.concatenate((parameters, batch))
                qois = batch_qois if qois is None else np.concatenate((qois, batch_qois))
                training_dirs.extend(batch_dirs)
                run_times.extend(times)
                round_directory = os.path.join(directory, f"iteration_{iteration}")
                rom = _build_rom(
                    rom_model_builder, os.path.join(round_directory, "offline_data"),
                    training_dirs, parameters, qois, int(build_rng.integers(0, 2**32)),
                )
                for method in ("populate_run_directory", "run_model", "compute_qoi", "compute_error_estimate"):
                    if not callable(getattr(rom, method, None)):
                        raise TypeError(
                            f"hybrid ROM from iteration {iteration} must implement "
                            f"QoiModelWithErrorEstimate; missing callable {method}"
                        )
                validation_directory = os.path.join(round_directory, "validation")
                os.makedirs(validation_directory)
                validation_seed = int(validation_rng.integers(0, 2**32))
                context = SpaceFillingValidationContext(
                    rom, rom_model_builder, names, _readonly_copy(parameters),
                    _readonly_copy(qois), tuple(training_dirs), iteration,
                    validation_directory, validation_seed, error_functor,
                )
                with _process_state(validation_seed):
                    error = _error_value(strategy(context), "validation strategy result")
                errors.append(error)
                counts.append(len(training_dirs))
                if iteration:
                    greedy_parameters.append(greedy[0].copy())
                    greedy_indicators.append(float(indicators[greedy_index]))
                log.write(f"Round {iteration}: {len(training_dirs)} training samples, error={error:.16g}\n")
                log.flush()
                if error <= tolerance:
                    reason = "tolerance"
                elif max_fom_samples is not None and len(training_dirs) >= max_fom_samples:
                    reason = "max_fom_samples"
                elif iteration + 1 == max_iterations:
                    reason = "max_iterations"
                else:
                    reason = "running"
                _save_stats(directory, names, parameters, qois, training_dirs,
                            errors, counts, run_times, greedy_parameters, greedy_indicators, reason)
                if reason != "running":
                    break
            _save_stats(directory, names, parameters, qois, training_dirs,
                        errors, counts, run_times, greedy_parameters, greedy_indicators, reason)
            log.write(f"Stopped: {reason}\n")
        except Exception as exception:
            log.write(f"Failed: {exception}\n")
            raise
    return HybridGreedyResult(
        rom, parameters, qois, tuple(training_dirs), np.asarray(errors), np.asarray(counts),
        np.asarray(greedy_parameters, dtype=parameters.dtype).reshape(len(greedy_parameters), len(names)),
        np.asarray(greedy_indicators), reason == "tolerance", reason,
    )
