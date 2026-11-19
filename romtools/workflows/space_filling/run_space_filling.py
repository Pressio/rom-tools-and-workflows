"""Sequential batches of geometrically selected FOM runs for ROM training."""

from dataclasses import dataclass
import json
import os
from tempfile import mkdtemp
from typing import Callable, Optional, Tuple

import numpy as np

from romtools.workflows.model_builders import QoiModelBuilderWithTrainingData
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace
from .._adaptive.execution import _evaluate_batch, _process_state
from .._adaptive.sampling import _parameter_names, _positive_integer, select_space_filling_samples
from .._adaptive.validation import (
    KFoldValidation, SpaceFillingValidationContext, _build_rom, _error_value,
    relative_l2_error, _readonly_copy,
)


@dataclass
class SpaceFillingResult:
    """Final production ROM and ordered data from completed training rounds."""

    rom_model: QoiModel
    training_parameters: np.ndarray
    training_qois: np.ndarray
    training_dirs: Tuple[str, ...]
    validation_errors: np.ndarray
    sample_counts: np.ndarray
    converged: bool
    termination_reason: str


def _rng_states(streams):
    """JSON-compatible PCG64 states, recorded after each committed round."""
    return json.dumps([stream.bit_generator.state for stream in streams])


def _load_restart(directory, names, config, streams):
    checkpoint = os.path.join(directory, "space_filling_stats.npz")
    if not os.path.isfile(checkpoint):
        raise ValueError("restart requires an existing space_filling_stats.npz checkpoint")
    with np.load(checkpoint, allow_pickle=False) as saved:
        required = {
            "parameter_names", "training_parameters", "training_qois",
            "training_dirs", "validation_errors", "sample_counts",
            "fom_run_times", "rng_states", "run_config", "last_build_seed",
        }
        if not required.issubset(saved.files):
            raise ValueError("checkpoint predates restart support or is incomplete")
        if tuple(saved["parameter_names"].tolist()) != tuple(names):
            raise ValueError("restart parameter names do not match the checkpoint")
        if json.loads(str(saved["run_config"])) != config:
            raise ValueError("restart sampling configuration differs from checkpoint")
        parameters = saved["training_parameters"].copy()
        qois = saved["training_qois"].copy()
        training_dirs = saved["training_dirs"].tolist()
        errors = saved["validation_errors"].tolist()
        counts = saved["sample_counts"].tolist()
        run_times = saved["fom_run_times"].tolist()
        last_build_seed = int(saved["last_build_seed"])
        states = json.loads(str(saved["rng_states"]))
    if (not counts or len(errors) != len(counts)
            or counts[-1] != len(training_dirs)
            or len(parameters) != len(qois)
            or len(parameters) != len(training_dirs)
            or len(run_times) != len(training_dirs)
            or parameters.ndim != 2 or parameters.shape[1] != len(names)
            or qois.ndim != 2 or len(states) != len(streams)):
        raise ValueError("restart checkpoint contains inconsistent training history")
    root = os.path.realpath(directory)
    for run_directory in training_dirs:
        resolved = os.path.realpath(run_directory)
        if os.path.commonpath((root, resolved)) != root or not os.path.isfile(
            os.path.join(resolved, "passed.txt")
        ):
            raise ValueError(f"restart FOM directory is missing or invalid: {run_directory}")
    for stream, state in zip(streams, states):
        stream.bit_generator.state = state
    return parameters, qois, training_dirs, errors, counts, run_times, last_build_seed


def _save_stats(directory, names, parameters, qois, training_dirs, errors,
                counts, run_times, reason, streams, config, last_build_seed):
    temporary = os.path.join(directory, ".space_filling_stats.tmp.npz")
    np.savez(
        temporary, parameter_names=np.asarray(names), training_parameters=parameters,
        training_qois=qois, training_dirs=np.asarray(training_dirs),
        validation_errors=np.asarray(errors), sample_counts=np.asarray(counts),
        fom_run_times=np.asarray(run_times), converged=reason == "tolerance",
        termination_reason=reason,
        rng_states=_rng_states(streams), run_config=json.dumps(config),
        last_build_seed=last_build_seed,
    )
    os.replace(temporary, os.path.join(directory, "space_filling_stats.npz"))


def run_space_filling(
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
    restart: bool = False,
) -> SpaceFillingResult:
    """Train a ROM by alternating complete maximin FOM batches and validation.

    ``batch_size`` controls new training points independently of the active
    process limit ``max_concurrent_fom_runs``. Concurrent FOM objects must be
    serializable with multiprocessing's spawn context. All three model methods
    execute in one worker task and may use that task's independent directory.
    Builders and validation run sequentially in the parent process.

    The builder receives directories, parameter rows, and QoI rows in matching
    order, using QoiModelBuilderWithTrainingData's four-argument method. It must
    build independent models from only the supplied data. Default validation
    rebuilds five fold-specific ROMs and returns maximum held-out relative L2
    QoI error. ``error_functor(qoi_rom, qoi_fom)`` replaces that per-sample error
    completely; no normalization is applied to a custom functor's result.
    A custom ``validation_strategy(context)`` owns validation and aggregation.

    ``max_iterations`` includes the initial batch; ``max_fom_samples`` bounds
    training FOM runs scheduled here, excluding any references prepared by a
    custom validation strategy. Short candidate pools and final budget-limited
    batches are allowed. Fresh runs require an absent or empty absolute
    directory. Set ``restart=True`` to resume from the last *completed* round
    in that directory, using the saved sampling/RNG states and rebuilding the
    production ROM from accepted FOM data. ``max_iterations`` and
    ``max_fom_samples`` may be increased on restart; sampling settings and
    parameter names must remain unchanged. Pass the same builder, parameter
    space, validation strategy, distance metric, and error functor.
    Automatic retry/overwrite of a partially completed round is unsupported:
    its files must be inspected and cleared before restarting. Completed
    FOM samples are never rerun.
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
    if not isinstance(restart, bool):
        raise TypeError("restart must be a bool")
    tolerance = _error_value(tolerance, "tolerance")
    for name, value in (("distance_metric", distance_metric),
                        ("error_functor", error_functor),
                        ("validation_strategy", validation_strategy)):
        if value is not None and not callable(value):
            raise TypeError(f"{name} must be callable or None")
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
    # Separate streams make sampling independent of validation/build activity.
    streams = list(map(np.random.default_rng, np.random.SeedSequence(random_seed).spawn(4)))
    sampling_rng, fom_rng, validation_rng, build_rng = streams
    config = {
        "random_seed": None if random_seed is None else int(random_seed),
        "initial_sample_size": initial_sample_size,
        "batch_size": batch_size,
        "candidate_pool_size": candidate_pool_size,
    }
    directory = os.fspath(absolute_work_dir)
    if not os.path.isabs(directory):
        raise ValueError("absolute_work_dir must be absolute")
    if restart:
        if not os.path.isdir(directory):
            raise ValueError("restart requires an existing work directory")
        (parameters, qois, training_dirs, errors, counts, run_times,
         last_build_seed) = _load_restart(directory, names, config, streams)
        if max_fom_samples is not None and max_fom_samples < len(training_dirs):
            raise ValueError("max_fom_samples cannot be smaller than completed training runs")
        next_iteration = len(counts)
        # Never silently overwrite work from a failed/incomplete iteration.
        if os.path.exists(os.path.join(directory, f"iteration_{next_iteration}")):
            raise ValueError(
                f"iteration_{next_iteration} contains uncommitted work; "
                "inspect/clear the incomplete iteration and orphan FOM directories before restart"
            )
        # The previous in-memory ROM was not serialized. Rebuild it using the
        # same random seed, without advancing the restored RNG stream.
        rebuild_dir = mkdtemp(prefix=".restart_build_", dir=directory)
        rom = _build_rom(
            rom_model_builder, os.path.join(rebuild_dir, "offline_data"),
            training_dirs, parameters, qois, last_build_seed,
        )
        reason = "running"
    else:
        if os.path.exists(directory) and (not os.path.isdir(directory) or os.listdir(directory)):
            raise ValueError("absolute_work_dir must be absent or empty; use restart=True to resume")
        os.makedirs(directory, exist_ok=True)
        parameters = qois = rom = None
        training_dirs, errors, counts, run_times = [], [], [], []
        last_build_seed = -1
        next_iteration = 0
        reason = "max_iterations"

    mode = "a" if restart else "w"
    with open(os.path.join(directory, "space_filling_status.log"), mode, encoding="utf-8") as log:
        try:
            if restart:
                log.write(f"Restarting after {next_iteration} completed rounds\\n")
                if errors[-1] <= tolerance:
                    reason = "tolerance"
                elif max_fom_samples is not None and len(training_dirs) >= max_fom_samples:
                    reason = "max_fom_samples"
                elif next_iteration >= max_iterations:
                    reason = "max_iterations"
            if reason == "running" or not restart:
                for iteration in range(next_iteration, max_iterations):
                    target = initial_sample_size if iteration == 0 else batch_size
                    if max_fom_samples is not None:
                        target = min(target, max_fom_samples - len(training_dirs))
                    if target <= 0:
                        reason = "max_fom_samples"
                        break
                    # Refuse to overwrite an incomplete round, even if only
                    # its FOM run directories survived a failed build.
                    if os.path.exists(os.path.join(directory, f"iteration_{iteration}")):
                        raise ValueError(f"iteration_{iteration} already exists")
                    if any(os.path.exists(os.path.join(directory, "fom", f"run_{i}"))
                           for i in range(len(training_dirs), len(training_dirs) + target)):
                        raise ValueError("uncommitted FOM directories exist; inspect/clear them before restart")
                    batch = select_space_filling_samples(
                        parameter_space, parameters, target, candidate_pool_size,
                        distance_metric, int(sampling_rng.integers(0, 2**32)),
                    )
                    if iteration == 0:
                        if not len(batch):
                            raise ValueError("initial candidate pool contains no samples")
                        if isinstance(strategy, KFoldValidation) and len(batch) < strategy.num_folds:
                            raise ValueError("initial candidate pool has fewer distinct samples than num_folds")
                    if not len(batch):
                        reason = "candidate_pool_exhausted"
                        break
                    start = len(training_dirs)
                    ids = list(range(start, start + len(batch)))
                    batch_dirs = [os.path.join(directory, "fom", f"run_{i}") for i in ids]
                    seeds = [int(fom_rng.integers(0, 2**32)) for _ in ids]
                    log.write(f"Round {iteration}: selected {len(batch)} FOM samples\\n")
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
                    last_build_seed = int(build_rng.integers(0, 2**32))
                    rom = _build_rom(
                        rom_model_builder, os.path.join(round_directory, "offline_data"),
                        training_dirs, parameters, qois, last_build_seed,
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
                    log.write(f"Round {iteration}: {len(training_dirs)} training samples, error={error:.16g}\\n")
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
                                errors, counts, run_times, reason, streams, config, last_build_seed)
                    if reason != "running":
                        break
            # Pool exhaustion is detected before another round is built.
            _save_stats(directory, names, parameters, qois, training_dirs,
                        errors, counts, run_times, reason, streams, config, last_build_seed)
            log.write(f"Stopped: {reason}\n")
        except Exception as exception:
            log.write(f"Failed: {exception}\n")
            raise
    return SpaceFillingResult(
        rom, parameters, qois, tuple(training_dirs), np.asarray(errors),
        np.asarray(counts), reason == "tolerance", reason,
    )
