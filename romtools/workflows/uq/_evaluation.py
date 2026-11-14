"""Ordered execution of QoI models for UQ workflows."""

from functools import partial
from numbers import Integral
import os
from typing import Optional, Sequence

import numpy as np

from romtools.hpc.dispatchers import BaseDispatcher, resolve_dispatcher
from romtools.workflows.models import QoiModel
from romtools.workflows.model_execution import (
    ModelEvaluation, execute_models, require_supported_concurrency,
)


def _parameter_dictionary(parameter_names: Sequence[str], values: np.ndarray) -> dict:
    return dict(zip(parameter_names, values))


def _flatten_qoi(model: QoiModel, run_directory: str, parameter_sample: dict) -> np.ndarray:
    try:
        qoi = np.asarray(
            model.compute_qoi(run_directory, parameter_sample), dtype=float
        ).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError("model returned a nonnumeric QoI") from error
    if qoi.size == 0:
        raise ValueError("model returned an empty QoI")
    if not np.all(np.isfinite(qoi)):
        raise ValueError("model returned a nonfinite QoI")
    return qoi.copy()


def _finish_qoi_sample(model, evaluation, dispatcher):
    qoi = _flatten_qoi(model, evaluation.run_directory, evaluation.parameter_sample)
    dispatcher.np_savetxt(
        os.path.join(evaluation.run_directory, "passed.txt"), np.array([0]), "%i"
    )
    return qoi


def validate_evaluation_concurrency(evaluation_concurrency: int) -> None:
    """Validate the requested concurrency in both preparation and execution."""
    if (
        isinstance(evaluation_concurrency, bool)
        or not isinstance(evaluation_concurrency, Integral)
        or evaluation_concurrency < 1
    ):
        raise ValueError("evaluation_concurrency must be a positive integer")


def evaluate_qoi_model(
    model: QoiModel,
    parameter_names: Sequence[str],
    parameter_samples: np.ndarray,
    run_directory_base: str,
    evaluation_concurrency: int = 1,
    overwrite: bool = False,
    dispatcher: Optional[BaseDispatcher] = None,
    starting_sample_index: int = 0,
):
    """Evaluate a QoI model while preserving the input sample ordering."""
    validate_evaluation_concurrency(evaluation_concurrency)
    samples = np.asarray(parameter_samples)
    if samples.ndim != 2:
        raise ValueError("parameter_samples must be two-dimensional")
    if samples.shape[1] != len(parameter_names):
        raise ValueError("parameter sample dimension does not match parameter names")

    dispatcher = resolve_dispatcher(dispatcher)
    require_supported_concurrency(model, dispatcher, evaluation_concurrency)
    qoi_values = [None] * samples.shape[0]
    run_times = np.full(samples.shape[0], np.nan)
    samples_to_run = []

    for local_index, parameter_values in enumerate(samples):
        sample_index = starting_sample_index + local_index
        run_directory = f"{run_directory_base}{sample_index}"
        passed_file = os.path.join(run_directory, "passed.txt")
        parameter_sample = _parameter_dictionary(parameter_names, parameter_values)
        dispatcher.create_empty_dir(run_directory)
        passed = dispatcher.path_exists(passed_file)
        if passed and not overwrite:
            qoi_values[local_index] = _flatten_qoi(
                model, run_directory, parameter_sample
            )
        else:
            if passed:
                dispatcher.remove(passed_file)
            model.populate_run_directory(run_directory, parameter_sample)
            samples_to_run.append((local_index, ModelEvaluation(run_directory, parameter_sample)))

    completed = execute_models(
        model, (evaluation for _, evaluation in samples_to_run), evaluation_concurrency,
        postprocess=partial(_finish_qoi_sample, dispatcher=dispatcher),
    )
    for (local_index, _), evaluation in zip(samples_to_run, completed):
        qoi_values[local_index] = evaluation.output
        run_times[local_index] = evaluation.run_time

    try:
        qoi_matrix = np.vstack(qoi_values)
    except ValueError as error:
        raise ValueError("model returned inconsistent QoI dimensions") from error
    return qoi_matrix, run_times


def prepare_qoi_model(
    model: QoiModel,
    parameter_names: Sequence[str],
    parameter_samples: np.ndarray,
    run_directory_base: str,
    overwrite: bool = False,
    dispatcher: Optional[BaseDispatcher] = None,
    starting_sample_index: int = 0,
) -> tuple[str, ...]:
    """Prepare model input directories without running simulations or QoIs."""
    samples = np.asarray(parameter_samples)
    if samples.ndim != 2 or samples.shape[1] != len(parameter_names):
        raise ValueError("parameter samples must be two-dimensional and match parameter names")
    dispatcher = resolve_dispatcher(dispatcher)
    directories = []
    for local_index, parameter_values in enumerate(samples):
        run_directory = f"{run_directory_base}{starting_sample_index + local_index}"
        directories.append(run_directory)
        dispatcher.create_empty_dir(run_directory)
        passed_file = os.path.join(run_directory, "passed.txt")
        if overwrite or not dispatcher.path_exists(passed_file):
            if overwrite and dispatcher.path_exists(passed_file):
                dispatcher.remove(passed_file)
            model.populate_run_directory(
                run_directory, _parameter_dictionary(parameter_names, parameter_values)
            )
    return tuple(directories)
