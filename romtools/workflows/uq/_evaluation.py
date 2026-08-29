"""Ordered execution of QoI models for UQ workflows."""

import concurrent.futures
import multiprocessing
import os
import time
from typing import Optional, Sequence

import numpy as np

from romtools.hpc.dispatchers import BaseDispatcher, LocalDispatcher
from romtools.workflows.models import QoiModel


def _parameter_dictionary(parameter_names: Sequence[str], values: np.ndarray) -> dict:
    return dict(zip(parameter_names, values))


def _flatten_qoi(model: QoiModel, run_directory: str, parameter_sample: dict) -> np.ndarray:
    qoi = np.asarray(model.compute_qoi(run_directory, parameter_sample))
    if qoi.ndim == 0:
        return qoi.reshape(1)
    return qoi.reshape(-1)


def _run_qoi_sample(
    model: QoiModel,
    run_directory: str,
    parameter_sample: dict,
    dispatcher: BaseDispatcher,
):
    start_time = time.time()
    flag = model.run_model(run_directory, parameter_sample)
    run_time = time.time() - start_time
    if flag != 0:
        raise RuntimeError(f"model evaluation failed in {run_directory} with flag {flag}")
    qoi = _flatten_qoi(model, run_directory, parameter_sample)
    dispatcher.np_savetxt(
        os.path.join(run_directory, "passed.txt"), np.array([0]), "%i"
    )
    return run_time, qoi


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
    if evaluation_concurrency < 1:
        raise ValueError("evaluation_concurrency must be at least one")
    samples = np.asarray(parameter_samples)
    if samples.ndim != 2:
        raise ValueError("parameter_samples must be two-dimensional")
    if samples.shape[1] != len(parameter_names):
        raise ValueError("parameter sample dimension does not match parameter names")

    dispatcher = dispatcher if dispatcher is not None else LocalDispatcher()
    qoi_values = [None] * samples.shape[0]
    run_times = np.full(samples.shape[0], np.nan)
    samples_to_run = []

    for local_index, parameter_values in enumerate(samples):
        sample_index = starting_sample_index + local_index
        run_directory = f"{run_directory_base}{sample_index}"
        passed_file = os.path.join(run_directory, "passed.txt")
        parameter_sample = _parameter_dictionary(parameter_names, parameter_values)
        dispatcher.create_empty_dir(run_directory)
        if dispatcher.path_exists(passed_file) and not overwrite:
            qoi_values[local_index] = _flatten_qoi(
                model, run_directory, parameter_sample
            )
        else:
            model.populate_run_directory(run_directory, parameter_sample)
            samples_to_run.append(
                (local_index, run_directory, parameter_sample)
            )

    if evaluation_concurrency == 1:
        for local_index, run_directory, parameter_sample in samples_to_run:
            run_time, qoi = _run_qoi_sample(
                model, run_directory, parameter_sample, dispatcher
            )
            run_times[local_index] = run_time
            qoi_values[local_index] = qoi
    elif samples_to_run:
        mp_context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=evaluation_concurrency, mp_context=mp_context
        ) as executor:
            futures = {
                executor.submit(
                    _run_qoi_sample,
                    model,
                    run_directory,
                    parameter_sample,
                    dispatcher,
                ): local_index
                for local_index, run_directory, parameter_sample in samples_to_run
            }
            for future, local_index in futures.items():
                run_time, qoi = future.result()
                run_times[local_index] = run_time
                qoi_values[local_index] = qoi

    try:
        qoi_matrix = np.vstack(qoi_values)
    except ValueError as error:
        raise ValueError("model returned inconsistent QoI dimensions") from error
    return qoi_matrix, run_times
