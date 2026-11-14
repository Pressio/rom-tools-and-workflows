"""Shared model execution primitives.

This module deliberately does not own workflow-specific directory naming,
restart markers, or QoI aggregation.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from numbers import Integral
import multiprocessing
import time

from .models import JobHandle


@dataclass
class ModelEvaluation:
    run_directory: str
    parameter_sample: dict
    run_time: float = float('nan')


def _run_blocking(model, evaluation):
    start = time.monotonic()
    status = model.run_model(evaluation.run_directory, evaluation.parameter_sample)
    evaluation.run_time = time.monotonic() - start
    if not isinstance(status, Integral) or status != 0:
        raise RuntimeError(
            f"Model failed in {evaluation.run_directory}: exit status {status!r}"
        )
    return evaluation


def execute_models(model, evaluations, max_concurrent=1, poll_interval=0.1):
    """Execute model evaluations, returning them in input order.

    Run-directory population is intentionally the caller's responsibility.
    Blocking models use spawned worker processes. Non-blocking models submit
    handles in the calling process, with at most max_concurrent in flight.
    QoIs must be evaluated by the caller after this function returns.
    """
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be positive")
    if poll_interval < 0:
        raise ValueError("poll_interval must be nonnegative")
    evaluations = list(evaluations)
    if not evaluations:
        return []
    if getattr(model, "blocking", True):
        if max_concurrent == 1:
            return [_run_blocking(model, evaluation) for evaluation in evaluations]
        results = [None] * len(evaluations)
        with ProcessPoolExecutor(
            max_workers=max_concurrent,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = {
                pool.submit(_run_blocking, model, evaluation): index
                for index, evaluation in enumerate(evaluations)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results

    results = [None] * len(evaluations)
    pending = {}
    submitted_at = {}
    next_index = 0
    while next_index < len(evaluations) or pending:
        while next_index < len(evaluations) and len(pending) < max_concurrent:
            evaluation = evaluations[next_index]
            handle = model.run_model(
                evaluation.run_directory, evaluation.parameter_sample
            )
            if not isinstance(handle, JobHandle):
                raise TypeError(
                    f"Non-blocking model must return a JobHandle, got {type(handle)!r}"
                )
            pending[next_index] = handle
            submitted_at[next_index] = time.monotonic()
            next_index += 1
        completed = [index for index, handle in pending.items() if handle.done()]
        for index in completed:
            status = pending.pop(index).result()
            if not isinstance(status, Integral) or status != 0:
                raise RuntimeError(
                    f"Model failed in {evaluations[index].run_directory}: "
                    f"exit status {status!r}"
                )
            evaluations[index].run_time = time.monotonic() - submitted_at.pop(index)
            results[index] = evaluations[index]
        if pending and not completed:
            time.sleep(poll_interval)
    return results
