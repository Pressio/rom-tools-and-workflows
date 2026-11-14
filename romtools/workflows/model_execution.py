"""Shared execution of blocking models and scheduler-backed job handles."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from numbers import Integral
import multiprocessing
import os
import time
from typing import Any

from .models import JobHandle


@dataclass
class ModelEvaluation:
    run_directory: str
    parameter_sample: dict
    run_time: float = float("nan")
    output: Any = None


class ModelRunError(RuntimeError):
    """A model returned an unsuccessful exit status."""

    def __init__(self, directory, status, run_time):
        super().__init__(directory, status, run_time)
        self.run_time = run_time

    def __str__(self):
        directory, status, _ = self.args
        return f"Model failed in {directory}: exit status {status!r}"


def require_supported_concurrency(model, dispatcher, concurrency):
    """Apply dispatcher process-safety restrictions to blocking models."""
    if getattr(model, "blocking", True):
        dispatcher.require_supported_concurrency(concurrency)


def _finish(model, evaluation, status, start, postprocess):
    evaluation.run_time = time.monotonic() - start
    if not isinstance(status, Integral) or status != 0:
        raise ModelRunError(evaluation.run_directory, status, evaluation.run_time)
    if postprocess is not None:
        evaluation.output = postprocess(model, evaluation)
    return evaluation


def _run_blocking(model, evaluation, prepare, postprocess, scope, allow_implicit_success):
    with scope(evaluation) if scope is not None else nullcontext():
        if prepare is not None:
            prepare(model, evaluation)
        start = time.monotonic()
        status = model.run_model(evaluation.run_directory, evaluation.parameter_sample)
        if status is None and allow_implicit_success:
            status = 0
        return _finish(model, evaluation, status, start, postprocess)


@contextmanager
def _restore_directory():
    directory = os.getcwd()
    try:
        yield
    finally:
        os.chdir(directory)


def _execute_nonblocking(model, evaluations, max_concurrent, poll_interval,
                         prepare, postprocess, scope, return_exceptions):
    results = [None] * len(evaluations)
    pending = {}
    next_index = 0
    try:
        while next_index < len(evaluations) or pending:
            while next_index < len(evaluations) and len(pending) < max_concurrent:
                index = next_index
                next_index += 1
                evaluation = evaluations[index]
                try:
                    with scope(evaluation) if scope is not None else _restore_directory():
                        if prepare is not None:
                            prepare(model, evaluation)
                        start = time.monotonic()
                        handle = model.run_model(
                            evaluation.run_directory, evaluation.parameter_sample
                        )
                        if not isinstance(handle, JobHandle):
                            raise TypeError(
                                f"Non-blocking model must return a JobHandle, got {type(handle)!r}"
                            )
                        pending[index] = (handle, start)
                except Exception as error:
                    if not return_exceptions:
                        raise
                    results[index] = error

            completed = False
            for index, (handle, start) in list(pending.items()):
                evaluation = evaluations[index]
                try:
                    with scope(evaluation) if scope is not None else _restore_directory():
                        if not handle.done():
                            continue
                        completed = True
                        del pending[index]
                        results[index] = _finish(
                            model, evaluation, handle.result(), start, postprocess
                        )
                except Exception as error:
                    if not return_exceptions:
                        raise
                    if index in pending:
                        del pending[index]
                        _drain(handle)
                    completed = True
                    results[index] = error
            if pending and not completed:
                time.sleep(poll_interval)
    finally:
        # The minimal handle has no cancellation API. Wait for work that was
        # already submitted before unwinding and preserve the original error.
        for handle, _ in pending.values():
            _drain(handle)
    return results


def _drain(handle):
    try:
        handle.result()
    except Exception:
        pass


def execute_models(model, evaluations, max_concurrent=1, poll_interval=0.1, *,
                   prepare=None, postprocess=None, scope=None, start_method="spawn",
                   return_exceptions=False, allow_implicit_success=False):
    """Execute evaluations and return results in input order.

    Optional prepare and postprocess callbacks take (model, evaluation).
    Postprocessing runs only after a successful exit and its return value is
    stored in evaluation.output. Blocking model callbacks run in the same
    process as the model, using a process pool when concurrency exceeds one.
    Non-blocking models use the caller's process and keep at most
    max_concurrent handles in flight. An optional scope wraps each blocking
    evaluation or each submission and completion phase of a non-blocking one.

    By default failures raise. With return_exceptions, failed entries contain
    exceptions and the remaining evaluations continue. Workflows that used
    to ignore blocking return values can opt in to accepting a legacy None.
    """
    if (isinstance(max_concurrent, bool) or not isinstance(max_concurrent, Integral)
            or max_concurrent < 1):
        raise ValueError("max_concurrent must be a positive integer")
    if poll_interval < 0:
        raise ValueError("poll_interval must be nonnegative")
    evaluations = list(evaluations)
    if not evaluations:
        return []
    if not getattr(model, "blocking", True):
        return _execute_nonblocking(
            model, evaluations, max_concurrent, poll_interval, prepare,
            postprocess, scope, return_exceptions,
        )

    def collect(function, arguments):
        try:
            return function(*arguments)
        except Exception as error:
            if not return_exceptions:
                raise
            return error

    def arguments(evaluation):
        return model, evaluation, prepare, postprocess, scope, allow_implicit_success

    if max_concurrent == 1:
        return [
            collect(_run_blocking, arguments(evaluation))
            for evaluation in evaluations
        ]

    # Spawn by default to avoid inheriting initialized MPI state.
    with ProcessPoolExecutor(
        max_workers=int(max_concurrent),
        mp_context=multiprocessing.get_context(start_method),
    ) as pool:
        futures = [
            pool.submit(_run_blocking, *arguments(evaluation))
            for evaluation in evaluations
        ]
        return [collect(future.result, ()) for future in futures]
