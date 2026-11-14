"""Shared execution of blocking models and scheduler-backed job handles."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from numbers import Integral
import multiprocessing
import os
import time
from typing import Any, Optional

from .models import JobHandle


@dataclass
class ModelEvaluation:
    run_directory: str
    parameter_sample: dict
    run_time: float = float("nan")
    output: Any = None
    _completion_directory: Optional[str] = field(default=None, init=False, repr=False, compare=False)

    def compute_qoi(self, model):
        """Read the QoI in the completion directory of a non-blocking run."""
        if self._completion_directory is None:
            return compute_evaluation_qoi(model, self)
        with _nonblocking_scope(self, None, self._completion_directory):
            return compute_evaluation_qoi(model, self)


def compute_evaluation_qoi(model, evaluation):
    """Read a model's QoI as an executor completion callback."""
    return model.compute_qoi(evaluation.run_directory, evaluation.parameter_sample)


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


def _finish(model, evaluation, status, start, postprocess, *, require_integer_status=True):
    evaluation.run_time = time.monotonic() - start
    if (require_integer_status and not isinstance(status, Integral)) or status != 0:
        raise ModelRunError(evaluation.run_directory, status, evaluation.run_time)
    if postprocess is not None:
        evaluation.output = postprocess(model, evaluation)
    return evaluation


def _run_blocking(model, evaluation, prepare, postprocess, scope, ignore_blocking_status):
    with scope(evaluation) if scope is not None else nullcontext():
        if prepare is not None:
            prepare(model, evaluation)
        start = time.monotonic()
        status = model.run_model(evaluation.run_directory, evaluation.parameter_sample)
        if ignore_blocking_status:
            status = 0
        return _finish(model, evaluation, status, start, postprocess, require_integer_status=False)


@contextmanager
def _nonblocking_scope(evaluation, scope, working_directory=None):
    directory = os.getcwd()
    try:
        with scope(evaluation) if scope is not None else nullcontext():
            if working_directory is not None:
                os.chdir(working_directory)
            yield
    finally:
        os.chdir(directory)


def _execute_nonblocking(model, evaluations, max_concurrent, poll_interval,
                         prepare, postprocess, scope, return_exceptions):
    results = [None] * len(evaluations)
    pending = {}
    next_index = 0
    failed_index = None
    try:
        while next_index < len(evaluations) or pending:
            while next_index < len(evaluations) and len(pending) < max_concurrent:
                index = next_index
                next_index += 1
                evaluation = evaluations[index]
                try:
                    with _nonblocking_scope(evaluation, scope):
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
                        pending[index] = (handle, start, os.getcwd())
                except Exception as error:
                    if not return_exceptions:
                        failed_index = index
                        raise
                    if index in pending:
                        _drain(model, evaluation, pending.pop(index), scope)
                    results[index] = error

            completed = False
            for index, job in list(pending.items()):
                evaluation = evaluations[index]
                handle, start, working_directory = job
                try:
                    with _nonblocking_scope(evaluation, scope, working_directory):
                        if not handle.done():
                            continue
                        completed = True
                        del pending[index]
                        status = handle.result()
                        evaluation._completion_directory = os.getcwd()
                        results[index] = _finish(
                            model, evaluation, status, start, postprocess
                        )
                except Exception as error:
                    if not return_exceptions:
                        failed_index = index
                        raise
                    if index in pending:
                        del pending[index]
                        _drain(model, evaluation, job, scope)
                    completed = True
                    results[index] = error
            if pending and not completed:
                time.sleep(poll_interval)
    except BaseException as error:
        # The minimal handle cannot cancel submitted work. On a batch error,
        # finish successful siblings so restartable workflows can reuse them.
        # On interruption, wait for the handles without running callbacks.
        finish_pending = isinstance(error, Exception)
        for index, job in pending.items():
            callback = postprocess if finish_pending and index != failed_index else None
            _drain(model, evaluations[index], job, scope, callback)
        raise
    return results


def _drain(model, evaluation, job, scope, postprocess=None):
    handle, start, working_directory = job
    try:
        with _nonblocking_scope(evaluation, scope, working_directory):
            status = handle.result()
            if postprocess is not None:
                evaluation._completion_directory = os.getcwd()
                _finish(model, evaluation, status, start, postprocess)
    except Exception:
        pass


def execute_models(model, evaluations, max_concurrent=1, poll_interval=0.1, *,
                   prepare=None, postprocess=None, scope=None, start_method="spawn",
                   return_exceptions=False, ignore_blocking_status=False):
    """Execute evaluations and return results in input order.

    Optional prepare and postprocess callbacks take (model, evaluation).
    Postprocessing runs only after a successful exit and its return value is
    stored in evaluation.output. Blocking model callbacks run in the same
    process as the model, using a process pool when concurrency exceeds one.
    Non-blocking models use the caller's process and keep at most
    max_concurrent handles in flight. An optional scope wraps each blocking
    evaluation or each submission and completion phase of a non-blocking one.
    Non-blocking polling and completion resume in the working directory left
    by submission and restore the caller's directory afterward.

    By default failures raise. With return_exceptions, failed entries contain
    exceptions and the remaining evaluations continue. Workflows that used
    to ignore blocking return values can preserve that behavior with
    ignore_blocking_status. Non-blocking exit statuses are always checked.
    After a non-blocking batch error, successful in-flight siblings still
    run postprocessing before the original error is raised.
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
        return model, evaluation, prepare, postprocess, scope, ignore_blocking_status

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
