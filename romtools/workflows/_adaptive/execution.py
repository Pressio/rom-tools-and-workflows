"""Process-isolated model evaluation for adaptive sampling."""

from contextlib import contextmanager
from dataclasses import dataclass
import os
import time
from typing import Optional

import numpy as np

from romtools.workflows.model_execution import ModelEvaluation, ModelRunError, execute_models


@contextmanager
def _process_state(seed=None):
    """Keep legacy chdir/global-NumPy users from affecting later operations."""
    directory = os.getcwd()
    random_state = np.random.get_state()
    try:
        if seed is not None:
            np.random.seed(seed)
        yield
    finally:
        os.chdir(directory)
        np.random.set_state(random_state)


@dataclass
class _AdaptiveEvaluation(ModelEvaluation):
    seed: Optional[int] = None
    start: float = 0.0


def _evaluation_scope(evaluation):
    return _process_state(evaluation.seed)


def _prepare_sample(model, evaluation):
    evaluation.start = time.perf_counter()
    directory = evaluation.run_directory
    os.makedirs(directory, exist_ok=False)
    model.populate_run_directory(directory, evaluation.parameter_sample)


def _finish_sample(model, evaluation):
    directory = evaluation.run_directory
    raw = model.compute_qoi(directory, evaluation.parameter_sample)
    if np.iscomplexobj(raw):
        raise ValueError("QoI must be real-valued")
    qoi = np.asarray(raw, dtype=float).reshape(-1)
    if not qoi.size or not np.all(np.isfinite(qoi)):
        raise ValueError("QoI must be a nonempty finite numeric vector")
    np.savetxt(os.path.join(directory, "passed.txt"), [0], fmt="%i")
    return qoi.copy(), time.perf_counter() - evaluation.start


def _failure_message(error):
    if isinstance(error, ModelRunError):
        return f"nonzero model return code: {error}"
    return str(error)


def _evaluate_sample(model, names, values, directory, sample_id, seed=None):
    evaluation = _AdaptiveEvaluation(directory, dict(zip(names, values)), seed=seed)
    try:
        return execute_models(
            model, [evaluation], prepare=_prepare_sample, postprocess=_finish_sample,
            scope=_evaluation_scope,
        )[0].output
    except Exception as error:
        raise RuntimeError(f"sample {sample_id} failed in {directory}: {_failure_message(error)}") from error


def _evaluate_batch(model, names, samples, directories, sample_ids, seeds,
                    max_concurrent_fom_runs, qoi_dimension=None):
    evaluations = [
        _AdaptiveEvaluation(directory, dict(zip(names, values)), seed=seed)
        for values, directory, seed in zip(samples, directories, seeds)
    ]
    results = execute_models(
        model, evaluations, max_concurrent_fom_runs, prepare=_prepare_sample,
        postprocess=_finish_sample, scope=_evaluation_scope, return_exceptions=True,
    )
    failures = [(i, result) for i, result in enumerate(results) if isinstance(result, Exception)]
    # All work has finished before an error is reported or a builder sees this batch.
    if failures:
        details = "\n".join(
            f"sample {sample_ids[i]} ({directories[i]}): {_failure_message(error)}" for i, error in failures
        )
        raise RuntimeError(f"FOM batch failed; no ROM was rebuilt:\n{details}") from failures[0][1]
    results = [evaluation.output for evaluation in results]
    dimension = len(results[0][0]) if qoi_dimension is None else qoi_dimension
    for index, (qoi, _) in enumerate(results):
        if len(qoi) != dimension:
            raise RuntimeError(
                f"sample {sample_ids[index]} ({directories[index]}) returned inconsistent QoI dimension"
            )
    return np.vstack([r[0] for r in results]), np.asarray([r[1] for r in results])
