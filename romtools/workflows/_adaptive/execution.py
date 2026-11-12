"""Process-isolated model evaluation; workers never write campaign files."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import multiprocessing
import os
import time

import numpy as np


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


def _evaluate_sample(model, names, values, directory, sample_id, seed=None):
    """Run preparation, execution, and QoI extraction together in one process."""
    stage = "directory creation"
    start = time.perf_counter()
    try:
        with _process_state(seed):
            os.makedirs(directory, exist_ok=False)
            parameters = dict(zip(names, values))
            stage = "populate_run_directory"
            model.populate_run_directory(directory, parameters)
            stage = "run_model"
            flag = model.run_model(directory, parameters)
            if flag != 0:
                raise RuntimeError(f"nonzero model return code: {flag!r}")
            stage = "compute_qoi"
            raw = model.compute_qoi(directory, parameters)
            if np.iscomplexobj(raw):
                raise ValueError("QoI must be real-valued")
            qoi = np.asarray(raw, dtype=float).reshape(-1)
            if not qoi.size or not np.all(np.isfinite(qoi)):
                raise ValueError("QoI must be a nonempty finite numeric vector")
            np.savetxt(os.path.join(directory, "passed.txt"), [0], fmt="%i")
        return qoi, time.perf_counter() - start
    except Exception as error:
        raise RuntimeError(
            f"sample {sample_id} failed during {stage} in {directory}: {error}"
        ) from error


def _evaluate_batch(model, names, samples, directories, sample_ids, seeds,
                    max_concurrent_fom_runs, qoi_dimension=None):
    results = [None] * len(samples)
    failures = []
    arguments = list(zip([model] * len(samples), [names] * len(samples),
                         samples, directories, sample_ids, seeds))
    if max_concurrent_fom_runs == 1:
        for index, args in enumerate(arguments):
            try:
                results[index] = _evaluate_sample(*args)
            except Exception as error:
                failures.append((index, error))
    else:
        with ProcessPoolExecutor(
            max_workers=min(max_concurrent_fom_runs, len(samples)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = {}
            for index, args in enumerate(arguments):
                try:
                    futures[executor.submit(_evaluate_sample, *args)] = index
                except Exception as error:
                    failures.append((index, error))
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as error:
                    failures.append((index, error))
    # All submitted work has finished before an error is reported or a builder
    # can see this batch. Preserve every directory for diagnosis.
    if failures:
        failures.sort(key=lambda item: item[0])
        details = "\n".join(
            f"sample {sample_ids[i]} ({directories[i]}): {error}" for i, error in failures
        )
        raise RuntimeError(f"FOM batch failed; no ROM was rebuilt:\n{details}") from failures[0][1]
    dimension = len(results[0][0]) if qoi_dimension is None else qoi_dimension
    for index, (qoi, _) in enumerate(results):
        if len(qoi) != dimension:
            raise RuntimeError(
                f"sample {sample_ids[index]} ({directories[index]}) returned inconsistent QoI dimension"
            )
    return np.vstack([r[0] for r in results]), np.asarray([r[1] for r in results])
