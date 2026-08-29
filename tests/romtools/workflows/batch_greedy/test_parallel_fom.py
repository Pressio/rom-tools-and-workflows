"""Tests for concurrent FOM execution in the batch greedy workflow."""

import os
import time

import numpy as np
import pytest

from romtools.workflows.batch_greedy.run_batch_greedy import _run_fom_samples


class ParallelTestQoiModel:
    """Small picklable model that records the worker process used."""

    def populate_run_directory(self, run_directory, parameter_sample):
        del parameter_sample
        os.makedirs(run_directory, exist_ok=True)

    def run_model(self, run_directory, parameter_sample):
        del parameter_sample
        time.sleep(0.2)
        with open(os.path.join(run_directory, "pid.txt"), "w", encoding="utf-8") as file:
            file.write(str(os.getpid()))
        return 0

    def compute_qoi(self, run_directory, parameter_sample):
        del run_directory
        return np.array([parameter_sample["x"]])


class FailingParallelTestQoiModel(ParallelTestQoiModel):
    def run_model(self, run_directory, parameter_sample):
        if parameter_sample["x"] > 0.5:
            return 1
        return super().run_model(run_directory, parameter_sample)


def _prepare_run_directories(tmp_path, model, samples):
    for sample_index, sample in enumerate(samples):
        run_directory = tmp_path / "fom" / f"run_{sample_index}"
        model.populate_run_directory(str(run_directory), {"x": sample[0]})


@pytest.mark.mpi_skip
def test_fom_samples_execute_in_parallel(tmp_path):
    model = ParallelTestQoiModel()
    samples = np.array([[0.0], [1.0], [2.0], [3.0]])
    _prepare_run_directories(tmp_path, model, samples)

    qois = _run_fom_samples(
        model,
        ["x"],
        samples,
        np.arange(4),
        str(tmp_path),
        "run_",
        fom_evaluation_concurrency=2,
    )

    assert list(qois) == [0, 1, 2, 3]
    for sample_index in range(4):
        np.testing.assert_allclose(qois[sample_index], [float(sample_index)])

    worker_pids = {
        int((tmp_path / "fom" / f"run_{sample_index}" / "pid.txt").read_text())
        for sample_index in range(4)
    }
    assert len(worker_pids) == 2
    assert os.getpid() not in worker_pids


@pytest.mark.mpi_skip
def test_fom_sample_failure_propagates(tmp_path):
    model = FailingParallelTestQoiModel()
    samples = np.array([[0.0], [1.0]])
    _prepare_run_directories(tmp_path, model, samples)

    with pytest.raises(RuntimeError, match="FOM evaluation failed"):
        _run_fom_samples(
            model,
            ["x"],
            samples,
            np.arange(2),
            str(tmp_path),
            "run_",
            fom_evaluation_concurrency=2,
        )
