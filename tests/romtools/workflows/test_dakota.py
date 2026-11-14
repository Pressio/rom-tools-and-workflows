"""Coverage for the Dakota driver with both model execution modes."""

import os
import sys

import numpy as np
import pytest

from romtools.workflows.dakota.dakota_coupler import run_model_for_dakota


class RelativeOutputModel:
    def populate_run_directory(self, directory, parameters):
        os.mkdir(os.path.join(directory, "solver"))

    def run_model(self, directory, parameters):
        os.chdir(os.path.join(directory, "solver"))
        np.save("solution.npy", [2 * parameters["x"]])
        return 0

    def compute_qoi(self, directory, parameters):
        return np.load("solution.npy")


@pytest.mark.mpi_skip
@pytest.mark.parametrize("nonblocking", [False, True], ids=["blocking", "nonblocking"])
def test_dakota_driver_reads_relative_model_output(tmp_path, monkeypatch, nonblocking, deferred_model):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.in").write_text("1 variables\n2 x\n")
    output = tmp_path / "results.out"
    monkeypatch.setattr(sys, "argv", ["driver.py", "params.in", str(output)])
    model = RelativeOutputModel()
    if nonblocking:
        model = deferred_model(model)

    run_model_for_dakota(model, multifidelity_flag=True, add_core_time_metadata=True)

    qoi, run_time = np.loadtxt(output)
    assert qoi == 4
    assert np.isfinite(run_time) and run_time >= 0
    if nonblocking:
        assert os.getcwd() == str(tmp_path)
        assert model.completed == {str(tmp_path)}
