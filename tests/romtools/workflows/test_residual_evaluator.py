import pytest
import os
import numpy as np

from romtools.workflows.residual_evaluator import *


class MockSteadyResidualEvaluator:
    def __init__(self):
        pass

    def compute_reduced_state(self, filename: str) -> np.ndarray:
        return np.zeros((3, 4))

    def evaluate_full_residuals(
        self,
        run_directory: str,
        full_model_directory: str,
        reduced_state: np.ndarray,
        parameter_sample: dict,
    ) -> np.ndarray:
        return np.zeros((3, 10, 1))


class MockUnsteadyResidualEvaluator:
    def __init__(self):
        pass

    def compute_reduced_states(self, filename: str) -> (np.ndarray, np.ndarray):
        return np.zeros((4, 5)), np.ones(5)

    def evaluate_full_residuals(
        self,
        run_directory: str,
        full_model_directory: str,
        reduced_states: np.ndarray,
        parameter_sample: dict,
        times: np.ndarray,
    ) -> np.ndarray:
        return np.zeros((3, 10, 5))


class MockUnsteadyResidualEvaluatorNonSequentialTemporalData:
    def __init__(self):
        pass

    def compute_reduced_states(self, filename: str) -> (np.ndarray, np.ndarray):
        return np.zeros((4, 5, 2)), np.ones((5, 2))

    def evaluate_full_residuals(
        self,
        run_directory: str,
        full_model_directory: str,
        reduced_states: np.ndarray,
        parameter_sample: dict,
        times: np.ndarray,
    ) -> np.ndarray:
        return np.zeros((3, 10, 5))


@pytest.mark.mpi_skip
def test_steady_residual_evaluator(tmp_path):
    # see https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html for more info
    print("\n", tmp_path)

    residual_evaluator = MockSteadyResidualEvaluator()
    num_dirs = 5
    fom_dirs = ["{tmp_path}/run_{i}" for i in range(num_dirs)]
    fom_filename = "dummy"

    res_snaps = evaluate_and_load_steady_residual_snapshots(
        residual_evaluator, fom_dirs, fom_filename, tmp_path
    )

    assert res_snaps.shape == (3, 10, num_dirs)

    for i in range(num_dirs):
        assert os.path.isdir(f"{tmp_path}/res_" + str(i))


@pytest.mark.mpi_skip
def test_unsteady_residual_evaluator(tmp_path):
    # see https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html for more info
    print("\n", tmp_path)

    residual_evaluator = MockUnsteadyResidualEvaluator()
    num_dirs = 3
    fom_dirs = ["{tmp_path}/run_{i}" for i in range(num_dirs)]
    fom_filename = "dummy"

    res_snaps = evaluate_and_load_unsteady_residual_snapshots(
        residual_evaluator, fom_dirs, fom_filename, tmp_path
    )

    assert res_snaps.shape == (3, 10, num_dirs * 5)

    for i in range(num_dirs):
        assert os.path.isdir(f"{tmp_path}/res_" + str(i))


@pytest.mark.mpi_skip
def test_unsteady_residual_evaluator_nonsequential_temporal_data(tmp_path):
    # see https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html for more info
    print("\n", tmp_path)

    residual_evaluator = MockUnsteadyResidualEvaluatorNonSequentialTemporalData()
    num_dirs = 3
    fom_dirs = ["{tmp_path}/run_{i}" for i in range(num_dirs)]
    fom_filename = "dummy"

    res_snaps = evaluate_and_load_unsteady_residual_snapshots(
        residual_evaluator, fom_dirs, fom_filename, tmp_path
    )

    assert res_snaps.shape == (3, 10, num_dirs * 5)

    for i in range(num_dirs):
        assert os.path.isdir(f"{tmp_path}/res_" + str(i))


if __name__ == "__main__":
    test_steady_residual_evaluator(".")
    test_unsteady_residual_evaluator(".")
    test_unsteady_residual_evaluator_nonsequential_temporal_data(".")
