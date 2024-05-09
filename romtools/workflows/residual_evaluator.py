"""
Protocol for interfacing with external application to compute residual snapshots
corresponding to existing state snapshots.
"""

from typing import Protocol, Iterable
import numpy as np

from romtools.workflows.workflow_utils import create_empty_dir


class ResidualEvaluator(Protocol):
    """
    Baseline residual evaluator protocol
    """

    def compute_reduced_state(self, filename: str) -> np.ndarray:
        """
        Reads the full-order model solution from the specified filename
        and computes the corresponding reduced state

        Args:
            filename (str): filename of the file containing the full-order model solution data

        Returns:
            `np.ndarray`: The projected full-order solution in a 1-dimensional array

        """
        pass

    def compute_reduced_states(self, filename: str) -> (np.ndarray, np.ndarray):
        """
        Reads the full-order model solution and time stamps from the specified filename
        and computes the corresponding reduced states

        Args:
            filename (str): filename of the file containing the full-order model solution data

        Returns:
            `np.ndarray`: The projected full-order solution in a 2-dimensional array
            `np.ndarray`: The corresponding solution time stamps in a 1-dimensional array

        """
        pass

    def evaluate_full_residuals(
        self,
        run_directory: str,
        full_model_directory: str,
        reduced_states: np.ndarray,
        times: np.ndarray = None,
    ) -> np.ndarray:
        """
        Evaluate the full-order model residuals corresponding to full states reconstructed from
        an array of reduced states, reduced_states

        Args:
            run_directory (str): Absolute path to directory in which residual is being computed.
            full_model_directory (str): Absolute path to directory in which the full model data was computed.
            reduced_state (np.ndarray): 2-dimensional reduced state vector.
            times (np.ndarray, optional): 1-dimensional vector of time stamps.

        Returns:
            `np.ndarray`: The full-order residual in tensor form, should be 3-dimensional, even for a single time step
        """
        # TODO should this just return a list of directories instead?
        pass


def evaluate_and_load_steady_residual_snapshots(
    residual_evaluator: ResidualEvaluator,
    full_state_directories: list[str],
    state_filename: str,
    absolute_run_directory: str,
) -> np.ndarray:
    """
    Core algorithm that takes a residual_evaluator, a list of steady full-order model
    snapshot directories, and a snapshot_filename, and computes the corresponding
    residual snapshots.

    Args:
        residual_evaluator (UnsteadyResidualEvaluator): steady residual evaluator we wish to use
        full_state_directories (list[str]): list of directories containing full state data
        state_filename (str): filename or base filename of file containing state data
        absolute_run_directory (str): absolute path to base directory in which residuals are evaluated

    Returns:
        `np.ndarray`: The full-order residual snapshots in tensor form.
    """

    run_directory_base = f"{absolute_run_directory}/res_"

    all_residual_snapshots = []
    n_vars = -1
    n_x = -1
    for index, full_model_dir in enumerate(full_state_directories):
        # Read and project FOM snapshot
        reduced_state = residual_evaluator.compute_reduced_state(
            full_model_dir + "/" + state_filename
        )

        # Set up corresponding directory
        run_directory = f"{run_directory_base}{index}"
        create_empty_dir(run_directory)

        # Evaluate residual
        residual_snapshot = residual_evaluator.evaluate_full_residuals(
            run_directory, full_model_dir, reduced_state
        )

        # check residual snapshot size and shape
        if n_vars == -1 and n_x == -1:
            n_vars = residual_snapshot.shape[0]
            n_x = residual_snapshot.shape[1]
        assert residual_snapshot.shape[0] == n_vars
        assert residual_snapshot.shape[1] == n_x

        all_residual_snapshots.append(residual_snapshot)

    # convert list to array
    # does it kill memory usage to do this?
    return np.concatenate(all_residual_snapshots, axis=2)


def evaluate_and_load_unsteady_residual_snapshots(
    residual_evaluator: ResidualEvaluator,
    full_state_directories: list[str],
    state_filename: str,
    absolute_run_directory: str,
) -> np.ndarray:
    """
    Core algorithm that takes a residual_evaluator, a list of unsteady full-order model
    snapshot directories, and a snapshot_filename, and computes the corresponding
    residual snapshots.

    Args:
        residual_evaluator (UnsteadyResidualEvaluator): steady residual evaluator we wish to use
        full_state_directories (list[str]): list of directories containing full state data
        state_filename (str): filename or base filename of file containing state data
        absolute_run_directory (str): absolute path to base directory in which residuals are evaluated

    Returns:
        `np.ndarray`: The full-order residual snapshots in tensor form.
    """

    run_directory_base = f"{absolute_run_directory}/res_"

    all_residual_snapshots = []
    n_vars = -1
    n_x = -1
    for index, full_model_dir in enumerate(full_state_directories):
        # Read and project FOM snapshots
        reduced_states, times = residual_evaluator.compute_reduced_states(
            full_model_dir + "/" + state_filename
        )
        reduced_states, times = residual_evaluator.load_projected_full_solutions(
            full_model_dir + "/" + state_filename
        )

        # Set up corresponding directory
        run_directory = f"{run_directory_base}{index}"
        create_empty_dir(run_directory)

        # Evaluate residuals
        residual_snapshots = residual_evaluator.evaluate_full_residuals(
            run_directory, full_model_dir, reduced_states, times
        )

        # check residual snapshot size and shape
        assert residual_snapshots.ndim == 3
        if n_vars == -1 and n_x == -1:
            n_vars = residual_snapshots.shape[0]
            n_x = residual_snapshots.shape[1]
        assert residual_snapshots.shape[0] == n_vars
        assert residual_snapshots.shape[1] == n_x

        all_residual_snapshots.append(residual_snapshots)

    # convert list to array
    # does it kill memory usage to do this?
    return np.concatenate(all_residual_snapshots, axis=2)
