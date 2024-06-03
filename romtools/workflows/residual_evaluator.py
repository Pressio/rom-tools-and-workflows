"""
Protocol for interfacing with external application to compute residual snapshots
corresponding to existing state snapshots.
"""

from typing import Protocol, Iterable, Tuple
import numpy as np

from romtools.workflows.workflow_utils import create_empty_dir


class ResidualEvaluator(Protocol):
    """
    Baseline residual evaluator protocol
    """

    def compute_reduced_states(self, filename: str) -> np.ndarray:
        """
        Reads the full-order model solution(s) from the specified filename
        and computes the corresponding reduced state(s)

        Args:
            filename (str): filename of the file containing the full-order model solution data

        Returns:
            `np.ndarray`: The projected full-order solution in a 1- or 2-dimensional array.
            1st dimension: reduced state, 2nd dimension: sample index corresponding full-model
            solution data file.

        """
        pass

    def evaluate_full_residuals(
        self,
        run_directory: str,
        full_model_directory: str,
        reduced_states: np.ndarray,
        parameter_sample: dict,
    ) -> np.ndarray:
        """
        Evaluate the full-order model residuals corresponding to full states reconstructed from
        an array of reduced states

        Args:
            run_directory (str): Absolute path to directory in which residual is being computed.
            full_model_directory (str): Absolute path to directory in which the full model data was computed.
            reduced_state (np.ndarray): 2-dimensional reduced state vector. 1st dimension: reduced state,
            2nd dimension: sample index
            parameter_sample: Dictionary contatining parameter names and sample values

        Returns:
            `np.ndarray`: The full-order residual in tensor form, should be 3-dimensional, even for a single sample
        """
        pass


class TransientResidualEvaluator(Protocol):
    """
    Baseline residual evaluator protocol
    """

    def compute_reduced_states(self, filename: str) -> np.ndarray:
        """
        Reads the full-order model solution and time stamps from the specified filename
        and computes the corresponding reduced states

        Args:
            filename (str): filename of the file containing the full-order model solution data

        Returns:
            `np.ndarray`: The projected full-order solution in a 2- or 3-dimensional array. 1st
            dimension: reduced state, 2nd dimension: time, 3rd dimension: other states in time
            stencil. If the array is 2-dimensional, it is assumed that the states are sequential
            in time.

        """
        pass

    def get_times(self, filename: str) -> np.ndarray:
        """
        Reads and outputs time stamps from the specified filename

        Args:
            filename (str): filename of the file containing the full-order model solution data

        Returns:
            `np.ndarray`: The corresponding solution time stamps in a 1- or 2-dimensional array.
            The optional 2nd dimension is only needed if the reduced projected full-order solution
            array is 3-dimensional; the 2nd dimension contains the time stamps of the other states
            in the time-stencil for a given time stamp.

        """
        pass

    def evaluate_full_residuals(
        self,
        run_directory: str,
        full_model_directory: str,
        reduced_states: np.ndarray,
        parameter_sample: dict,
        times: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the full-order model residuals corresponding to full states reconstructed from
        an array of reduced states

        Args:
            run_directory (str): Absolute path to directory in which residual is being computed.
            full_model_directory (str): Absolute path to directory in which the full model data was computed.
            reduced_state (np.ndarray): 2- or 3-dimensional reduced state vector. 1st dimension: reduced state,
            2nd dimension: time, 3rd dimension: other states in time stencil. If the array is 2-dimensional, it
            is assumed that the states are sequential in time.
            parameter_sample: Dictionary contatining parameter names and sample values
            times (np.ndarray): 1-dimensional or 2-dimesional vector of time stamps. The optional
            2nd dimension is only needed if the reduced state array is 3-dimensional; the 2nd dimension
            contains the time stamps of the other states in the time-stencil for a given time stamp.

        Returns:
            `np.ndarray`: The full-order residual in tensor form, should be 3-dimensional, even for a single time step
        """
        pass


def evaluate_and_load_steady_residual_snapshots(
    residual_evaluator: ResidualEvaluator,
    full_state_directories: Iterable[str],
    state_filename: str,
    absolute_run_directory: str,
) -> np.ndarray:
    """
    Core algorithm that takes a residual_evaluator, a list of steady full-order model
    snapshot directories, and a snapshot_filename, and computes the corresponding
    residual snapshots.

    Args:
        residual_evaluator (ResidualEvaluator): steady residual evaluator we wish to use
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
        reduced_state = residual_evaluator.compute_reduced_states(
            full_model_dir + "/" + state_filename
        )

        # Set up corresponding directory
        run_directory = f"{run_directory_base}{index}"
        create_empty_dir(run_directory)

        # Evaluate residual
        residual_snapshot = residual_evaluator.evaluate_full_residuals(
            run_directory, full_model_dir, reduced_state, None
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
    residual_evaluator: TransientResidualEvaluator,
    full_state_directories: Iterable[str],
    state_filename: str,
    absolute_run_directory: str,
) -> np.ndarray:
    """
    Core algorithm that takes a residual_evaluator, a list of unsteady full-order model
    snapshot directories, and a snapshot_filename, and computes the corresponding
    residual snapshots.

    Args:
        residual_evaluator (TransientResidualEvaluator): transient residual evaluator we wish to use
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
        reduced_states = residual_evaluator.compute_reduced_states(
            full_model_dir + "/" + state_filename
        )
        times = residual_evaluator.get_times(
            full_model_dir + "/" + state_filename
        )

        # Set up corresponding directory
        run_directory = f"{run_directory_base}{index}"
        create_empty_dir(run_directory)

        # Evaluate residuals
        residual_snapshots = residual_evaluator.evaluate_full_residuals(
            run_directory, full_model_dir, reduced_states, None, times
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
