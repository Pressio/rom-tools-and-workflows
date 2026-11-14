'''
Protocol for interfacing with external applications
'''

from typing import Protocol, Union, runtime_checkable
import numpy as np


@runtime_checkable
class JobHandle(Protocol):
    """Minimal handle returned by a non-blocking model."""

    def done(self) -> bool:
        """Return whether execution has completed."""
        ...

    def result(self) -> int:
        """Wait if needed and return the process exit status."""
        ...


class Model(Protocol):
    '''
    Baseline model protocol.

    Set ``blocking = False`` on models that return a JobHandle.
    The executor defaults to blocking when the attribute is absent.
    '''

    def __init__(self) -> None:
        '''
        Initialize coupler
        '''
        pass

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        '''
        This function is called from the base directory and is
        responsible for populating the run directory located at run_directory.

        Examples would be setting up input files, linking mesh files.

        Args:
          run_directory (str): Absolute path to run_directory.
          parameter_sample: Dictionary containing parameter names and sample values

        '''
        pass

    def run_model(self, run_directory: str, parameter_sample: dict) -> Union[int, JobHandle]:
        '''
        Execute the model. Blocking models return an integer exit status.
        Non-blocking models return a JobHandle.

        Args:
          run_directory (str): Absolute path to run_directory.
          parameter_sample: Dictionary containing parameter names and sample values

        '''
        pass


class QoiModel(Model, Protocol):
    '''
    Protocol for a model that has a return_qoi implementation
    '''

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        '''
        This function is called from a run directory
        AFTER run_model has been run
        '''
        pass


class QoiModelWithErrorEstimate(QoiModel, Protocol):
    '''
    Protocol for a model that has a return_qoi and compute_error_estimate
    '''

    def compute_error_estimate(self, run_directory: str, parameter_sample: dict) -> float:
        '''
        This function is called from a run directory
        AFTER run_model has been run
        '''
        pass
