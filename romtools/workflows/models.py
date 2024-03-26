'''
Protocol for interfacing with external applications
'''

from typing import Protocol
import numpy as np


class Model(Protocol):
    '''
    Baseline model protocol
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

        Examples would be setuping up input files, linking mesh files.

        Args:
          run_directory (str): Absolute path to run_directory.
          parameter_sample: Dictionary contatining parameter names and sample values

        '''
        pass

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        '''
        This function is called from the base directory. It needs to execute our
        model.  If the model runs succesfully, return 0.  If fails, return 1.

        Args:
          run_directory (str): Absolute path to run_directory.
          parameter_sample: Dictionary contatining parameter names and sample values

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

