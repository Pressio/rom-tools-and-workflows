'''
Protocol for interfacing with external applications
'''

from pathlib import Path
from typing import Protocol


class ApplicationCoupler(Protocol):
    '''
    Partially explicit implementation
    '''
    def __init__(self) -> None:
        '''
        Initialize coupler
        '''
        pass

    def setup_directory(self, directory: Path) -> None:
        '''
        This function sets up a directory in which to execute the model.
        '''
        pass

    def run_model(self, run_directory: Path, parameter_values: dict) -> int:
        '''
        This function is called from a run directory. It needs to execute our
        model.  If the model runs succesfully, return 0.  If fails, return 1.
        '''
        pass
