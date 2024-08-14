'''
Protocol for building a model
This is mainly used for constructing data-driven models in 
iterative workflows like greedy 
'''

from pathlib import Path
from typing import Protocol, List
from romtools.workflows.models import *

class ModelBuilder(Protocol):
    '''
    Main protocol for a ModelBuilder.

    Methods:
    '''
    def __init__(self):
        pass

    def build_from_training_dirs(self,offline_data_dir: str, training_data_dirs: List[str]) -> Model:
        pass


class QoiModelBuilder(Protocol):
    '''
    Main protocol for a QoiModelBuilder.

    Methods:
    '''

    def __init__(self):
        pass

    def build_from_training_dirs(self,offline_data_dir: str, training_data_dirs: List[str]) -> QoiModel:
        pass


class QoiModelWithErrorEstimateBuilder(Protocol):
    '''
    Main protocol for a QoiModelWithErrorEstimateBuilder.

    Methods:
    '''

    def __init__(self):
        pass

    def build_from_training_dirs(self,offline_data_dir: str, training_data_dirs: List[str]) -> QoiModelWithErrorEstimate:
        return QoiModelWithErrorEstimate


