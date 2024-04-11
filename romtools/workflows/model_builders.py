'''
Protocol for building a model
This is mainly used for constructing data-driven models in 
iterative workflows like greedy 
'''

from pathlib import Path
from typing import Protocol
from romtools.workflows.models import *

class ModelBuilder(Protocol):
    def __init__(self):
        pass

    def build_from_training_dirs(self,offline_data_dir: str, training_data_dirs: list[str]) -> Model:
        pass


class QoiModelBuilder(Protocol):
    def __init__(self):
        pass

    def build_from_training_dirs(self,offline_data_dir: str, training_data_dirs: list[str]) -> QoiModel:
        pass


class QoiModelWithErrorEstimateBuilder(Protocol):
    def __init__(self):
        pass

    def build_from_training_dirs(self,offline_data_dir: str, training_data_dirs: list[str]) -> QoiModelWithErrorEstimate:
        return QoiModelWithErrorEstimate


