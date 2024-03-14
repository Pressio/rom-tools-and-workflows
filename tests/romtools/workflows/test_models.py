import numpy as np
from romtools.workflows.models import *


class MockModel:
    def __init__(self):
        pass

    def populate_run_directory(self, run_dir,parameter_sample):
        pass 

    def run_model(self, run_dir, parameter_sample):
        return 0


class MockQoiModel:
    def __init__(self):
        pass

    def populate_run_directory(self, run_dir,parameter_sample):
        pass 

    def run_model(self, run_dir, parameter_sample):
        return 0

    def compute_qoi(self, run_dir, parameter_sample):
        return np.zeros(0)


class MockQoiModelWithErrorEstimate:
    def __init__(self):
        pass

    def populate_run_directory(self, run_dir,parameter_sample):
        pass 

    def run_model(self, run_dir, parameter_sample):
        return 0

    def compute_qoi(self, run_dir, parameter_sample):
        return 0

    def compute_error_estimate(self, run_dir, parameter_sample):
        return np.zeros(0)


def test_model():
    my_model = MockModel()
    run_dir = 'test'
    parameter_sample = None
    my_model.populate_run_directory(run_dir,parameter_sample)
    my_model.run_model(run_dir,parameter_sample)


def test_qoi_model():
    my_model = MockQoiModel()
    run_dir = 'test'
    parameter_sample = None
    my_model.populate_run_directory(run_dir,parameter_sample)
    my_model.run_model(run_dir,parameter_sample)
    my_model.compute_qoi(run_dir,parameter_sample)


def test_qoi_model_w_error_estimate():
    my_model = MockQoiModelWithErrorEstimate()
    run_dir = 'test'
    parameter_sample = None
    my_model.populate_run_directory(run_dir,parameter_sample)
    my_model.run_model(run_dir,parameter_sample)
    my_model.compute_qoi(run_dir,parameter_sample)
    my_model.compute_error_estimate(run_dir,parameter_sample)



if __name__ == "__main__":
    test_model()
    test_qoi_model()
    test_qoi_model_w_error_estimate()
