import pytest
import os
import numpy as np
from romtools.workflows.models import *
from romtools.workflows.model_builders import *
from romtools.workflows.greedy.run_greedy import run_greedy
from romtools.workflows.parameter_spaces import MonteCarloSampler, UniformParameterSpace


class MockQoiModel:
    def __init__(self):
        self.my_qois_ = np.array([1.,1.,1.,1.,1.,1.,1.])
        self.counter_ = 0

    def populate_run_directory(self, run_dir,parameter_sample):
        os.chdir(run_dir)
        parameter_values = np.zeros(0)
        for parameter_name in list(parameter_sample.keys()):
            parameter_values = np.append(parameter_values,parameter_sample[parameter_name])
        np.savez('parameter_values.npz',parameter_values=parameter_values)

    def run_model(self, run_dir, parameter_sample):
        print(os.getcwd())
        os.chdir(run_dir)
        params_input = np.load('parameter_values.npz')['parameter_values']
        for i in range(0,len(parameter_sample)):
          parameter_name = list(parameter_sample.keys())[i]
          assert(params_input[i] == parameter_sample[parameter_name])
        np.savetxt('fom_succesful.dat',np.array([0]),'%i')
        return 0

    def compute_qoi(self, run_dir, parameter_sample):
        self.counter_ += 1
        return self.my_qois_[self.counter_ - 1]




class MockQoiModelWithErrorEstimateBuilder:
    def __init__(self):
        self.counter_ = 0
        n_iterations = 4

        self.my_error_estimates_ = [None]*n_iterations
        self.my_error_estimates_[0] = np.array([1., 2., 4.]) # First iteration, should identify 5th entry as the sample to run
        self.my_error_estimates_[1] = np.array([0.9, 0.4, 0.6])
        self.my_error_estimates_[2] = np.array([0.09, 0.1, 0.06])
        self.my_error_estimates_[3] = np.array([1e-7, 1e-6, 1e-5])

        self.my_qois_ = np.array([1.5,1.4,1.3,1.2])

    def build_from_training_dirs(self,offline_data_dir, training_data_dirs):
        rom_model = MockQoiModelWithErrorEstimate(self.my_error_estimates_[self.counter_],self.my_qois_[self.counter_])
        print(offline_data_dir)
        np.savetxt(f'{offline_data_dir}/offline_data.dat',np.array([0]),'%i')
        self.counter_ += 1
        return rom_model


class MockQoiModelWithErrorEstimate:
    def __init__(self,my_error_estimates,my_qoi):
        self.counter = 0
        self.my_error_estimates_ = my_error_estimates
        self.my_qoi_ = my_qoi

    def populate_run_directory(self, run_dir,parameter_sample):
        os.chdir(run_dir)
        parameter_values = np.zeros(0)
        for parameter_name in list(parameter_sample.keys()):
            parameter_values = np.append(parameter_values,parameter_sample[parameter_name])
        np.savez('parameter_values.npz',parameter_values=parameter_values)

    def run_model(self, run_dir, parameter_sample):
        os.chdir(run_dir)
        params_input = np.load('parameter_values.npz')['parameter_values']
        for i in range(0,len(parameter_sample)):
          parameter_name = list(parameter_sample.keys())[i]
          assert(params_input[i] == parameter_sample[parameter_name])
        np.savetxt('passed.txt',np.array([0]),'%i')
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> float:
        return self.my_qoi_

    def compute_error_estimate(self, run_directory: str, parameter_sample: dict) -> float:
        self.counter += 1
        return self.my_error_estimates_[self.counter-1]




@pytest.mark.mpi_skip
def test_greedy(tmp_path):
    # see https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html for more info
    #   about tmp_path
    wdir = str(tmp_path)  # does not like posixpaths
    print('\n', wdir)

    my_dir = os.path.realpath(os.path.dirname(__file__))

    init_sample_size = 5

    QoiModel = MockQoiModel()
    RomModelBuilder = MockQoiModelWithErrorEstimateBuilder()

    my_parameter_space = UniformParameterSpace(['u', 'v', 'w'],
                                            np.array([0, 1, 2]),
                                            np.array([1, 2, 3]),
                                            sampler=MonteCarloSampler)


    run_greedy(QoiModel,RomModelBuilder,my_parameter_space,wdir, 1e-5,5)

    # First greedy pass
    foms_samples_run = [0, 1, 4, 2, 5]
    foms_samples_not_run = [3, 6, 7]


    for sample in foms_samples_run:
        assert os.path.isfile(f'{wdir}/fom/run_{sample}/fom_succesful.dat'), sample

    for sample in foms_samples_not_run:
        assert not os.path.isfile(f'{wdir}/fom/run_{sample}/fom_succesful.dat'), sample

    greedy_output = np.load(f'{wdir}/greedy_stats.npz')
    assert np.allclose(greedy_output['max_error_indicators'],
                       np.array([4., 0.9, 0.1]))
    assert np.allclose(greedy_output['training_samples'],
                       np.array([0, 1, 4, 2, 5]))
    assert np.allclose(greedy_output['qoi_errors'],
                       np.array([0.5, 0.4, 0.3]))

    for i in range(0,4):
         assert os.path.isfile(f'{wdir}/rom_iteration_{i}/offline_data/offline_data.dat')
    # Test parameter_samples output in greedy_status.log
    total_sample_size = len(foms_samples_not_run + foms_samples_run)
    log_dir = wdir

    # Initialize variables
    in_parameter_samples_block = False
    parameter_samples_row_dimensions = []
    parameter_samples_col_dimensions = []
    row_count = 0
    col_count = 0

    # Find dimensions of parameter_samples arrays
    with open(os.path.join(log_dir, "greedy_status.log"), 'r', encoding="utf-8") as greedy_log:
        for line in greedy_log:
            if in_parameter_samples_block:
                if line.startswith("    Running"):
                    # Check for end of parameter_samples array
                    in_parameter_samples_block = False
                    parameter_samples_row_dimensions.append(row_count)
                    parameter_samples_col_dimensions.append(col_count)
                    row_count = 0
                else:
                    # Count rows & columns in given parameter_samples array
                    row_count += 1
                    col_count = line.count('.')
            elif "Parameter samples:" in line:
                in_parameter_samples_block = True

    # Assert correct number of arrays with correct number of rows & columns
    assert len(parameter_samples_row_dimensions) == len(parameter_samples_col_dimensions)
    assert len(parameter_samples_row_dimensions) == total_sample_size - init_sample_size + 1
    for i, _ in enumerate(parameter_samples_row_dimensions):
        assert parameter_samples_row_dimensions[i] == init_sample_size + i
        assert parameter_samples_col_dimensions[i] == len(my_parameter_space.get_names())

if __name__ == "__main__":
    test_greedy(os.getcwd() + '/greedy_test_tmp/')
