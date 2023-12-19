import pytest
import os
import numpy as np
from romtools.workflows.greedy.greedy_coupler_base import GreedyCouplerBase
from romtools.workflows.greedy.run_greedy import run_greedy
from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.sampling.\
    sampling_coupler_base import SamplingCouplerBase


class ConcreteSampler(SamplingCouplerBase):
    def __init__(self,
                 template_directory,
                 template_file,
                 workDir=None,
                 sol_directory_basename='run'):

        super().__init__(template_directory=template_directory,
                         template_input_file=template_file,
                         base_directory=workDir,
                         sol_directory_basename=sol_directory_basename)

        self.myParameterSpace = UniformParameterSpace(['u', 'v', 'w'],
                                                      np.array([0, 1, 2]),
                                                      np.array([1, 2, 3]))
        self.counter_ = 0
        self.template_file = template_file

    def set_parameters_in_input(self, filename, parameter_sample):
        file = np.genfromtxt(self.template_file)
        np.savetxt(self.template_file, [self.counter_])
        self.counter_ += 1

    def run_model(self, filename, parameter_values):
        return 0

    def get_parameter_space(self):
        return self.myParameterSpace


class ConcreteGreedyCoupler(GreedyCouplerBase):
    def __init__(self, template_directory,
                 template_fom_file,
                 template_rom_file,
                 workDir=None):

        self.my_parameter_space = UniformParameterSpace(['u', 'v', 'w'],
                                                        np.array([0, 1, 2]),
                                                        np.array([1, 2, 3]))

        rom_coupler = ConcreteSampler(template_directory=template_directory,
                                      template_file=template_rom_file,
                                      workDir=f'{workDir}/rom')
        fom_coupler = ConcreteSampler(template_directory=template_directory,
                                      template_file=template_fom_file,
                                      workDir=f'{workDir}/fom')
        super().__init__(rom_coupler=rom_coupler,
                         fom_coupler=fom_coupler,
                         base_directory=workDir)

        self.counter_ = 0
        self.template_fom_file = template_fom_file

        my_error_estimates = np.array([1., 2., 3., 1.5, 4.])  # First iteration, should identify 5th entry as the sample to run
        my_error_estimates_iteration_2 = np.array([0.9, 0.4, 0.6])
        my_error_estimates_iteration_3 = np.array([0.09, 0.1, 0.06])
        my_error_estimates_iteration_4 = np.array([1e-7, 1e-6, 1e-5])
        my_error_estimates = np.append(my_error_estimates,
                                       my_error_estimates_iteration_2)
        my_error_estimates = np.append(my_error_estimates,
                                       my_error_estimates_iteration_3)
        self.my_error_estimates_ = np.append(my_error_estimates,
                                             my_error_estimates_iteration_4)
        self.error_estimate_counter_ = 0

        self.my_errors_ = np.array([1., 1., 0.4, 0.09, 0.01, 1e-6])
        self.my_errors_counter_ = 0

    def compute_error(self, case_num):
        error = self.my_errors_[self.my_errors_counter_]
        self.my_errors_counter_ += 1
        return error

    def compute_error_indicator(self):
        error_estimate = self.my_error_estimates_[self.error_estimate_counter_]
        self.error_estimate_counter_ += 1
        return error_estimate

    def compute_qoi(self):
        return 0

    def run_rom(self, filename, parameter_values):
        pass

    def run_fom(self, filename, parameter_values):
        np.savetxt('fom_succesful.dat', parameter_values)

    def create_trial_space(self, training_sample_indices):
        pass

    def get_parameter_space(self):
        return self.my_parameter_space


@pytest.mark.mpi_skip
def test_greedy_coupler_builder():
    my_dir = os.path.realpath(os.path.dirname(__file__))
    ConcreteGreedyCoupler(my_dir + '/templates/',
                          'test_template.dat',
                          'test_template.dat')


@pytest.mark.mpi_skip
def test_greedy(tmp_path):
    # see https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html for more info
    #   about tmp_path
    wdir = str(tmp_path)  # does not like posixpaths
    print('\n', wdir)

    my_dir = os.path.realpath(os.path.dirname(__file__))
    my_greedy_coupler = ConcreteGreedyCoupler(my_dir + '/templates/',
                                              'test_template.dat',
                                              'test_template.dat',
                                              workDir=wdir)
    init_sample_size = 5
    run_greedy(my_greedy_coupler, 1e-5, init_sample_size)
    # First greedy pass
    foms_samples_run = [0, 1, 4, 2, 5]
    foms_samples_not_run = [3, 6, 7]

    for sample in foms_samples_run:
        assert os.path.isfile(f'{wdir}/fom/run{sample}/fom_succesful.dat'), sample

    for sample in foms_samples_not_run:
        assert not os.path.isfile(f'{wdir}/fom/run{sample}/fom_succesful.dat'), sample

    greedy_output = np.load(f'{wdir}/rom/greedy_stats.npz')
    assert np.allclose(greedy_output['max_error_indicators'],
                       np.array([4., 0.9, 0.1]))
    assert np.allclose(greedy_output['training_samples'],
                       np.array([0, 1, 4, 2, 5]))
    assert np.allclose(greedy_output['qoi_errors'],
                       np.array([0.4, 0.09, 0.01]))

    # Test parameter_samples output in greedy_status.log
    total_sample_size = len(foms_samples_not_run + foms_samples_run)
    log_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(my_dir))))

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
        assert parameter_samples_col_dimensions[i] == len(my_greedy_coupler.get_parameter_space().get_names())


if __name__ == "__main__":
    test_greedy_coupler_builder()
    test_greedy('.')
