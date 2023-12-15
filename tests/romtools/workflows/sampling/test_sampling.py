import pytest
import os
import numpy as np

from romtools.workflows.sampling.sampling import run_sampling
from romtools.workflows.sampling.sampling_coupler_base import SamplingCouplerBase
from romtools.workflows.parameter_spaces import UniformParameterSpace


class ConcreteSampler(SamplingCouplerBase):
    def __init__(self,
                 template_directory,
                 template_file,
                 workDir=None,
                 sol_directory_base_name='run'):

        super().__init__(template_directory=template_directory,
                         template_input_file=template_file,
                         work_directory=workDir,
                         sol_directory_base_name=sol_directory_base_name)

        self.myParameterSpace = UniformParameterSpace(['u', 'v', 'w'],
                                                      np.array([0, 1, 2]),
                                                      np.array([1, 2, 3]))
        self.counter_ = 0
        self.template_file = template_file

    def set_parameters_in_input(self, filename, parameter_sample):
        file = np.genfromtxt(self.template_file)
        np.savetxt(self.template_file, [self.counter_])
        self.counter_ += 1

    def run_model(self, input_filename, parameter_values):
        return 0

    def get_parameter_space(self):
        return self.myParameterSpace


@pytest.mark.mpi_skip
def test_sampler_builder():
    my_dir = os.path.realpath(os.path.dirname(__file__))
    ConcreteSampler(my_dir + '/templates/', 'test_template.dat')


@pytest.mark.mpi_skip
def test_sampler(tmp_path):
    # see https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html for more info
    wdir = str(tmp_path)  # SamplingCouplerBase does not like posixpaths
    print('\n', wdir)
    my_dir = os.path.realpath(os.path.dirname(__file__))
    my_sampler = ConcreteSampler(my_dir + '/templates/',
                                 'test_template.dat',
                                 workDir=wdir)
    run_sampling(my_sampler, 10)
    for i in range(0, 10):
        assert os.path.isdir(wdir + '/work/run' + str(i))
        data = int(np.genfromtxt(f'{wdir}/work/run{i}/test_template.dat'))
        assert data == i


if __name__ == "__main__":
    test_sampler_builder()
    test_sampler()
