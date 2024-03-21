import pytest
import os
import numpy as np

from romtools.workflows.sampling.sampling import run_sampling
from romtools.workflows.sampling.\
    sampling_coupler_base import SamplingCouplerBase
from romtools.workflows.parameter_spaces import UniformParameterSpace


class MockModel:
    def __init__(self):
        pass

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


@pytest.mark.mpi_skip
def test_sampler(tmp_path):
    # see https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html for more info
    print('\n', tmp_path)

    my_parameter_space = UniformParameterSpace(['u', 'v', 'w'],
                                               np.array([0, 1, 2]),
                                               np.array([1, 2, 3]))
    my_model = MockModel()
    run_sampling(my_model, my_parameter_space,
                 run_directory_prefix=f'{tmp_path}/run_',
                 number_of_samples=10)

    for i in range(0, 10):
        assert os.path.isdir(f'{tmp_path}/run_' + str(i))
        data = int(np.genfromtxt(f'{tmp_path}/run_{i}/passed.txt'))
        assert data == 0
    assert os.path.isfile(f'{tmp_path}/sampling_stats.npz')


if __name__ == "__main__":
    test_sampler('.')
