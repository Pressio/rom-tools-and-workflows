import pytest
import os
import numpy as np
import time

from romtools.workflows.sampling.sampling import run_sampling
from romtools.workflows.parameter_spaces import MonteCarloSampler, UniformParameterSpace

def _get_run_id(run_dir):
    return int(run_dir.split('_')[-1])

class MockModel:
    def __init__(self):
        pass

    def populate_run_directory(self, run_dir, parameter_sample):
        parameter_values = np.zeros(0)
        for parameter_name in list(parameter_sample.keys()):
            parameter_values = np.append(parameter_values, parameter_sample[parameter_name])
        np.savez(f'{run_dir}/parameter_values.npz', parameter_values=parameter_values)

    def run_model(self, run_dir, parameter_sample):
        print("running model in ", run_dir)
        params_input = np.load(f'{run_dir}/parameter_values.npz')['parameter_values']
        for i in range(0, len(parameter_sample)):
            parameter_name = list(parameter_sample.keys())[i]
            assert params_input[i] == parameter_sample[parameter_name]
        np.savetxt(f'{run_dir}/passed.txt', np.array([0]), '%i')

        # add artificial lag centered around run_id=5
        # such that all runs with id close to 5 wait less
        # totally arbitrary choice.
        seconds_to_wait = abs(_get_run_id(run_dir) - 5) * 5
        time.sleep( seconds_to_wait )

        return 0


@pytest.mark.mpi_skip
def test_sampler(tmp_path):
    # see https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html for more info
    print('\n', tmp_path)

    my_parameter_space = UniformParameterSpace(['u', 'v', 'w'],
                                               np.array([0, 1, 2]),
                                               np.array([1, 2, 3]),
                                               sampler=MonteCarloSampler)
    my_model = MockModel()
    run_directories = run_sampling(my_model, my_parameter_space,
                                   absolute_sampling_directory=tmp_path,
                                   evaluation_concurrency=4,
                                   number_of_samples=10)
    assert(len(run_directories)==10)

    for i in range(0, 10):
        assert os.path.isdir(f'{tmp_path}/run_' + str(i))
        data = int(np.genfromtxt(f'{tmp_path}/run_{i}/passed.txt'))
        assert data == 0
    # assert os.path.isfile(f'{tmp_path}/sampling_stats.npz')


if __name__ == "__main__":
    test_sampler()
