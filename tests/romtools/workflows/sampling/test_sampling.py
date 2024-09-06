import pytest
import os
import numpy as np

from romtools.workflows.sampling.sampling import run_sampling
from romtools.workflows.parameter_spaces import MonteCarloSampler, UniformParameterSpace


class MockModel:
    def __init__(self):
        pass

    def populate_run_directory(self, run_dir, parameter_sample):
        parameter_values = np.zeros(0)
        for parameter_name in list(parameter_sample.keys()):
            parameter_values = np.append(parameter_values, parameter_sample[parameter_name])
        np.savez(f'{run_dir}/parameter_values.npz', parameter_values=parameter_values)

    def run_model(self, run_dir, parameter_sample):
        params_input = np.load(f'{run_dir}/parameter_values.npz')['parameter_values']
        for i in range(0, len(parameter_sample)):
            parameter_name = list(parameter_sample.keys())[i]
            assert params_input[i] == parameter_sample[parameter_name]
        np.savetxt(f'{run_dir}/passed.txt', np.array([0]), '%i')
        return 0


def run_sampler(tmp_path, dry_run=False, overwrite=True):
    print("running sampler with dry_run: ", dry_run)
    my_parameter_space = UniformParameterSpace(['u', 'v', 'w'],
                                               np.array([0, 1, 2]),
                                               np.array([1, 2, 3]),
                                               sampler=MonteCarloSampler)
    my_model = MockModel()
    run_directories = run_sampling(my_model, my_parameter_space,
                                   absolute_sampling_directory=tmp_path,
                                   number_of_samples=10, dry_run=dry_run,
                                   overwrite=overwrite)

    assert(len(run_directories)==10)

    for i in range(0, 10):
        run_dir = os.path.join(tmp_path, f'run_{i}')
        assert os.path.isdir(run_dir)
        if not dry_run:
            data = int(np.genfromtxt(os.path.join(run_dir, 'passed.txt')))
            assert data == 0

    print(os.listdir(tmp_path))
    print(found_passed_file)
    print(dry_run)
    print("passed.txt" in os.listdir(tmp_path) != dry_run)
    assert "passed.txt" in os.listdir(tmp_path) != dry_run
    assert "sampling_stats.npz" in os.listdir(tmp_path) != dry_run

@pytest.mark.mpi_skip
def test_sampler(tmp_path):
    # see https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html for more info
    # remove_existing_files(tmp_path)
    run_sampler(tmp_path, dry_run=False)

@pytest.mark.mpi_skip
def test_sampler_dry_run(tmp_path):
    run_sampler(tmp_path, dry_run=True)

@pytest.mark.mpi_skip
def test_sampler_ovewrite(tmp_path):
    run_sampler(tmp_path, overwrite=True)
    run_sampler(tmp_path, overwrite=False)


if __name__ == "__main__":
    test_sampler()
    test_sampler_dry_run()
