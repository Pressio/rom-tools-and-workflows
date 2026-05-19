import pytest
import os
import numpy as np
import time

from romtools.workflows.sampling.sampling import run_sampling
from romtools.workflows.parameter_spaces import MonteCarloSampler, UniformParameterSpace, ConstParameterSpace

def _get_run_id(run_dir):
    return int(run_dir.split('_')[-1])

class MockModel:
    def __init__(self):
        pass

    def populate_run_directory(self, run_dir, parameter_sample, dispatcher=None):
        parameter_values = np.zeros(0)
        for parameter_name in list(parameter_sample.keys()):
            parameter_values = np.append(parameter_values, parameter_sample[parameter_name])
        np.savez(f'{run_dir}/parameter_values.npz', parameter_values=parameter_values)

    def run_model(self, run_dir, parameter_sample, dispatcher=None):
        print("running model in ", run_dir)
        params_input = np.load(f'{run_dir}/parameter_values.npz')['parameter_values']
        for i in range(0, len(parameter_sample)):
            parameter_name = list(parameter_sample.keys())[i]
            assert params_input[i] == parameter_sample[parameter_name]
        np.savetxt(f'{run_dir}/passed.txt', np.array([0]), '%i')

        # add artificial lag centered around run_id=5
        # such that the closer the ID is to 5, the less the task waits.
        # totally arbitrary choice.
        seconds_to_wait = abs(_get_run_id(run_dir) - 5) * 4
        time.sleep( seconds_to_wait )
        return 0


class MockQoiModel:
    def populate_run_directory(self, run_dir, parameter_sample, dispatcher=None):
        parameter_values = np.asarray(list(parameter_sample.values()), dtype=float)
        np.savez(f'{run_dir}/parameter_values.npz', parameter_values=parameter_values)

    def run_model(self, run_dir, parameter_sample, dispatcher=None):
        np.savetxt(f'{run_dir}/passed.txt', np.array([0]), '%i')
        return 0

    def compute_qoi(self, run_dir, parameter_sample, dispatcher=None):
        parameter_values = np.load(f'{run_dir}/parameter_values.npz')['parameter_values']
        return np.array([np.sum(parameter_values)])


def run_sampler(tmp_path, dry_run=False, overwrite=True):
    my_parameter_space = UniformParameterSpace(['u', 'v', 'w'],
                                               np.array([0, 1, 2]),
                                               np.array([1, 2, 3]),
                                               sampler=MonteCarloSampler)
    my_model = MockModel()
    run_directories = run_sampling(my_model, my_parameter_space,
                                   absolute_sampling_directory=tmp_path,
                                   evaluation_concurrency=2,
                                   number_of_samples=10,dry_run=dry_run,
                                   overwrite=overwrite)
    assert(len(run_directories)==10)

    timestamps = []
    for i in range(0, 10):
        run_dir = os.path.join(tmp_path, f'run_{i}')
        assert os.path.isdir(run_dir)
        assert ("passed.txt" in os.listdir(run_dir)) != dry_run
        if not dry_run:
            timestamps.append(os.stat(os.path.join(run_dir, "passed.txt")).st_mtime)

    assert ("sampling_stats.npz" in os.listdir(tmp_path)) != dry_run
    if not dry_run:
        stats = np.load(os.path.join(tmp_path, "sampling_stats.npz"))
        assert "run_times" in stats
        assert "qoi_mean" not in stats

    # Return the time that each passed.txt was last modified
    return timestamps


def run_sampler_hetero(tmp_path):
    my_parameter_space = ConstParameterSpace(["int", "float", "string"],
                                             [1, 2.1, "test_string"])
    my_model = MockModel()
    run_directories = run_sampling(my_model, my_parameter_space,
                                   absolute_sampling_directory=tmp_path,
                                   evaluation_concurrency=2,
                                   number_of_samples=4, dry_run=True,
                                   overwrite=True)
    assert(len(run_directories)==4)


@pytest.mark.mpi_skip
def test_sampler(tmp_path):
    # see https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html for more info
    print('\n', tmp_path)
    run_sampler(tmp_path, dry_run=False)

@pytest.mark.mpi_skip
def test_sampler_hetero(tmp_path):
    run_sampler_hetero(tmp_path)

@pytest.mark.mpi_skip
def test_sampler_dry_run(tmp_path):
    run_sampler(tmp_path, dry_run=True)

@pytest.mark.mpi_skip
def test_sampler_overwrite(tmp_path):
    initial_timestamps = run_sampler(tmp_path)
    exp_initial_timestamps = run_sampler(tmp_path, overwrite=False)
    exp_new_timestamps = run_sampler(tmp_path, overwrite=True)

    for i in range(10):
        assert initial_timestamps[i] == exp_initial_timestamps[i]
        assert exp_initial_timestamps[i] != exp_new_timestamps[i]


@pytest.mark.mpi_skip
def test_sampler_qoi_stats(tmp_path):
    parameter_space = ConstParameterSpace(['u', 'v'], [1.0, 2.0])
    model = MockQoiModel()
    run_sampling(
        model,
        parameter_space,
        absolute_sampling_directory=tmp_path,
        evaluation_concurrency=1,
        number_of_samples=4,
        dry_run=False,
        overwrite=True,
    )

    stats = np.load(os.path.join(tmp_path, "sampling_stats.npz"))
    assert "qoi_values" in stats
    assert "qoi_mean" in stats
    assert "qoi_std" in stats
    assert "qoi_min" in stats
    assert "qoi_max" in stats
    assert "qoi_num_samples" in stats
    assert stats["qoi_values"].shape == (4, 1)
    np.testing.assert_allclose(stats["qoi_values"], 3.0)
    np.testing.assert_allclose(stats["qoi_mean"], np.array([3.0]))
    np.testing.assert_allclose(stats["qoi_std"], np.array([0.0]))
    np.testing.assert_allclose(stats["qoi_min"], np.array([3.0]))
    np.testing.assert_allclose(stats["qoi_max"], np.array([3.0]))
    np.testing.assert_array_equal(stats["qoi_num_samples"], np.array([4]))

if __name__ == "__main__":
    test_sampler()
    test_sampler_dry_run()
    test_sampler_overwrite()
