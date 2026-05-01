import pytest
import os
import numpy as np
import time
import shutil

from types import SimpleNamespace

from romtools.workflows.sampling.sampling import run_sampling
from romtools.workflows.parameter_spaces import MonteCarloSampler, UniformParameterSpace, ConstParameterSpace
from romtools.hpc.connection import Result
from romtools.hpc.dispatcher import Dispatcher

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
        # such that the closer the ID is to 5, the less the task waits.
        # totally arbitrary choice.
        seconds_to_wait = abs(_get_run_id(run_dir) - 5) * 4
        time.sleep( seconds_to_wait )
        return 0


class MockQoiModel:
    def populate_run_directory(self, run_dir, parameter_sample):
        parameter_values = np.asarray(list(parameter_sample.values()), dtype=float)
        np.savez(f'{run_dir}/parameter_values.npz', parameter_values=parameter_values)

    def run_model(self, run_dir, parameter_sample):
        np.savetxt(f'{run_dir}/passed.txt', np.array([0]), '%i')
        return 0

    def compute_qoi(self, run_dir, parameter_sample):
        parameter_values = np.load(f'{run_dir}/parameter_values.npz')['parameter_values']
        return np.array([np.sum(parameter_values)])


class RemoteMockDispatcher:
    def __init__(self, remote_root):
        self.remote_root = str(remote_root)

    def _resolve_remote_path(self, remote_path):
        if os.path.isabs(remote_path):
            return remote_path
        return os.path.join(self.remote_root, remote_path)

    def create_remote_directory(self, remote_dir):
        os.makedirs(self._resolve_remote_path(remote_dir), exist_ok=True)

    def write_text(self, remote_path, content):
        resolved_path = self._resolve_remote_path(remote_path)
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        with open(resolved_path, 'w', encoding='utf-8') as output_file:
            output_file.write(content)

    def put(self, local_path, remote_path):
        resolved_path = self._resolve_remote_path(remote_path)
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        shutil.copy(local_path, resolved_path)

    def path_exists(self, remote_path):
        return os.path.exists(self._resolve_remote_path(remote_path))


class RemoteMockModel:
    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    def populate_run_directory(self, run_dir, parameter_sample):
        self.dispatcher.write_text(
            f'{run_dir}/parameter_values.txt',
            ' '.join(str(value) for value in parameter_sample.values()),
        )

    def run_model(self, run_dir, parameter_sample):
        self.dispatcher.write_text(f'{run_dir}/passed.txt', '0\n')
        return 0


class RecordingConnection:
    def __init__(self):
        self.host = 'remote-host'
        self.commands = []
        self.put_calls = []
        self.get_calls = []

    def run(self, command):
        self.commands.append(command)
        return Result('', '', 0)

    def put(self, local_path, remote_path):
        self.put_calls.append((local_path, remote_path))

    def get(self, remote_path, local_path):
        self.get_calls.append((remote_path, local_path))


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


@pytest.mark.mpi_skip
def test_sampler_dispatcher_uses_remote_filesystem(tmp_path):
    remote_root = tmp_path / 'remote-root'
    dispatcher = RemoteMockDispatcher(remote_root)
    parameter_space = ConstParameterSpace(['u', 'v'], [1.0, 2.0])
    model = RemoteMockModel(dispatcher)

    run_sampling(
        model,
        parameter_space,
        absolute_sampling_directory='samples_01',
        evaluation_concurrency=1,
        number_of_samples=2,
        dry_run=False,
        overwrite=False,
        dispatcher=dispatcher,
    )
    first_timestamp = os.stat(remote_root / 'samples_01' / 'run_0' / 'passed.txt').st_mtime

    run_sampling(
        model,
        parameter_space,
        absolute_sampling_directory='samples_01',
        evaluation_concurrency=1,
        number_of_samples=2,
        dry_run=False,
        overwrite=False,
        dispatcher=dispatcher,
    )

    assert os.path.exists(remote_root / 'samples_01' / 'sample_parameters.txt')
    assert os.path.exists(remote_root / 'samples_01' / 'sampling_stats.npz')
    assert os.path.exists(remote_root / 'samples_01' / 'run_0' / 'parameter_values.txt')
    assert os.stat(remote_root / 'samples_01' / 'run_0' / 'passed.txt').st_mtime == first_timestamp


def test_dispatcher_resolves_relative_remote_paths(tmp_path):
    dispatcher = object.__new__(Dispatcher)
    dispatcher.logger = SimpleNamespace(log=lambda *args, **kwargs: None)
    dispatcher.conn = RecordingConnection()
    dispatcher.config = SimpleNamespace(remote_root='/remote/root')

    dispatcher.create_remote_directory('samples_01/run_0')
    dispatcher.write_text('samples_01/output.txt', 'value\n')

    local_file = tmp_path / 'stats.npz'
    local_file.write_bytes(b'npz')
    dispatcher.put(str(local_file), 'samples_01/sampling_stats.npz')
    dispatcher.get('samples_01/sampling_stats.npz', str(tmp_path / 'downloaded.npz'))
    assert dispatcher.path_exists('samples_01/output.txt')

    assert dispatcher.conn.commands[0] == 'mkdir -p /remote/root/samples_01/run_0'
    assert dispatcher.conn.commands[1].startswith("cat > /remote/root/samples_01/output.txt << '")
    assert dispatcher.conn.put_calls == [
        (str(local_file), '/remote/root/samples_01/sampling_stats.npz')
    ]
    assert dispatcher.conn.get_calls == [
        ('/remote/root/samples_01/sampling_stats.npz', str(tmp_path / 'downloaded.npz'))
    ]
    assert dispatcher.conn.commands[2] == 'test -e /remote/root/samples_01/output.txt'

if __name__ == "__main__":
    test_sampler()
    test_sampler_dry_run()
    test_sampler_overwrite()
