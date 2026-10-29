import pytest
import os
import numpy as np
from romtools.workflows.models import *
from romtools.workflows.model_builders import *
from romtools.workflows.greedy.run_greedy import run_greedy, QoIvsErrorIndicatorRegressor
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
        self.training_calls = []
        n_iterations = 4

        self.my_error_estimates_ = [None]*n_iterations
        self.my_error_estimates_[0] = np.array([1., 2., 4.]) # First iteration, should identify 5th entry as the sample to run
        self.my_error_estimates_[1] = np.array([0.9, 0.4, 0.6])
        self.my_error_estimates_[2] = np.array([0.09, 0.1, 0.06])
        self.my_error_estimates_[3] = np.array([1e-7, 1e-6, 1e-5])

        self.my_qois_ = np.array([1.5,1.4,1.3,1.2])

    def build_from_training_dirs(self, offline_data_dir, training_data_dirs,
                                 training_parameters, training_qois):
        self.training_calls.append((
            list(training_data_dirs), training_parameters.copy(), training_qois.copy(),
        ))
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
def test_greedy_error_regressor():
    calibrated_reg = QoIvsErrorIndicatorRegressor(calibrated_error=True)
    standard_reg = QoIvsErrorIndicatorRegressor(calibrated_error=False)

    x = np.random.rand(5)
    y = np.random.rand(5)

    calibrated_reg.fit(x, y)
    calibrated_prediction = calibrated_reg.predict(x)
    exp_prediction = np.mean(y) / np.mean(x) * x

    standard_reg.fit(x, y)
    standard_prediction = standard_reg.predict(x)

    assert np.allclose(calibrated_prediction, exp_prediction)
    assert np.allclose(standard_prediction, x)

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

    run_greedy(QoiModel,RomModelBuilder,my_parameter_space,wdir, 1e-5, 5)

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
    assert [len(call[0]) for call in RomModelBuilder.training_calls] == [2, 3, 4, 5]
    for dirs, parameters, qois in RomModelBuilder.training_calls:
        assert parameters.shape == (len(dirs), 3)
        np.testing.assert_array_equal(qois, np.ones((len(dirs), 1)))
        for directory, parameter in zip(dirs, parameters):
            with np.load(os.path.join(directory, 'parameter_values.npz')) as data:
                np.testing.assert_array_equal(parameter, data['parameter_values'])

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
    test_greedy_error_regressor()


class ParameterQoiModel:
    """QoIs vary by sample; array returns reuse a mutable model-owned buffer."""

    def __init__(self, qoi_shape, factor=1.0):
        self.qoi_shape = qoi_shape
        self.factor = factor
        self.buffer = np.zeros(qoi_shape, dtype=np.result_type(factor, float)) if qoi_shape else None
        self.compute_calls = 0

    def populate_run_directory(self, directory, parameters):
        np.savez(os.path.join(directory, 'parameters.npz'), values=list(parameters.values()))

    def run_model(self, directory, parameters):
        return 0

    def compute_qoi(self, directory, parameters):
        self.compute_calls += 1
        base = (1 + sum(parameters.values())) * self.factor
        if self.qoi_shape:
            self.buffer[...] = base + np.arange(self.buffer.size).reshape(self.qoi_shape)
            value = self.buffer
        else:
            value = float(base)
        np.savez(os.path.join(directory, 'qoi.npz'), values=value)
        return value


class DataCheckingBuilder(MockQoiModelWithErrorEstimateBuilder):
    def __init__(self, mutate_inputs):
        super().__init__()
        self.mutate_inputs = mutate_inputs

    def build_from_training_dirs(self, offline, directories, parameters, qois):
        assert parameters.shape == (len(directories), 3)
        assert qois.ndim == 2 and qois.shape[0] == len(directories)
        for directory, row, qoi in zip(directories, parameters, qois):
            with np.load(os.path.join(directory, 'parameters.npz')) as saved:
                np.testing.assert_array_equal(row, saved['values'])
            with np.load(os.path.join(directory, 'qoi.npz')) as saved:
                np.testing.assert_array_equal(qoi, saved['values'].reshape(-1))
        rom = super().build_from_training_dirs(offline, directories, parameters, qois)
        rom.my_qoi_ = np.zeros(qois.shape[1])
        if self.mutate_inputs:
            parameters[:] = -999
            qois[:] = -999
            directories.clear()
        return rom


@pytest.mark.mpi_skip
@pytest.mark.parametrize('qoi_shape,factor', [((), 1.0), ((2,), 1.0), ((2, 2), 1.0), ((2,), 1 + 2j)])
@pytest.mark.parametrize('mutate_inputs', [False, True])
def test_greedy_passes_aligned_accumulated_training_data(tmp_path, qoi_shape, factor, mutate_inputs):
    fom = ParameterQoiModel(qoi_shape, factor)
    builder = DataCheckingBuilder(mutate_inputs)
    parameter_space = UniformParameterSpace(
        ['u', 'v', 'w'], [0, 1, 2], [1, 2, 3], sampler=MonteCarloSampler,
    )
    original_directory = os.getcwd()
    try:
        run_greedy(fom, builder, parameter_space, str(tmp_path),
                   tolerance=1e-4, testing_sample_size=5, calibrated_error=False)
    finally:
        os.chdir(original_directory)
    expected_order = [0, 1, 4, 2, 5]
    assert fom.compute_calls == len(expected_order)
    assert [len(call[0]) for call in builder.training_calls] == [2, 3, 4, 5]
    for directories, parameters, qois in builder.training_calls:
        count = len(directories)
        assert [int(os.path.basename(d).split('_')[-1]) for d in directories] == expected_order[:count]
        assert qois.shape == (count, int(np.prod(qoi_shape)) if qoi_shape else 1)
        assert np.all(parameters >= 0) and np.all(np.real(qois) > 0)
    # Earlier records remain independent as subsequent FOMs update their buffer.
    np.testing.assert_array_equal(builder.training_calls[0][2], builder.training_calls[-1][2][:2])


@pytest.mark.mpi_skip
@pytest.mark.parametrize('initial_count', [1, 2, 4])
@pytest.mark.parametrize('iteration_limit', [0, 2])
def test_configurable_initial_size_and_enrichment_limit(tmp_path, initial_count, iteration_limit):
    fom = ParameterQoiModel((2,))
    builder = DataCheckingBuilder(False)
    space = UniformParameterSpace(['u', 'v', 'w'], [0, 1, 2], [1, 2, 3], MonteCarloSampler)
    # Existing positional arguments (including random_seed and calibrated_error)
    # retain their meaning; the new controls are keyword-only.
    original_directory = os.getcwd()
    try:
        run_greedy(fom, builder, space, str(tmp_path), 0.0,
                   initial_count + 3, 7, False,
                   initial_sample_size=initial_count, max_iterations=iteration_limit)
    finally:
        os.chdir(original_directory)
    assert fom.compute_calls == initial_count + iteration_limit
    assert [len(c[0]) for c in builder.training_calls] == list(
        range(initial_count, initial_count + iteration_limit + 1)
    )
    assert [int(os.path.basename(d).split('_')[-1]) for d in builder.training_calls[0][0]] == list(range(initial_count))
    assert len(list(tmp_path.glob('fom/run_*'))) == initial_count + iteration_limit
    with np.load(tmp_path / 'greedy_stats.npz') as stats:
        assert stats['initial_sample_size'] == initial_count
        assert stats['max_iterations'] == iteration_limit
        assert stats['iterations'] == iteration_limit
        assert not bool(stats['converged'])
        assert str(stats['termination_reason']) == 'max_iterations'
        assert len(stats['qoi_errors']) == iteration_limit
        assert len(stats['max_error_indicators']) == iteration_limit
    assert 'Stopped: max_iterations' in (tmp_path / 'greedy_status.log').read_text()


class StrictPreparationFom(ParameterQoiModel):
    def __init__(self):
        super().__init__((2,))
        self.prepared, self.executed = [], []

    def populate_run_directory(self, directory, parameters):
        # Non-idempotent setup exposes duplicate calls immediately.
        with open(os.path.join(directory, 'setup_once.txt'), 'x') as marker:
            marker.write('prepared')
        self.prepared.append(directory)
        super().populate_run_directory(directory, parameters)

    def run_model(self, directory, parameters):
        assert os.path.exists(os.path.join(directory, 'setup_once.txt'))
        self.executed.append(directory)
        return 0


class StrictPreparationRom(MockQoiModelWithErrorEstimate):
    def __init__(self, errors, qoi):
        super().__init__(errors, qoi)
        self.prepared, self.executed = [], []

    def populate_run_directory(self, directory, parameters):
        with open(os.path.join(directory, 'setup_once.txt'), 'x') as marker:
            marker.write('prepared')
        self.prepared.append(directory)

    def run_model(self, directory, parameters):
        assert os.path.exists(os.path.join(directory, 'setup_once.txt'))
        self.executed.append(directory)
        return 0


class StrictPreparationBuilder(DataCheckingBuilder):
    def __init__(self):
        super().__init__(False)
        self.roms = []

    def build_from_training_dirs(self, *args):
        base = super().build_from_training_dirs(*args)
        rom = StrictPreparationRom(base.my_error_estimates_, base.my_qoi_)
        self.roms.append(rom)
        return rom


@pytest.mark.mpi_skip
@pytest.mark.parametrize('limit', [0, 3])
def test_fom_and_rom_directories_populated_once_only_when_needed(tmp_path, limit):
    fom, builder = StrictPreparationFom(), StrictPreparationBuilder()
    space = UniformParameterSpace(['u', 'v', 'w'], [0, 1, 2], [1, 2, 3], MonteCarloSampler)
    run_greedy(fom, builder, space, str(tmp_path), 0.0, testing_sample_size=5,
               max_iterations=limit, calibrated_error=False)
    assert fom.prepared == fom.executed
    assert len(fom.prepared) == 2 + limit
    assert len(set(fom.prepared)) == 2 + limit
    for rom in builder.roms:
        assert rom.prepared == rom.executed
        assert len(rom.prepared) == len(set(rom.prepared))
    assert sum(len(rom.prepared) for rom in builder.roms) == 3 * limit
    # The final rebuilt ROM has no eagerly prepared candidate directories.
    assert not builder.roms[-1].prepared
    assert not list((tmp_path / f'rom_iteration_{limit}').glob('run_*'))


@pytest.mark.mpi_skip
def test_default_iteration_limit_stops_nonconverging_model(tmp_path):
    class NeverConvergingRom:
        def populate_run_directory(self, directory, parameters):
            pass
        def run_model(self, directory, parameters):
            return 0
        def compute_qoi(self, directory, parameters):
            return np.ones(1)
        def compute_error_estimate(self, directory, parameters):
            return 1.0
    class Builder:
        def __init__(self):
            self.sizes = []
        def build_from_training_dirs(self, offline, dirs, parameters, qois):
            self.sizes.append(len(dirs))
            return NeverConvergingRom()
    fom, builder = ParameterQoiModel(()), Builder()
    space = UniformParameterSpace(['u', 'v', 'w'], [0, 1, 2], [1, 2, 3], MonteCarloSampler)
    run_greedy(fom, builder, space, str(tmp_path), 1e-9, testing_sample_size=3,
               calibrated_error=False)
    assert fom.compute_calls == 102
    assert builder.sizes == list(range(2, 103))
    with np.load(tmp_path / 'greedy_stats.npz') as stats:
        assert stats['max_iterations'] == stats['iterations'] == 100
        assert str(stats['termination_reason']) == 'max_iterations'


@pytest.mark.mpi_skip
def test_convergence_before_iteration_limit_records_tolerance(tmp_path):
    fom, builder = ParameterQoiModel((2,)), DataCheckingBuilder(False)
    space = UniformParameterSpace(['u', 'v', 'w'], [0, 1, 2], [1, 2, 3], MonteCarloSampler)
    original_directory = os.getcwd()
    try:
        run_greedy(fom, builder, space, str(tmp_path), 10.0, testing_sample_size=5,
                   initial_sample_size=3, max_iterations=20, calibrated_error=False)
    finally:
        os.chdir(original_directory)
    assert fom.compute_calls == 4  # Preserve the first-check-after-one-enrichment policy.
    with np.load(tmp_path / 'greedy_stats.npz') as stats:
        assert stats['iterations'] == 1
        assert bool(stats['converged'])
        assert str(stats['termination_reason']) == 'tolerance'
        assert len(stats['max_error_indicators']) == len(stats['qoi_errors']) == 1


@pytest.mark.mpi_skip
def test_initial_only_can_use_entire_candidate_pool(tmp_path):
    fom, builder = ParameterQoiModel(()), DataCheckingBuilder(False)
    space = UniformParameterSpace(['u', 'v', 'w'], [0, 1, 2], [1, 2, 3], MonteCarloSampler)
    run_greedy(fom, builder, space, str(tmp_path), 0,
               testing_sample_size=1, initial_sample_size=1, max_iterations=0)
    assert fom.compute_calls == 1
    assert len(builder.training_calls) == 1
    with np.load(tmp_path / 'greedy_stats.npz') as stats:
        np.testing.assert_array_equal(stats['training_samples'], [0])


@pytest.mark.parametrize('options', [
    {'initial_sample_size': 0}, {'initial_sample_size': -1},
    {'initial_sample_size': True}, {'initial_sample_size': 1.5},
    {'initial_sample_size': 6, 'testing_sample_size': 5},
    {'initial_sample_size': 5, 'testing_sample_size': 5},
    {'testing_sample_size': 0}, {'testing_sample_size': True},
    {'max_iterations': -1}, {'max_iterations': True}, {'max_iterations': 1.5},
])
def test_invalid_initial_size_and_iteration_limit_fail_before_io(tmp_path, options):
    fom, builder = ParameterQoiModel(()), DataCheckingBuilder(False)
    space = UniformParameterSpace(['u', 'v', 'w'], [0, 1, 2], [1, 2, 3], MonteCarloSampler)
    work = tmp_path / 'unused'
    with pytest.raises(ValueError, match='initial_sample_size|testing_sample_size|max_iterations'):
        run_greedy(fom, builder, space, str(work), 1e-5, **options)
    assert not work.exists()
    assert not builder.training_calls and fom.compute_calls == 0


@pytest.mark.mpi_skip
@pytest.mark.parametrize('limit', [0, 2])
def test_greedy_result_contains_final_rom_and_independent_training_data(tmp_path, limit):
    from romtools.workflows import GreedyResult
    fom, builder = StrictPreparationFom(), StrictPreparationBuilder()
    space = UniformParameterSpace(['u', 'v', 'w'], [0, 1, 2], [1, 2, 3], MonteCarloSampler)
    result = run_greedy(fom, builder, space, str(tmp_path), 0.0,
                        testing_sample_size=5, max_iterations=limit, calibrated_error=False)
    assert isinstance(result, GreedyResult)
    assert result.rom_model is builder.roms[-1]
    assert result.training_dirs == tuple(builder.training_calls[-1][0])
    np.testing.assert_array_equal(result.training_parameters, builder.training_calls[-1][1])
    np.testing.assert_array_equal(result.training_qois, builder.training_calls[-1][2])
    np.testing.assert_array_equal(result.sample_counts, np.arange(2, 3 + limit))
    assert result.iterations == limit and not result.converged
    assert result.termination_reason == 'max_iterations'
    with np.load(tmp_path / 'greedy_stats.npz') as stats:
        for name in ('training_parameters', 'training_qois', 'sample_counts',
                     'max_error_indicators', 'qoi_errors', 'predicted_qoi_errors'):
            np.testing.assert_array_equal(getattr(result, name), stats[name])
        np.testing.assert_array_equal(result.training_sample_indices, stats['training_samples'])
    result.training_parameters[:] = -1
    result.training_qois[:] = -1
    assert np.all(builder.training_calls[-1][1] >= 0)
    assert np.all(builder.training_calls[-1][2] > 0)


class InMemoryRom(MockQoiModelWithErrorEstimate):
    def __init__(self, errors, qoi):
        super().__init__(errors, qoi)
        self.prepared, self.executed, self.estimated, self.qoi_paths = [], [], [], []

    def populate_run_directory(self, directory, parameters):
        assert not os.path.exists(directory)
        self.prepared.append(directory)

    def run_model(self, directory, parameters):
        assert not os.path.exists(directory)
        self.executed.append(directory)
        return 0

    def compute_error_estimate(self, directory, parameters):
        assert not os.path.exists(directory)
        self.estimated.append(directory)
        return super().compute_error_estimate(directory, parameters)

    def compute_qoi(self, directory, parameters):
        assert not os.path.exists(directory)
        self.qoi_paths.append(directory)
        return super().compute_qoi(directory, parameters)


class InMemoryBuilder(DataCheckingBuilder):
    def __init__(self):
        super().__init__(False)
        self.roms = []

    def build_from_training_dirs(self, *args):
        base = super().build_from_training_dirs(*args)
        rom = InMemoryRom(base.my_error_estimates_, base.my_qoi_)
        self.roms.append(rom)
        return rom


@pytest.mark.mpi_skip
def test_skip_rom_directories_keeps_protocol_calls_fom_and_offline_output(tmp_path):
    fom, builder = StrictPreparationFom(), InMemoryBuilder()
    space = UniformParameterSpace(['u', 'v', 'w'], [0, 1, 2], [1, 2, 3], MonteCarloSampler)
    result = run_greedy(fom, builder, space, str(tmp_path), 0.0,
                        testing_sample_size=5, max_iterations=3, calibrated_error=False,
                        create_rom_run_directories=False)
    assert len(result.training_dirs) == 5
    assert all(os.path.isdir(d) for d in result.training_dirs)
    assert len(list(tmp_path.glob('rom_iteration_*/offline_data'))) == 4
    assert not list(tmp_path.glob('rom_iteration_*/run_*'))
    assert (tmp_path / 'greedy_status.log').is_file()
    assert (tmp_path / 'greedy_stats.npz').is_file()
    for rom in builder.roms:
        assert rom.prepared == rom.executed == rom.estimated
        assert all(os.path.isabs(d) for d in rom.prepared)
    assert sum(len(r.prepared) for r in builder.roms) == 9
    assert sum(len(r.qoi_paths) for r in builder.roms) == 3
    assert result.rom_model is builder.roms[-1]


class ExplicitCandidateSpace:
    def __init__(self, initial):
        self.initial = initial
        self.requests = []

    def get_names(self):
        return ['u', 'v', 'w']

    def generate_samples(self, count):
        self.requests.append(count)
        if len(self.requests) == 1:
            return np.vstack((self.initial, [[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]]))
        return np.full((count, 3), 10.0 * len(self.requests))


@pytest.mark.mpi_skip
def test_explicit_initial_samples_preserve_order_and_override_size(tmp_path):
    initial = np.array([[0.2, 1.2, 2.2], [0.3, 1.3, 2.3]])
    space = ExplicitCandidateSpace(initial.copy())
    builder = InMemoryBuilder()
    result = run_greedy(ParameterQoiModel((2,)), builder, space, str(tmp_path), 0.0,
                        initial_samples=initial, initial_sample_size=99,
                        testing_sample_size=5, max_iterations=3, calibrated_error=False,
                        create_rom_run_directories=False)
    np.testing.assert_array_equal(builder.training_calls[0][1], initial)
    np.testing.assert_array_equal(result.training_parameters[:2], initial)
    # Provided initial points are excluded from the separate candidate pool.
    # The appended candidate index is based on the combined pool's true size.
    np.testing.assert_array_equal(result.training_sample_indices, [0, 1, 6, 4, 7])
    np.testing.assert_array_equal(result.sample_counts, [2, 3, 4, 5])
    assert space.requests == [5, 1, 1]
    assert len({tuple(row) for row in result.training_parameters}) == 5
    initial[:] = -100
    assert np.all(result.training_parameters[:2] > 0)
    with np.load(tmp_path / 'greedy_stats.npz') as stats:
        assert stats['initial_sample_size'] == 2
        assert not bool(stats['create_rom_run_directories'])


@pytest.mark.mpi_skip
def test_provided_initial_data_can_exceed_candidate_pool_size(tmp_path):
    initial = np.array([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]])
    space = UniformParameterSpace(['u', 'v', 'w'], [0, 1, 2], [1, 2, 3], MonteCarloSampler)
    result = run_greedy(ParameterQoiModel(()), DataCheckingBuilder(False), space,
                        str(tmp_path), 0.0, initial_samples=initial,
                        testing_sample_size=1, max_iterations=0)
    np.testing.assert_array_equal(result.training_parameters, initial)
    assert result.iterations == 0


@pytest.mark.mpi_skip
def test_provided_samples_covering_candidate_pool_return_initial_rom(tmp_path):
    initial = np.array([[1., 2., 3.], [2., 3., 4.]])
    class RepeatedSpace:
        def get_names(self):
            return ['u', 'v', 'w']
        def generate_samples(self, count):
            return np.tile(initial[:1], (count, 1))
    result = run_greedy(ParameterQoiModel(()), DataCheckingBuilder(False), RepeatedSpace(),
                        str(tmp_path), 0.0, initial_samples=initial,
                        testing_sample_size=3, max_iterations=4)
    np.testing.assert_array_equal(result.training_parameters, initial)
    np.testing.assert_array_equal(result.sample_counts, [2])
    assert result.termination_reason == 'candidate_pool_exhausted'
    assert not result.converged and result.iterations == 0


@pytest.mark.mpi_skip
def test_provided_initial_samples_filter_replenished_training_duplicates(tmp_path):
    initial = np.array([[1., 2., 3.]])
    class ExhaustingSpace:
        calls = 0
        def get_names(self):
            return ['u', 'v', 'w']
        def generate_samples(self, count):
            self.calls += 1
            return np.array([[2., 3., 4.]]) if self.calls == 1 else initial.copy()
    result = run_greedy(ParameterQoiModel(()), InMemoryBuilder(), ExhaustingSpace(),
                        str(tmp_path), 0.0, initial_samples=initial,
                        testing_sample_size=1, max_iterations=3,
                        calibrated_error=False, create_rom_run_directories=False)
    assert result.termination_reason == 'candidate_pool_exhausted'
    assert result.iterations == 1
    np.testing.assert_array_equal(result.training_parameters, [[1, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(result.sample_counts, [1, 2])


@pytest.mark.parametrize('samples', [[], np.empty((0, 3)), [1, 2, 3], [[1, 2]],
                                     [[1, 2, np.nan]], [[1, np.inf, 3]]])
def test_bad_explicit_initial_samples_fail_before_io(tmp_path, samples):
    space = UniformParameterSpace(['u', 'v', 'w'], [0, 1, 2], [1, 2, 3], MonteCarloSampler)
    work = tmp_path / 'unused'
    with pytest.raises(ValueError, match='initial_samples'):
        run_greedy(ParameterQoiModel(()), DataCheckingBuilder(False), space,
                   str(work), 1e-5, initial_samples=samples)
    assert not work.exists()


@pytest.mark.parametrize('flag', [None, 0, 1, 'false', np.bool_(False)])
def test_bad_rom_directory_flag_fails_before_io(tmp_path, flag):
    space = UniformParameterSpace(['u', 'v', 'w'], [0, 1, 2], [1, 2, 3], MonteCarloSampler)
    work = tmp_path / 'unused'
    with pytest.raises(TypeError, match='create_rom_run_directories must be a bool'):
        run_greedy(ParameterQoiModel(()), DataCheckingBuilder(False), space,
                   str(work), 1e-5, create_rom_run_directories=flag)
    assert not work.exists()


@pytest.mark.mpi_skip
def test_result_reports_convergence_and_keeps_final_predictor(tmp_path):
    space = UniformParameterSpace(['u', 'v', 'w'], [0, 1, 2], [1, 2, 3], MonteCarloSampler)
    original = os.getcwd()
    try:
        result = run_greedy(ParameterQoiModel((2,)), DataCheckingBuilder(False), space,
                            str(tmp_path), 10.0, testing_sample_size=5,
                            calibrated_error=False)
    finally:
        os.chdir(original)
    assert result.converged and result.termination_reason == 'tolerance'
    assert result.iterations == 1
    np.testing.assert_array_equal(result.sample_counts, [2, 3])
    assert result.rom_model.compute_qoi('', {}).shape == (2,)


@pytest.mark.mpi_skip
def test_actual_gp_builder_supports_in_memory_rom_and_returned_model(tmp_path):
    from romtools.workflows.inverse.mf_eki_drivers import GaussianProcessQoiModelBuilderWithTrainingData
    gp_builder = GaussianProcessQoiModelBuilderWithTrainingData(parameter_names=['u', 'v', 'w'])
    class GpWithErrorEstimate:
        def __init__(self, model):
            self.model = model
        def populate_run_directory(self, directory, parameters):
            assert not os.path.exists(directory)
            self.model.populate_run_directory(directory, parameters)
        def run_model(self, directory, parameters):
            assert not os.path.exists(directory)
            return self.model.run_model(directory, parameters)
        def compute_qoi(self, directory, parameters):
            assert not os.path.exists(directory)
            return self.model.compute_qoi(directory, parameters)
        def compute_error_estimate(self, directory, parameters):
            assert not os.path.exists(directory)
            return float(np.linalg.norm(self.model.compute_qoi_and_var(directory, parameters)[1]))
    class Builder:
        def build_from_training_dirs(self, *args):
            return GpWithErrorEstimate(gp_builder.build_from_training_dirs(*args))
    space = UniformParameterSpace(['u', 'v', 'w'], [0, 1, 2], [1, 2, 3], MonteCarloSampler)
    initial = np.array([[.1, 1.1, 2.1], [.5, 1.5, 2.5]])
    result = run_greedy(ParameterQoiModel(()), Builder(), space, str(tmp_path), 0.0,
                        testing_sample_size=8, max_iterations=2, initial_samples=initial,
                        calibrated_error=False, create_rom_run_directories=False)
    prediction = result.rom_model.compute_qoi(str(tmp_path / 'no_directory'),
                                             {'u': .3, 'v': 1.3, 'w': 2.3})
    assert prediction.shape == (1,) and np.isfinite(prediction).all()
    assert len(result.training_dirs) == 4
    assert not list(tmp_path.glob('rom_iteration_*/run_*'))


@pytest.mark.mpi_skip
def test_supplied_initial_only_run_does_not_draw_unused_candidates(tmp_path):
    class NamesOnlySpace:
        def get_names(self):
            return ['u', 'v', 'w']
        def generate_samples(self, count):
            raise AssertionError('initial-only run must not sample adaptation candidates')
    initial = np.array([[1., 2., 3.], [2., 3., 4.]])
    result = run_greedy(ParameterQoiModel(()), DataCheckingBuilder(False), NamesOnlySpace(),
                        str(tmp_path), 0, initial_samples=initial, max_iterations=0)
    np.testing.assert_array_equal(result.training_parameters, initial)
    assert result.iterations == 0 and result.termination_reason == 'max_iterations'
