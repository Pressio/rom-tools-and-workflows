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
