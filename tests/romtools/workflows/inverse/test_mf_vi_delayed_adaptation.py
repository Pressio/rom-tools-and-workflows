import inspect

import numpy as np
import pytest

import romtools.workflows as workflows
from romtools.workflows.inverse.vi import mf_drivers as mf
from romtools.workflows.inverse.vi import drivers as vi
from romtools.workflows.parameter_spaces import (
    GaussianParameterSpace, MonteCarloSampler, MultivariateGaussianParameterSpace,
)


class LinearModel:
    def __init__(self, slope=2.0, version=None, evaluations=None):
        self.slope = slope
        self.version = version
        self.evaluations = [] if evaluations is None else evaluations

    def populate_run_directory(self, run_directory, parameter_sample):
        pass

    def run_model(self, run_directory, parameter_sample):
        return 0

    def compute_qoi(self, run_directory, parameter_sample):
        self.evaluations.append((self.version, run_directory))
        return np.array([self.slope * parameter_sample['theta']])


class VersionedBuilder:
    def __init__(self, random_draw=False):
        self.random_draw = random_draw
        self.builds = []
        self.evaluations = []

    def build_from_training_dirs(self, directory, dirs, parameters, qois):
        if self.random_draw:
            np.random.random()
        self.builds.append((directory, list(dirs), parameters.copy()))
        version = len(dirs)
        return LinearModel(1.0 + 0.1 * version, version, self.evaluations)


def options(directory, family='diagonal', iterations=3):
    prior = GaussianParameterSpace(['theta'], np.zeros(1), np.ones(1), MonteCarloSampler)
    initializer = prior if family == 'diagonal' else MultivariateGaussianParameterSpace(
        ['theta'], np.zeros(1), np.eye(1), MonteCarloSampler,
    )
    return dict(
        prior_parameter_space=prior, initial_variational_parameter_space=initializer,
        observations=np.array([0.5]), observations_covariance=np.array([[0.1]]),
        absolute_work_dir=str(directory), fom_sample_size=6, rom_extra_sample_size=4,
        max_rom_training_history=2, rom_tolerance=0.0, random_seed=17,
        fom_evaluation_concurrency=1, rom_evaluation_concurrency=1,
        rom_retraining_strategy='delayed', gradient_gain_estimation_strategy='loo', baseline_method='none',
        elbo_scaling_factor=1.0, bounded_parameter_handling='clip',
        optimizer_method='adam', optimizer_config=workflows.VIAdamOptimizerConfig(
            max_iterations=iterations, gradient_norm_tolerance=0.0,
        ),
    )


def capture_evaluations(monkeypatch):
    original = mf._evaluate_mf_vi_state
    signature = inspect.signature(original)
    calls = []

    def capture(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        state = original(*args, **kwargs)
        calls.append((dict(arguments.arguments), state))
        return state

    monkeypatch.setattr(mf, '_evaluate_mf_vi_state', capture)
    return calls


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
@pytest.mark.parametrize('extra,control', [(4, True), (0, True), (4, False)])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
@pytest.mark.parametrize('entropy', ['analytic', 'joint'])
def test_delayed_models_activate_on_next_iteration(tmp_path, monkeypatch, family, extra, control, baseline, entropy):
    builder = VersionedBuilder()
    model = LinearModel()
    calls = capture_evaluations(monkeypatch)
    request = options(tmp_path, family)
    request.update(rom_extra_sample_size=extra, use_mfmc_control_variate=control,
                   baseline_method=baseline, score_function_entropy_strategy=entropy)
    workflows.run_mf_vi(model=model, rom_model_builder=builder, **request)
    assert len(calls) == 3
    assert [None if state['rom_model'] is None else state['rom_model'].version
            for _, state in calls] == [6, 6, 12]
    assert [len(dirs) for _, dirs, _ in builder.builds] == [6, 12, 12]
    assert all(directory.endswith('/rom_delayed_update') for directory, _, _ in builder.builds[1:])
    assert len(model.evaluations) == 18
    assert len(builder.evaluations) == 3 * (6 + extra)
    assert all('gain_sample' not in path and 'kfold' not in path for _, path in builder.evaluations)
    for index, (arguments, state) in enumerate(calls):
        assert np.all(np.isnan(state['gradient_standard_error']))
        assert state['training_parameters'].shape == (6 * (index + 1), 1)
        expected_qois = state['rom_model'].slope * state['parameter_samples_fom'].T
        np.testing.assert_allclose(state['qois_rom_base'], expected_qois)
        np.testing.assert_array_equal(state['parameter_samples_rom_base'], state['parameter_samples_fom'])
        if index == 0:
            assert state['bootstrap_rom']
            np.testing.assert_array_equal(state['mfmc_alpha_mean'], [1.0 if extra else 0.0])
            np.testing.assert_array_equal(state['mfmc_alpha_log_std'], [1.0 if extra else 0.0])
    with np.load(tmp_path / 'iteration_1' / 'restart.npz', allow_pickle=True) as restart:
        saved = restart['mf_vi_state'].item()
        assert len(saved['rom_training_dirs']) == 6
        assert len(saved['pending_rom_training_dirs']) == 12
        assert saved['has_pending_rom']


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
@pytest.mark.parametrize('checkpoint', [0, 1])
@pytest.mark.parametrize('random_build', [False, True])
def test_delayed_restart_preserves_next_activation(tmp_path, family, checkpoint, random_build):
    complete = tmp_path / 'complete'
    partial = tmp_path / 'partial'
    uninterrupted = workflows.run_mf_vi(
        model=LinearModel(), rom_model_builder=VersionedBuilder(random_build), **options(complete, family, 4),
    )
    workflows.run_mf_vi(
        model=LinearModel(), rom_model_builder=VersionedBuilder(random_build), **options(partial, family, checkpoint + 1),
    )
    builder = VersionedBuilder(random_build)
    resumed = workflows.run_mf_vi(
        model=LinearModel(), rom_model_builder=builder,
        restart_file=str(partial / f'iteration_{checkpoint}' / 'restart.npz'), **options(partial, family, 4),
    )
    for actual, expected in zip(resumed, uninterrupted):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    # Only later iterations evaluate ROM samples; completed batches are restored.
    assert len(builder.evaluations) == (3 - checkpoint) * 10
    with np.load(complete / 'iteration_3' / 'restart.npz', allow_pickle=True) as full:
        with np.load(partial / 'iteration_3' / 'restart.npz', allow_pickle=True) as restarted:
            a = full['mf_vi_state'].item()
            b = restarted['mf_vi_state'].item()
            for key in ('gradient_mean', 'gradient_standard_error', 'elbo', 'optimizer_samples'):
                np.testing.assert_allclose(a[key], b[key], rtol=0.0, atol=1e-12)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_delayed_no_adaptation_keeps_existing_rom(tmp_path, monkeypatch, family):
    builder = VersionedBuilder()
    calls = capture_evaluations(monkeypatch)
    request = options(tmp_path, family)
    request['rom_tolerance'] = 1e10
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=builder, **request)
    assert len(builder.builds) == 1
    assert [None if state['rom_model'] is None else state['rom_model'].version
            for _, state in calls] == [6, 6, 6]




@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_delayed_rejected_trials_do_not_train_or_activate(tmp_path, monkeypatch, family):
    from romtools.workflows.inverse.vi import full_covariance_mf_drivers as fc
    target = mf if family == 'diagonal' else fc
    evaluator = '_evaluate_mf_vi_state' if family == 'diagonal' else '_evaluate_mf_state'
    original = getattr(target, evaluator)
    calls = []
    builder = VersionedBuilder()

    def evaluate(*args, **kwargs):
        state = original(*args, **kwargs)
        if len(calls) == 1:
            state['elbo'] = -1e30
        elif len(calls) == 2:
            state['elbo'] = 1e30
        calls.append(state)
        return state

    monkeypatch.setattr(target, evaluator, evaluate)
    request = options(tmp_path, family, 2)
    request.update(
        optimizer_method='gradient',
        optimizer_config=workflows.VIGradientOptimizerConfig(max_iterations=2, gradient_norm_tolerance=0.0),
        line_search_method='legacy',
        line_search_config=workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-3, max_step_size=1e-3, step_size_growth_factor=1.0,
            relaxation_parameter=10.0, line_search_sample_growth_factor=1.0,
        ),
    )
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=builder, **request)
    assert len(calls) == 3
    assert len(builder.builds) == 2
    assert calls[1]['rom_model'] is calls[2]['rom_model']
    assert calls[1]['rom_model'].version == 6
    rejected_parameters = calls[1]['parameter_samples_fom'][:, 0]
    accepted_parameters = calls[2]['parameter_samples_fom'][:, 0]
    assert not np.isin(rejected_parameters, builder.builds[-1][2][:, 0]).any()
    np.testing.assert_array_equal(builder.builds[-1][2][-6:, 0], accepted_parameters)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_delayed_rom_substeps_use_activated_rom(tmp_path, monkeypatch, family):
    from romtools.workflows.inverse.vi import full_covariance_mf_drivers as fc
    seen = []

    def substep(**kwargs):
        seen.append(kwargs['rom_model'].version)
        if family == 'diagonal':
            return kwargs['candidate_mean'], kwargs['candidate_log_std']
        return kwargs['candidate_mean'], kwargs['candidate_cholesky']

    if family == 'diagonal':
        monkeypatch.setattr(mf, 'apply_diagonal_rom_substeps', substep)
    else:
        monkeypatch.setattr(fc, 'apply_full_covariance_rom_substeps', substep)
    request = options(tmp_path, family)
    request.update(num_rom_substeps=1)
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    assert seen == [6, 12]


@pytest.mark.mpi_skip
def test_delayed_independent_newton_curvature_uses_active_rom(tmp_path, monkeypatch):
    calls = capture_evaluations(monkeypatch)
    request = options(tmp_path, iterations=2)
    request.update(
        optimizer_method='newton',
        optimizer_config=workflows.VINewtonOptimizerConfig(
            max_iterations=2, gradient_norm_tolerance=0.0, newton_hessian_type='full',
            newton_curvature_strategy='independent', newton_hessian_num_samples=4,
        ),
        line_search_method='legacy',
        line_search_config=workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-3, max_step_size=1e-3, step_size_growth_factor=1.0,
            relaxation_parameter=1e10, line_search_sample_growth_factor=1.0,
        ),
    )
    builder = VersionedBuilder()
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=builder, **request)
    hessian_calls = [(args, state) for args, state in calls if args['iteration_directory'].endswith('/hessian')]
    assert len(hessian_calls) == 1
    assert hessian_calls[0][1]['rom_model'].version == 6
    assert len(builder.builds) == 2
    assert all('/hessian' not in path for path, _, _ in builder.builds)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_delayed_restart_requires_matching_metadata(tmp_path, family):
    request = options(tmp_path, family, 1)
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    restart_path = tmp_path / 'iteration_0' / 'restart.npz'
    with pytest.raises(ValueError, match='MF-VI'):
        workflows.run_mf_vi(
            model=LinearModel(), rom_model_builder=VersionedBuilder(),
            **dict(request, rom_retraining_strategy='same_sample', restart_file=str(restart_path)),
        )
    with np.load(restart_path, allow_pickle=True) as restart:
        old = {key: restart[key] for key in restart.files
               if key not in ('mf_vi_state', 'rom_retraining_strategy')}
    legacy_path = tmp_path / 'legacy.npz'
    np.savez(legacy_path, **old)
    with pytest.raises(ValueError, match='MF-VI'):
        workflows.run_mf_vi(
            model=LinearModel(), rom_model_builder=VersionedBuilder(),
            **dict(request, restart_file=str(legacy_path)),
        )


@pytest.mark.mpi_skip
def test_delayed_rejects_enabled_sample_reuse(tmp_path):
    request = options(tmp_path, iterations=1)
    request.update(sample_reuse_config=workflows.VISampleReuseConfig())
    with pytest.raises(NotImplementedError, match="rom_retraining_strategy='same_sample'"):
        workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_delayed_auto_rom_wrapper(tmp_path, monkeypatch, family):
    calls = capture_evaluations(monkeypatch)
    request = options(tmp_path, family, 2)
    request.update(
        optimizer_method='gradient',
        optimizer_config=workflows.VIGradientOptimizerConfig(max_iterations=2, gradient_norm_tolerance=0.0),
        line_search_method='legacy',
        line_search_config=workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-3, max_step_size=1e-3, relaxation_parameter=1e10,
        ),
    )
    workflows.mf_vi_with_auto_rom(
        model=LinearModel(), rom_args={'tune_hyperparameters': False}, **request,
    )
    assert len(calls) == 2
    assert calls[0][1]['bootstrap_rom']
    assert calls[1][1]['rom_model'] is not None
    assert all(state['gradient_gain_estimation_strategy'] == 'loo' for _, state in calls)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_delayed_natural_newton(tmp_path, family):
    request = options(tmp_path, family, 2)
    request.update(
        optimizer_method='newton',
        optimizer_config=workflows.VINewtonOptimizerConfig(
            max_iterations=2, gradient_norm_tolerance=0.0,
            newton_metric='natural', newton_hessian_type='full',
        ),
        line_search_method='legacy',
        line_search_config=workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-3, max_step_size=1e-3, step_size_growth_factor=1.0,
            relaxation_parameter=1e10, line_search_sample_growth_factor=1.0,
        ),
    )
    result = workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    assert all(np.all(np.isfinite(value)) for value in result)


def test_delayed_rejects_builder_that_reuses_active_model(tmp_path):
    from romtools.workflows.inverse.vi.adaptation import stage_delayed_rom
    shared = LinearModel()

    class SharedBuilder:
        def build_from_training_dirs(self, *args):
            return shared

    state = dict(
        rom_retraining_strategy='delayed', gradient_gain_estimation_strategy='loo', rom_adaptation_required=True,
        rom_model=shared, training_dirs=['sample'], training_parameters=np.zeros((1, 1)),
        training_qois=np.zeros((1, 1)),
    )
    with pytest.raises(ValueError, match='distinct replacement'):
        stage_delayed_rom(state, SharedBuilder(), str(tmp_path), 1, None)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
@pytest.mark.parametrize('mode', ['scalar', 'matrix'])
def test_delayed_multidimensional_gain_history_and_restart(tmp_path, monkeypatch, family, mode):
    dimension = 7
    names = [f'theta_{index}' for index in range(dimension)]

    class Model(LinearModel):
        def compute_qoi(self, run_directory, parameter_sample):
            return np.array([self.slope * sum(parameter_sample[name] for name in names)])

    class Builder:
        def build_from_training_dirs(self, directory, dirs, parameters, qois):
            return Model(slope=1.6)

    prior = GaussianParameterSpace(names, np.zeros(dimension), np.ones(dimension), MonteCarloSampler)
    initializer = prior if family == 'diagonal' else MultivariateGaussianParameterSpace(
        names, np.zeros(dimension), np.eye(dimension), MonteCarloSampler,
    )
    request = options(tmp_path, family, 2)
    request.update(prior_parameter_space=prior, initial_variational_parameter_space=initializer,
                   fom_sample_size=12, mfmc_control_variate_mode=mode)
    calls = capture_evaluations(monkeypatch)
    workflows.run_mf_vi(model=Model(), rom_model_builder=Builder(), **request)
    assert len(calls) == 2
    assert calls[0][1]['bootstrap_rom']
    assert calls[1][1]['rom_model'] is not None
    gain_shape = () if mode == 'scalar' else (dimension, dimension)
    for _, state in calls:
        assert state['mfmc_alpha_mean'].shape == gain_shape
        assert state['mfmc_alpha_log_std'].shape == gain_shape
    np.testing.assert_array_equal(calls[0][1]['mfmc_alpha_mean'],
                                  np.array(1.0) if mode == 'scalar' else np.eye(dimension))
    assert np.any(calls[1][1]['mfmc_alpha_mean'] != 0.0)

    restart_path = tmp_path / 'iteration_1' / 'restart.npz'
    with np.load(restart_path, allow_pickle=True) as restart:
        if family == 'diagonal':
            for key in ('vi_history_mfmc_alpha_mean', 'vi_history_mfmc_alpha_log_std'):
                assert restart[key].shape == (2,) + gain_shape
                assert restart[key].dtype != object
        else:
            saved = restart['mf_vi_state'].item()
            assert saved['mfmc_alpha_packed'].shape == (() if mode == 'scalar' else (35, 35))
    before_resume = len(calls)
    request.update(optimizer_config=workflows.VIAdamOptimizerConfig(
        max_iterations=3, gradient_norm_tolerance=0.0,
    ), restart_file=str(restart_path))
    workflows.run_mf_vi(model=Model(), rom_model_builder=Builder(), **request)
    assert len(calls) == before_resume + 1
    with np.load(tmp_path / 'history.npz', allow_pickle=True) as history:
        if family == 'diagonal':
            for key in ('vi_history_mfmc_alpha_mean', 'vi_history_mfmc_alpha_log_std'):
                assert history[key].shape == (3,) + gain_shape
                assert history[key].dtype != object
        else:
            assert history['mfmc_alpha_packed'].shape == (() if mode == 'scalar' else (35, 35))


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
@pytest.mark.parametrize('retraining', ['same_sample', 'delayed'])
@pytest.mark.parametrize('gain', ['same_sample', 'loo'])
@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
def test_separate_strategy_axes(tmp_path, monkeypatch, family, retraining, gain, mode):
    builder = VersionedBuilder()
    calls = capture_evaluations(monkeypatch)
    request = options(tmp_path, family)
    request.update(rom_retraining_strategy=retraining, gradient_gain_estimation_strategy=gain,
                   mfmc_control_variate_mode=mode)
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=builder, **request)
    assert len(calls) == 3
    versions = [state['rom_model'].version for _, state in calls]
    assert versions == ([6, 12, 12] if retraining == 'same_sample' else [6, 6, 12])
    assert len(builder.builds) == 3
    for i, (_, state) in enumerate(calls):
        assert state['rom_retraining_strategy'] == retraining
        assert state['gradient_gain_estimation_strategy'] == gain
        assert state['bootstrap_rom'] == (i == 0)
        if gain == 'loo':
            assert np.all(np.isnan(state['gradient_standard_error']))
            assert np.isnan(state['gradient_signal_to_noise_ratio'])
        else:
            assert np.all(np.isfinite(state['gradient_standard_error']))
        assert state['gradient_gain_samples']['mean'].shape[0] == 6
    with np.load(tmp_path/'iteration_2'/'restart.npz', allow_pickle=True) as saved:
        assert str(saved['rom_retraining_strategy'].item()) == retraining
        assert str(saved['gradient_gain_estimation_strategy'].item()) == gain
        assert 'gradient_gain_samples' in saved['mf_vi_state'].item()


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
@pytest.mark.parametrize('removed', ['correlation_estimator', 'correlation_k_folds', 'rom_base_sampling_strategy'])
def test_removed_strategy_settings_are_not_accepted(tmp_path, family, removed):
    request = options(tmp_path, family, 1)
    request[removed] = 'independent' if removed != 'correlation_k_folds' else 3
    with pytest.raises(TypeError, match=removed):
        workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('field,value', [('rom_retraining_strategy', 'independent'),
                                         ('gradient_gain_estimation_strategy', 'unknown_gain'),
                                         ('gradient_gain_estimation_strategy', 'kfold')])
def test_invalid_new_strategy_settings(tmp_path, field, value):
    request = options(tmp_path, iterations=1)
    request[field] = value
    with pytest.raises(ValueError, match=field):
        workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_loo_rejects_correlated_rqmc_pairs(tmp_path, family):
    request = options(tmp_path, family, 2)
    request.update(sampling_method='rqmc')
    with pytest.raises(ValueError, match='independent paired samples'):
        workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    request.update(gradient_gain_estimation_strategy='same_sample')
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
