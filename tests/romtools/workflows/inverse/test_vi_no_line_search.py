"""Fixed-step Newton behavior through the public VI/MF-VI entrypoints."""

import numpy as np
import pytest

import romtools.workflows as workflows
from romtools.workflows.inverse.vi import optimization as methods
from romtools.workflows.inverse.vi import drivers as vi
from romtools.workflows.inverse.vi import mf_drivers as mf
from tests.romtools.workflows.inverse.test_vi_control_resets import request, run, force_two_rejections


def fixed_request(directory, workflow, family, iterations=3, step_size=0.004):
    result = request(directory, workflow, family, iterations, optimizer='newton')
    result.update(
        line_search_method='none',
        line_search_config=workflows.VINoLineSearchConfig(step_size=step_size),
    )
    return result


def test_config_defaults_and_normalization():
    method, config = methods._resolve_line_search_config(' NONE ', None)
    assert method == 'none'
    assert isinstance(config, workflows.VINoLineSearchConfig)
    assert config.step_size == 1.0
    assert config.log_std_learning_rate_factor == 1.0
    supplied = workflows.VINoLineSearchConfig(step_size=0.25)
    _, resolved = methods._resolve_line_search_config('none', supplied)
    assert resolved == supplied
    assert resolved is not supplied
    with pytest.raises(TypeError, match='VINoLineSearchConfig'):
        methods._resolve_line_search_config('none', workflows.VILegacyLineSearchConfig())


@pytest.mark.parametrize('field', ['step_size', 'log_std_learning_rate_factor'])
@pytest.mark.parametrize('value', [0.0, -0.1, np.inf, -np.inf, np.nan])
def test_invalid_fixed_controls(field, value):
    with pytest.raises(ValueError, match=field):
        methods._resolve_line_search_config('none', workflows.VINoLineSearchConfig(**{field: value}))


@pytest.mark.mpi_skip
@pytest.mark.parametrize('workflow', ['vi', 'mf'])
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
@pytest.mark.parametrize('optimizer', ['gradient', 'adam'])
def test_requires_newton(tmp_path, workflow, family, optimizer):
    options = fixed_request(tmp_path, workflow, family)
    options['optimizer_method'] = optimizer
    with pytest.raises(ValueError, match="requires optimizer_method='newton'"):
        run(options, workflow)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('workflow', ['vi', 'mf'])
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_accepts_worsening_elbo_without_retries(tmp_path, monkeypatch, workflow, family):
    states = force_two_rejections(monkeypatch, workflow, family, 'legacy')
    run(fixed_request(tmp_path, workflow, family), workflow)
    assert len(states) == 3  # Initial state and one candidate per update.
    assert states[-1]['elbo'] < states[0]['elbo']
    with np.load(tmp_path / 'history.npz', allow_pickle=True) as history:
        np.testing.assert_allclose(history['vi_history_accepted_step_size'][-2:], [0.004, 0.004])
    with np.load(tmp_path / 'iteration_2/restart.npz', allow_pickle=True) as saved:
        assert saved['line_search_method'].item() == 'none'
        assert saved['step_size'].item() == 0.004


@pytest.mark.mpi_skip
@pytest.mark.parametrize('workflow', ['vi', 'mf'])
@pytest.mark.parametrize('limited', [False, True])
def test_fixed_direction_scaling_and_update_limits(tmp_path, monkeypatch, workflow, limited):
    module = vi if workflow == 'vi' else mf
    monkeypatch.setattr(module, '_compute_newton_step', lambda *a, **k: (np.ones(1), np.ones(1)))
    options = fixed_request(tmp_path, workflow, 'diagonal', iterations=2, step_size=0.2)
    options['line_search_config'].log_std_learning_rate_factor = 0.5
    if limited:
        options['optimizer_config'].max_log_std_update = 0.03
        options['optimizer_config'].max_mean_update_std = 0.05
    run(options, workflow)
    with np.load(tmp_path / 'iteration_0/restart.npz', allow_pickle=True) as initial:
        initial_mean = initial['variational_mean'].copy()
        initial_log_std = initial['variational_log_std'].copy()
    with np.load(tmp_path / 'iteration_1/restart.npz', allow_pickle=True) as final:
        np.testing.assert_allclose(final['variational_mean'] - initial_mean, [0.05 if limited else 0.2])
        np.testing.assert_allclose(final['variational_log_std'] - initial_log_std, [0.03 if limited else 0.1])
        assert final['step_size'].item() == 0.2


@pytest.mark.mpi_skip
@pytest.mark.parametrize('workflow', ['vi', 'mf'])
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_restart_matches_uninterrupted_run(tmp_path, workflow, family):
    complete_dir = tmp_path / 'complete'
    resumed_dir = tmp_path / 'resumed'
    run(fixed_request(complete_dir, workflow, family, iterations=4), workflow)
    run(fixed_request(resumed_dir, workflow, family, iterations=2), workflow)
    options = fixed_request(resumed_dir, workflow, family, iterations=4)
    options['restart_file'] = str(resumed_dir / 'iteration_1/restart.npz')
    run(options, workflow)
    with np.load(complete_dir / 'iteration_3/restart.npz', allow_pickle=True) as complete, np.load(
        resumed_dir / 'iteration_3/restart.npz', allow_pickle=True
    ) as resumed:
        fields = ['variational_mean', 'step_size']
        fields += ['variational_cholesky_optimizer'] if family == 'full_covariance' else ['variational_log_std']
        for field in fields:
            np.testing.assert_allclose(resumed[field], complete[field])
        assert resumed['line_search_method'].item() == 'none'
    options['line_search_method'] = 'legacy'
    options['line_search_config'] = workflows.VILegacyLineSearchConfig()
    with pytest.raises(ValueError, match='line_search_method does not match'):
        run(options, workflow)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('workflow', ['vi', 'mf'])
def test_no_retry_driven_damping_increases(tmp_path, monkeypatch, workflow):
    force_two_rejections(monkeypatch, workflow, 'diagonal', 'legacy')
    module = vi if workflow == 'vi' else mf
    regularizations = []
    original_step = module._compute_newton_step

    def step(*args, **kwargs):
        regularizations.append(kwargs['newton_additive_regularization'])
        return original_step(*args, **kwargs)

    monkeypatch.setattr(module, '_compute_newton_step', step)
    options = fixed_request(tmp_path, workflow, 'diagonal')
    options['optimizer_config'].newton_adaptive_regularization = True
    run(options, workflow)
    np.testing.assert_allclose(regularizations, [0.01, 0.01])
