"""Accepted-step damping schedules, retry caps, and restart continuity."""
import numpy as np
import pytest

from romtools.workflows.inverse.vi import drivers as vi
from romtools.workflows.inverse.vi import mf_drivers as mf
from tests.romtools.workflows.inverse.test_vi_control_resets import request, run


@pytest.mark.mpi_skip
@pytest.mark.parametrize('workflow', ['vi', 'mf'])
@pytest.mark.parametrize('adaptive', [False, True])
@pytest.mark.parametrize('method', ['legacy', 'stochastic_nonmonotone'])
@pytest.mark.parametrize('maximum', [None, .025])
def test_growth_with_rejected_retries(tmp_path, monkeypatch, workflow, adaptive, method, maximum):
    module = vi if workflow == 'vi' else mf
    name = '_build_vi_state_from_results' if workflow == 'vi' else '_evaluate_mf_vi_state'
    original = getattr(module, name)
    objectives = iter([10., 9., 0., -5., 8.5] if method == 'legacy'
                      else [10., 11., 0., -5., 12.])

    def evaluate(*args, **kwargs):
        state = original(*args, **kwargs)
        state['elbo'] = next(objectives)
        return state

    lambdas = []

    def step(*args, **kwargs):
        lambdas.append(kwargs['newton_additive_regularization'])
        return np.ones(1), np.zeros(1)

    monkeypatch.setattr(module, name, evaluate)
    monkeypatch.setattr(module, '_compute_newton_step', step)
    kwargs = request(tmp_path, workflow, 'diagonal', 3, method, 'newton', adaptive)
    kwargs['optimizer_config'].newton_additive_regularization_growth_factor = 2.
    kwargs['optimizer_config'].newton_additive_regularization_max = maximum
    run(kwargs, workflow)
    # First success advances the base to .02; retries cap at .02 * 5.
    expected = [.01, .02, .06, .1] if adaptive else [.01, .02, .02, .02]
    if maximum is not None:
        expected = np.minimum(expected, maximum)
    np.testing.assert_allclose(lambdas, expected)
    with np.load(tmp_path / 'iteration_2/restart.npz', allow_pickle=True) as saved:
        assert float(saved['current_newton_additive_regularization']) == min(.04, maximum or np.inf)
        assert float(saved['newton_additive_regularization_max']) == (maximum or np.inf)
        assert float(saved['initial_newton_additive_regularization']) == .01
        assert float(saved['newton_additive_regularization_growth_factor']) == 2.
        np.testing.assert_allclose(saved['vi_history_newton_additive_regularization'][-2:],
                                   np.asarray(expected)[[0, -1]])
        np.testing.assert_allclose(saved['vi_history_accepted_step_size'][-2:], [.004, .001])


@pytest.mark.mpi_skip
@pytest.mark.parametrize('workflow', ['vi', 'mf'])
@pytest.mark.parametrize('adaptive', [False, True])
@pytest.mark.parametrize('maximum', [None, .0106])
def test_restart_preserves_schedule_and_history(tmp_path, workflow, adaptive, maximum):
    def settings(directory, iterations):
        kwargs = request(directory, workflow, 'diagonal', iterations,
                         optimizer='newton', adaptive=adaptive)
        kwargs['optimizer_config'].newton_additive_regularization_growth_factor = 1.05
        kwargs['optimizer_config'].newton_additive_regularization_max = maximum
        return kwargs

    complete = run(settings(tmp_path / 'full', 4), workflow)
    run(settings(tmp_path / 'partial', 2), workflow)
    resumed_kwargs = settings(tmp_path / 'partial', 4)
    resumed_kwargs['restart_file'] = str(tmp_path / 'partial/iteration_1/restart.npz')
    resumed = run(resumed_kwargs, workflow)
    for a, b in zip(complete, resumed):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
    with np.load(tmp_path / 'full/history.npz', allow_pickle=True) as full, np.load(
        tmp_path / 'partial/history.npz', allow_pickle=True
    ) as partial:
        np.testing.assert_array_equal(full['vi_history_newton_additive_regularization'],
                                      partial['vi_history_newton_additive_regularization'])
    for key in ('newton_additive_regularization_growth_factor', 'newton_additive_regularization',
                'newton_additive_regularization_max'):
        kwargs = settings(tmp_path / 'mismatch', 4)
        kwargs['restart_file'] = resumed_kwargs['restart_file']
        setattr(kwargs['optimizer_config'], key, .0101 if key == 'newton_additive_regularization' else 1.2)
        with pytest.raises(ValueError, match='does not match current run'):
            run(kwargs, workflow)


@pytest.mark.parametrize('factor', [0., .99, -1., np.inf, np.nan, True, '1.05'])
def test_invalid_growth_factor(factor):
    with pytest.raises(ValueError, match='growth_factor'):
        vi._scheduled_newton_regularization(.01, factor, 0)


def test_schedule_zero_overflow_and_legacy_restart():
    assert vi._scheduled_newton_regularization(0., 2., 10000) == 0.
    assert vi._scheduled_newton_regularization(.01, 1., 10000) == .01
    np.testing.assert_allclose(vi._scheduled_newton_regularization(1e-300, 2., 1100),
                               np.ldexp(1e-300, 1100), rtol=1e-12)
    with pytest.raises(ValueError, match='regularization'):
        vi._scheduled_newton_regularization(.01, 2., 10000)
    assert vi._restore_adaptive_newton_regularization({}, True, .01, False, 3., 2., 5.) == .01
    with pytest.raises(ValueError, match='growth_factor'):
        vi._restore_adaptive_newton_regularization({}, True, .01, False, 3., 2., 5., 1.05)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('workflow', ['vi', 'mf'])
@pytest.mark.parametrize('field', ['newton_additive_regularization_growth_factor',
                                 'newton_additive_regularization_max'])
def test_full_covariance_rejects_growth(tmp_path, workflow, field):
    kwargs = request(tmp_path, workflow, 'full_covariance', 2, optimizer='newton')
    setattr(kwargs['optimizer_config'], field, 1.05)
    with pytest.raises(NotImplementedError, match='damping'):
        run(kwargs, workflow)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('workflow', ['vi', 'mf'])
@pytest.mark.parametrize('initial,factor', [(0., 1.05), (.01, .9), (.01, np.inf)])
def test_public_growth_configuration(tmp_path, workflow, initial, factor):
    kwargs = request(tmp_path, workflow, 'diagonal', 3, optimizer='newton')
    kwargs['optimizer_config'].newton_additive_regularization = initial
    kwargs['optimizer_config'].newton_additive_regularization_growth_factor = factor
    if initial:
        with pytest.raises(ValueError, match='growth_factor'):
            run(kwargs, workflow)
    else:
        run(kwargs, workflow)
        with np.load(tmp_path / 'history.npz', allow_pickle=True) as saved:
            np.testing.assert_array_equal(saved['vi_history_newton_additive_regularization'][1:], 0.)


@pytest.mark.parametrize('maximum', [-1., 0., .009, np.nan, np.inf, True, '1'])
def test_invalid_damping_maximum(maximum):
    with pytest.raises(ValueError, match='additive_regularization_max'):
        vi._scheduled_newton_regularization(.01, 1.05, 0, maximum)


def test_damping_maximum_plateau_and_overflow():
    np.testing.assert_allclose(
        [vi._scheduled_newton_regularization(.01, 2., k, .025) for k in range(6)],
        [.01, .02, .025, .025, .025, .025],
    )
    assert vi._scheduled_newton_regularization(.01, 2., 10000, .025) == .025
    assert vi._scheduled_newton_regularization(0., 2., 10000, 0.) == 0.
    assert vi._scheduled_newton_regularization(.01, 1.05, 1, .01) == .01


@pytest.mark.mpi_skip
@pytest.mark.parametrize('workflow', ['vi', 'mf'])
def test_public_invalid_damping_maximum(tmp_path, workflow):
    kwargs = request(tmp_path, workflow, 'diagonal', 3, optimizer='newton')
    kwargs['optimizer_config'].newton_additive_regularization_max = .009
    with pytest.raises(ValueError, match='additive_regularization_max'):
        run(kwargs, workflow)
