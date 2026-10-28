"""Independent gradient strategy selection with same-sample Hessian gains."""

import inspect

import numpy as np
import pytest

import romtools.workflows as workflows
from romtools.workflows.inverse.vi import mf_drivers as mf
from romtools.workflows.inverse.vi import newton as newton
from tests.romtools.workflows.inverse.test_mf_vi_delayed_adaptation import (
    LinearModel, VersionedBuilder, options, capture_evaluations,
)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
@pytest.mark.parametrize('strategy', ['same_sample', 'loo', 'k_fold', 'history'])
@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
def test_driver_hessian_uses_current_same_sample_reference(tmp_path, monkeypatch, family, strategy, mode, baseline):
    calls = capture_evaluations(monkeypatch)
    request = options(tmp_path, family, 2)
    request.update(gradient_gain_estimation_strategy=strategy, baseline_method=baseline,
                   mfmc_control_variate_mode=mode)
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    for arguments, state in calls:
        assert state['gradient_gain_estimation_strategy'] == strategy
        assert state['hessian_gain_estimation_strategy'] == 'same_sample'
        n, m = len(state['parameter_samples_fom']), len(state['parameter_samples_rom_only'])
        x = state['optimizer_samples']
        hessian, gains = mf._compute_mfmc_reinforce_hessian_full(
            x[:n], x[n:2*n], x[2*n:2*n+m],
            arguments['variational_mean'], np.exp(arguments['variational_log_std']),
            state['log_joint_terms_fom'], state['log_joint_terms_rom_base'],
            state['log_joint_terms_rom_only'], baseline, True, mode,
            variational_correlation_cholesky=arguments['variational_correlation_cholesky'],
            elbo_scaling_factor=arguments['elbo_scaling_factor'],
            force_unit_gain=state['bootstrap_rom'], return_gain_samples=True,
        )
        np.testing.assert_allclose(state['hessian_full'], hessian, atol=1e-12)
        for name, gain in gains.items():
            np.testing.assert_allclose(state['hessian_gain_samples'][name], gain, atol=1e-12)
            np.testing.assert_array_equal(gain, np.broadcast_to(gain[0], gain.shape))


@pytest.mark.mpi_skip
@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
def test_full_covariance_newton_hessian_ignores_gradient_strategy_and_history(tmp_path, monkeypatch, mode, baseline):
    calls = capture_evaluations(monkeypatch)
    request = options(tmp_path, 'full_covariance', 2)
    request.update(gradient_gain_estimation_strategy='history', baseline_method=baseline,
                   mfmc_control_variate_mode=mode)
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    arguments, state = calls[-1]
    mean = arguments['variational_mean']
    chol = np.diag(np.exp(arguments['variational_log_std']))
    def forbidden(*args, **kwargs):
        raise AssertionError('Hessian must not fit a cross-fold or historical gain')
    monkeypatch.setattr(mf, 'loo_acv_estimator', forbidden)
    monkeypatch.setattr(mf, 'fit_history_gain', forbidden)
    scale = arguments['elbo_scaling_factor']
    results = []
    for strategy in ('same_sample', 'loo', 'k_fold', 'history'):
        candidate = dict(state, gradient_gain_estimation_strategy=strategy,
                         gain_history_data=None, gain_num_folds=2)
        result = newton.estimate_mf_ordinary_hessian(candidate, mean, chol, baseline, True, mode, scale)
        results.append(result)
        gains = candidate['ordinary_hessian_gain_samples']
        np.testing.assert_array_equal(gains, np.broadcast_to(gains[0], gains.shape))
    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])
    n = len(state['parameter_samples_fom'])
    x = state['optimizer_samples']
    terms = [newton._mf_curvature_terms(points, mean, chol, scale*weights, baseline) for points, weights in (
        (x[:n], state['log_joint_terms_fom']),
        (x[n:2*n], state['log_joint_terms_rom_base']),
        (x[2*n:], state['log_joint_terms_rom_only']),
    )]
    expected = mf._mfmc_gradient_estimator(*terms, True, mode)[0]
    expected += newton._entropy_hessian(chol, scale)
    expected = 0.5*(expected+expected.T)
    np.testing.assert_allclose(results[0], expected, atol=1e-12)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
@pytest.mark.parametrize('invalid', ['loo', 'k_fold', 'history', 'invalid'])
@pytest.mark.parametrize('auto', [False, True])
def test_unsupported_hessian_strategy_rejected_before_evaluations(tmp_path, family, invalid, auto):
    request = options(tmp_path, family, 1)
    request['hessian_gain_estimation_strategy'] = invalid
    model = LinearModel()
    with pytest.raises(ValueError, match='hessian_gain_estimation_strategy'):
        if auto:
            workflows.mf_vi_with_auto_rom(model=model, **request)
        else:
            workflows.run_mf_vi(model=model, rom_model_builder=VersionedBuilder(), **request)
    assert not model.evaluations


@pytest.mark.mpi_skip
@pytest.mark.parametrize('auto', [False, True])
def test_old_public_argument_is_removed(tmp_path, auto):
    request = options(tmp_path, iterations=1)
    request['gain_estimation_strategy'] = 'loo'
    model = LinearModel()
    with pytest.raises(TypeError, match='gain_estimation_strategy'):
        if auto:
            workflows.mf_vi_with_auto_rom(model=model, **request)
        else:
            workflows.run_mf_vi(model=model, rom_model_builder=VersionedBuilder(), **request)
    assert not model.evaluations


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
@pytest.mark.parametrize('field', ['gradient_gain_estimation_strategy', 'hessian_gain_estimation_strategy'])
@pytest.mark.parametrize('change', ['missing_top', 'missing_state', 'mismatch_top', 'mismatch_state'])
def test_restart_requires_both_strategy_fields(tmp_path, family, field, change):
    request = options(tmp_path, family, 1)
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    with np.load(tmp_path/'iteration_0/restart.npz', allow_pickle=True) as saved:
        checkpoint = dict(saved)
    state = checkpoint['mf_vi_state'].item()
    target = checkpoint if change.endswith('top') else state
    if change.startswith('missing'):
        del target[field]
    else:
        target[field] = np.array('history') if target is checkpoint else 'history'
    checkpoint['mf_vi_state'] = np.array(state, dtype=object)
    broken = tmp_path/'broken.npz'
    np.savez(broken, **checkpoint)
    with pytest.raises(ValueError, match='metadata|state|strategies'):
        workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(),
                           restart_file=str(broken), **request)


def test_public_defaults_are_separate():
    for entrypoint in (workflows.run_mf_vi, workflows.mf_vi_with_auto_rom):
        parameters = inspect.signature(entrypoint).parameters
        assert 'gain_estimation_strategy' not in parameters
        assert parameters['gradient_gain_estimation_strategy'].default == 'same_sample'
        assert parameters['hessian_gain_estimation_strategy'].default == 'same_sample'


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
def test_same_sample_hessian_gain_responds_to_current_hf_sample(mode, baseline):
    rng = np.random.default_rng(38)
    x, z = rng.normal(size=(8, 2)), rng.normal(size=(7, 2))
    h, l, e = rng.normal(size=8), rng.normal(size=8), rng.normal(size=7)
    def evaluate(h):
        return mf._compute_mfmc_reinforce_hessian_full(
            x, x, z, np.zeros(2), np.ones(2), h, l, e, baseline, True, mode,
            return_gain_samples=True,
        )
    before = evaluate(h)[1]
    h[0] += 20
    after = evaluate(h)[1]
    assert any(not np.allclose(before[name], after[name]) for name in before)
    for gain in after.values():
        np.testing.assert_array_equal(gain, np.broadcast_to(gain[0], gain.shape))


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('extra', [0, 4])
def test_same_sample_hessian_bootstrap_and_no_extra_gains(mode, extra):
    from romtools.workflows.inverse.vi import drivers as vi
    rng = np.random.default_rng(209)
    x, z = rng.normal(size=(8, 2)), rng.normal(size=(extra, 2))
    h, l, e = rng.normal(size=8), rng.normal(size=8), rng.normal(size=extra)
    estimate, gains = mf._compute_mfmc_reinforce_hessian_full(
        x, x, z, np.zeros(2), np.ones(2), h, l, e, 'none', True, mode,
        force_unit_gain=True, return_gain_samples=True,
    )
    shape = () if mode == 'scalar' else ((4, 4) if mode == 'matrix' else (2, 2))
    expected_gain = (np.eye(4) if mode == 'matrix' else np.ones(shape)) if extra else np.zeros(shape)
    for gain in gains.values():
        assert gain.shape == (8,)+shape
        np.testing.assert_array_equal(gain, np.broadcast_to(expected_gain, gain.shape))
    if not extra:
        expected = vi._compute_reinforce_hessian_full(x, np.zeros(2), np.ones(2), h, 'none')
        np.testing.assert_allclose(estimate, expected, atol=1e-12)
