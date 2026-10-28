import itertools

import numpy as np
import pytest

import romtools.workflows as workflows
from romtools.workflows.inverse.vi.gains import gain_folds, k_fold_acv_estimator, loo_acv_estimator
from tests.romtools.workflows.inverse.test_mf_vi_acv import reference
from tests.romtools.workflows.inverse.test_mf_vi_delayed_adaptation import (
    LinearModel, VersionedBuilder, options, capture_evaluations,
)


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
@pytest.mark.parametrize('shape', [(3,), (2, 2)])
@pytest.mark.parametrize('n,k', [(1, 5), (2, 5), (3, 2), (8, 3), (8, 5)])
@pytest.mark.parametrize('m', [0, 5])
def test_k_fold_matches_nested_reference(mode, baseline, shape, n, k, m):
    rng = np.random.default_rng(522)
    h, l, e = rng.normal(size=n), rng.normal(size=n), rng.normal(size=m)
    s, es = rng.normal(size=(n,)+shape), rng.normal(size=(m,)+shape)
    actual = k_fold_acv_estimator(h, l, e, s, es, baseline, mode, gain_num_folds=k)
    expected = reference(h, l, e, s, es, baseline, mode, folds=gain_folds(n, k))
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=1e-8, rtol=1e-8)
    if k >= n:
        for a, b in zip(actual, loo_acv_estimator(h, l, e, s, es, baseline, mode)):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
def test_heldout_fold_and_extra_samples_do_not_fit_gain_or_baselines(mode, baseline):
    rng = np.random.default_rng(39)
    h, l, e = rng.normal(size=9), rng.normal(size=9), rng.normal(size=7)
    s, es = rng.normal(size=(9, 3)), rng.normal(size=(7, 3))
    for held in gain_folds(9, 3):
        before = k_fold_acv_estimator(h, l, e, s, es, baseline, mode,
                                     gain_num_folds=3, return_baselines=True)
        hh, ll, ss = h.copy(), l.copy(), s.copy()
        hh[held], ll[held], ss[held] = 1e8, -2e8, 3e7
        after = k_fold_acv_estimator(hh, ll, e*1e9, ss, es*-1e9, baseline, mode,
                                    gain_num_folds=3, return_baselines=True)
        np.testing.assert_array_equal(before[2][held], after[2][held])
        for key in ('high', 'low'):
            np.testing.assert_array_equal(before[3][key][held], after[3][key][held])
            train = np.setdiff1d(np.arange(9), held)
            if baseline == 'loo':
                weights = h if key == 'high' else l
                np.testing.assert_allclose(before[3][key][held], np.full((len(held), 3), weights[train].mean()))


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
def test_k_fold_exact_fixed_rom_unbiasedness(mode, baseline):
    p = 0.3
    expected = np.zeros(2)
    for realization in itertools.product((0., 1.), repeat=6):
        x, z = np.array(realization[:4]), np.array(realization[4:])
        probability = np.prod([p if value else 1-p for value in realization])
        s = np.column_stack([x-p, 2*(x-p)])
        es = np.column_stack([z-p, 2*(z-p)])
        result = k_fold_acv_estimator(x, 0.6*x+0.2, 0.6*z+0.2, s, es, baseline, mode, gain_num_folds=2)
        expected += probability*result[0]
    np.testing.assert_allclose(expected, p*(1-p)*np.array([1., 2.]), atol=1e-12)


def test_k_fold_gaussian_gradient_hessian_monte_carlo():
    rng = np.random.default_rng(67)
    estimates = []
    for _ in range(2000):
        x, z = rng.normal(size=8), rng.normal(size=11)
        s = np.column_stack([x, x*x-1])
        es = np.column_stack([z, z*z-1])
        estimates.append(k_fold_acv_estimator(x*x, 0.6*x*x+0.2*x,
                                            0.6*z*z+0.2*z, s, es, 'loo', 'scalar',
                                            gain_num_folds=3)[0])
    estimates = np.asarray(estimates)
    error = np.abs(estimates.mean(axis=0)-[0., 2.])
    assert np.all(error < 4*estimates.std(axis=0, ddof=1)/np.sqrt(len(estimates)))


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
def test_k_fold_driver_and_restart(tmp_path, monkeypatch, family, mode):
    calls = capture_evaluations(monkeypatch)
    request = options(tmp_path/'complete', family, 3)
    request.update(gradient_gain_estimation_strategy='k_fold', gain_num_folds=3, mfmc_control_variate_mode=mode)
    model, builder = LinearModel(), VersionedBuilder()
    full = workflows.run_mf_vi(model=model, rom_model_builder=builder, **request)
    assert len(model.evaluations) == 18
    assert len(builder.evaluations) == 30
    for _, state in calls:
        assert state['gain_num_folds'] == 3
        for a, b in zip(state['gain_fold_indices'], gain_folds(6, 3)):
            np.testing.assert_array_equal(a, b)
        for gain in state['gradient_gain_samples'].values():
            for fold in gain_folds(6, 3):
                np.testing.assert_allclose(gain[fold], np.broadcast_to(gain[fold[0]], gain[fold].shape))
    request = options(tmp_path/'partial', family, 2)
    request.update(gradient_gain_estimation_strategy='k_fold', gain_num_folds=3, mfmc_control_variate_mode=mode)
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    request = options(tmp_path/'partial', family, 3)
    request.update(gradient_gain_estimation_strategy='k_fold', gain_num_folds=3, mfmc_control_variate_mode=mode,
                   restart_file=str(tmp_path/'partial/iteration_1/restart.npz'))
    resumed = workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    for a, b in zip(full, resumed):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
    request['gain_num_folds'] = 2
    with pytest.raises(ValueError, match='k-fold count'):
        workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)


@pytest.mark.parametrize('value', [True, 1, 0, -1, 2.5, '5'])
def test_invalid_fold_counts(value):
    with pytest.raises(ValueError, match='gain_num_folds'):
        gain_folds(8, value)


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['loo', 'optimal'])
def test_k_fold_stability_and_degenerate_variance(mode, baseline):
    rng = np.random.default_rng(82)
    h, l, e = rng.normal(size=8), rng.normal(size=8), rng.normal(size=7)
    s, es = rng.normal(size=(8, 3)), rng.normal(size=(7, 3))
    args = (s, es, baseline, mode)
    plain = k_fold_acv_estimator(h, l, e, *args, gain_num_folds=3)
    shifted = k_fold_acv_estimator(h+1e8, l-1e8, e-1e8, *args, gain_num_folds=3)
    for a, b in zip(plain, shifted):
        np.testing.assert_allclose(a, b, atol=2e-7, rtol=2e-7)
    degenerate = k_fold_acv_estimator(h, np.full(8, 10.), np.full(7, 10.),
                                     *args, gain_num_folds=3)
    np.testing.assert_array_equal(degenerate[2], np.zeros_like(degenerate[2]))


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
def test_k_fold_forced_unit_gain_matches_reference(mode, baseline):
    rng = np.random.default_rng(61)
    h, l, e = rng.normal(size=7), rng.normal(size=7), rng.normal(size=5)
    s, es = rng.normal(size=(7, 3)), rng.normal(size=(5, 3))
    actual = k_fold_acv_estimator(h, l, e, s, es, baseline, mode,
                                 gain_num_folds=3, force_unit_gain=True)
    expected = reference(h, l, e, s, es, baseline, mode, True, gain_folds(7, 3))
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=1e-12)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_k_fold_auto_rom_wrapper(tmp_path, family):
    request = options(tmp_path, family, 2)
    request.update(gradient_gain_estimation_strategy='k_fold', gain_num_folds=3,
                   rom_args={'normalize_parameters': True, 'normalize_targets': True})
    result = workflows.mf_vi_with_auto_rom(model=LinearModel(), **request)
    assert all(np.all(np.isfinite(value)) for value in result)
    with np.load(tmp_path/'iteration_1/restart.npz', allow_pickle=True) as saved:
        state = saved['mf_vi_state'].item()
        assert state['gradient_gain_estimation_strategy'] == 'k_fold'
        assert state['gain_num_folds'] == 3


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family,metric', [('diagonal', 'standard'), ('diagonal', 'natural'), ('full_covariance', 'natural')])
def test_k_fold_newton(tmp_path, family, metric):
    request = options(tmp_path, family, 2)
    request.update(
        gradient_gain_estimation_strategy='k_fold', gain_num_folds=3, baseline_method='loo',
        optimizer_method='newton',
        optimizer_config=workflows.VINewtonOptimizerConfig(
            max_iterations=2, gradient_norm_tolerance=0.0,
            newton_metric=metric, newton_hessian_type='full',
        ),
        line_search_method='legacy',
        line_search_config=workflows.VILegacyLineSearchConfig(
            initial_step_size=1e-3, max_step_size=1e-3, step_size_growth_factor=1.0,
            relaxation_parameter=1e10, line_search_sample_growth_factor=1.0,
        ),
    )
    result = workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    assert all(np.all(np.isfinite(value)) for value in result)
    with np.load(tmp_path/'iteration_1/restart.npz', allow_pickle=True) as saved:
        state = saved['mf_vi_state'].item()
        samples = state.get('ordinary_hessian_gain_samples')
        if samples is not None:
            for fold in gain_folds(6, 3):
                np.testing.assert_allclose(samples[fold], np.broadcast_to(samples[fold[0]], samples[fold].shape))
        for samples in state['hessian_gain_samples'].values():
            for fold in gain_folds(6, 3):
                np.testing.assert_allclose(samples[fold], np.broadcast_to(samples[fold[0]], samples[fold].shape))


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
@pytest.mark.parametrize('missing', ['gain_num_folds', 'gain_fold_indices'])
def test_k_fold_restart_requires_metadata(tmp_path, family, missing):
    request = options(tmp_path, family, 1)
    request.update(gradient_gain_estimation_strategy='k_fold', gain_num_folds=3)
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    with np.load(tmp_path/'iteration_0/restart.npz', allow_pickle=True) as saved:
        checkpoint = dict(saved)
    state = checkpoint['mf_vi_state'].item()
    del state[missing]
    checkpoint['mf_vi_state'] = np.array(state, dtype=object)
    broken = tmp_path/'broken.npz'
    np.savez(broken, **checkpoint)
    with pytest.raises(ValueError, match='k-fold'):
        workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(),
                           restart_file=str(broken), **request)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_k_fold_requires_independent_mc_samples(tmp_path, family):
    request = options(tmp_path, family, 1)
    request.update(gradient_gain_estimation_strategy='k_fold', sampling_method='rqmc')
    model = LinearModel()
    with pytest.raises(ValueError, match='independent paired samples'):
        workflows.run_mf_vi(model=model, rom_model_builder=VersionedBuilder(), **request)
    assert not model.evaluations


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_k_fold_defaults_and_does_not_consume_sampling_rng(tmp_path, family):
    request = options(tmp_path/'k_fold', family, 1)
    request['gradient_gain_estimation_strategy'] = 'k_fold'
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    with np.load(tmp_path/'k_fold/iteration_0/restart.npz', allow_pickle=True) as saved:
        k_snapshot = dict(saved)
        k_state = saved['mf_vi_state'].item()
    assert k_state['gain_num_folds'] == 5
    assert len(k_state['gain_fold_indices']) == 5
    request = options(tmp_path/'loo', family, 1)
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    with np.load(tmp_path/'loo/iteration_0/restart.npz', allow_pickle=True) as saved:
        l_state = saved['mf_vi_state'].item()
        np.testing.assert_array_equal(k_state['optimizer_samples'], l_state['optimizer_samples'])
        for key in saved.files:
            if 'rng' in key or 'random_state' in key:
                a, b = k_snapshot[key], saved[key]
                assert repr(a.tolist()) == repr(b.tolist())
