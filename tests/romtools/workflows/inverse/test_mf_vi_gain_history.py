import numpy as np
import pytest

import romtools.workflows as workflows
from romtools.workflows.inverse.vi import mf_drivers as mf
from romtools.workflows.inverse.vi.gains import (
    fit_history_gain, gaussian_history_kernels, history_weights, importance_weights, append_training_batch,
)
from tests.romtools.workflows.inverse.test_mf_vi_delayed_adaptation import (
    LinearModel, VersionedBuilder, options, capture_evaluations,
)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
def test_history_uses_full_training_window_and_current_distribution(tmp_path, monkeypatch, family, mode, baseline):
    calls = capture_evaluations(monkeypatch)
    builder, model = VersionedBuilder(), LinearModel()
    request = options(tmp_path, family, 3)
    request.update(gradient_gain_estimation_strategy='history', mfmc_control_variate_mode=mode, baseline_method=baseline)
    workflows.run_mf_vi(model=model, rom_model_builder=builder, **request)
    assert len(model.evaluations) == 18
    # Existing paired predictions are reused until a replacement ROM activates.
    assert len(builder.evaluations) == 42
    assert sum('run_rom_gain_history_' in path for _, path in builder.evaluations) == 12
    assert calls[0][1]['gain_history_data'] is None
    for index in (1, 2):
        previous = calls[index-1][1]
        arguments, current = calls[index]
        data = current['gain_history_data']
        retained = [item[1] for item in calls[:index]]
        np.testing.assert_array_equal(data['parameter_samples'], np.concatenate([item['parameter_samples_fom'] for item in retained]))
        np.testing.assert_array_equal(data['optimizer_samples'], np.concatenate([item['optimizer_samples'][:6] for item in retained]))
        np.testing.assert_array_equal(data['high_log_joint'], np.concatenate([item['log_joint_terms_fom'] for item in retained]))
        np.testing.assert_array_equal(data['mean'], current['gain_distribution_mean'])
        np.testing.assert_array_equal(data['std'], current['gain_distribution_std'])
        assert data['source_directory'].endswith(f'iteration_{index-1}')
        np.testing.assert_array_equal(data['mean'], arguments['variational_mean'])
        assert len(data['proposal_distributions']) == index
        assert np.isclose(data['importance_weights'].sum(), 1)
        assert not np.allclose(data['importance_weights'], 1/len(data['importance_weights']))
        expected_qois = current['rom_model'].slope*data['parameter_samples'].T
        errors = expected_qois-np.array([0.5])[:, None]
        low_ll, _ = mf._compute_log_likelihoods(errors, np.array([[0.1]]), 1e-7)
        np.testing.assert_allclose(data['low_log_joint'], low_ll+np.concatenate([item['log_priors_fom'] for item in retained])+np.concatenate([item['log_transform_jacobian_fom'] for item in retained]))
        if family == 'diagonal':
            hw, lw = history_weights(data, 1.0)
            kernels = gaussian_history_kernels(data)[0]
            for key, scores in zip(('mean', 'log_std'), kernels):
                expected = fit_history_gain(hw, lw, scores, baseline, mode, sample_weights=data['importance_weights'])
                np.testing.assert_allclose(current['mfmc_alpha_'+key], expected)
                np.testing.assert_allclose(current['gradient_gain_samples'][key], np.broadcast_to(expected, current['gradient_gain_samples'][key].shape))
    with np.load(tmp_path/'iteration_2/restart.npz', allow_pickle=True) as saved:
        state = saved['mf_vi_state'].item()
        assert state['gain_history_data']['source_directory'].endswith('iteration_1')
        assert state['gradient_gain_estimation_strategy'] == 'history'


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
@pytest.mark.parametrize('checkpoint', [0, 1])
def test_history_restart_matches_uninterrupted(tmp_path, family, checkpoint):
    request = options(tmp_path/'complete', family, 4)
    request['gradient_gain_estimation_strategy'] = 'history'
    full = workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    request = options(tmp_path/'partial', family, checkpoint+1)
    request['gradient_gain_estimation_strategy'] = 'history'
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    request = options(tmp_path/'partial', family, 4)
    request.update(gradient_gain_estimation_strategy='history', restart_file=str(tmp_path/f'partial/iteration_{checkpoint}/restart.npz'))
    resumed = workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    for actual, expected in zip(resumed, full):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
@pytest.mark.parametrize('entropy', ['analytic', 'joint'])
def test_history_current_samples_cannot_change_gain(mode, baseline, entropy):
    rng = np.random.default_rng(81)
    data = dict(optimizer_samples=rng.normal(size=(8, 2)), mean=np.zeros(2), std=np.ones(2),
                correlation_cholesky=None, high_log_joint=rng.normal(size=8), low_log_joint=rng.normal(size=8))
    x, z = rng.normal(size=(7, 2)), rng.normal(size=(9, 2))
    high, low, extra = rng.normal(size=7), rng.normal(size=7), rng.normal(size=9)
    def evaluate(x, z, high, low, extra):
        return mf._compute_mfmc_reinforce_gradients(
            x, x, z, np.ones(2), np.array([2., 3.]), high, low, extra, baseline, True, mode,
            gradient_gain_estimation_strategy='history', gain_history_data=data,
            score_function_entropy_strategy=entropy, return_gain_samples=True,
        )
    before = evaluate(x, z, high, low, extra)
    after = evaluate(x*100, z*200, high*1000, low*2000, extra*3000)
    for index in (4, 5):
        np.testing.assert_array_equal(before[index], after[index])
    hw, lw = history_weights(data, 1.0, entropy)
    for index, scores in enumerate(gaussian_history_kernels(data)[0]):
        np.testing.assert_allclose(before[4+index], fit_history_gain(hw, lw, scores, baseline, mode))


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
@pytest.mark.parametrize('extra,control', [(0, True), (4, False)])
def test_history_does_not_evaluate_unused_gain_samples(tmp_path, family, extra, control):
    builder = VersionedBuilder()
    request = options(tmp_path, family, 3)
    request.update(gradient_gain_estimation_strategy='history', rom_extra_sample_size=extra, use_mfmc_control_variate=control)
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=builder, **request)
    assert not any('gain_history' in path for _, path in builder.evaluations)


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
def test_fixed_history_gain_unbiased_by_exact_enumeration(mode, baseline):
    import itertools
    p = 0.2
    scores = np.array([[-p], [1-p]])
    high, low = np.array([0.3, 1.7]), np.array([0.7, 1.1])
    expected = (1-p)*high[0]*scores[0]+p*high[1]*scores[1]
    gain = np.array([[0.8]]) if mode == 'matrix' else np.array(0.8) if mode == 'scalar' else np.array([0.8])
    average = np.zeros(1)
    for indices in itertools.product((0, 1), repeat=5):
        main, extra = np.array(indices[:3]), np.array(indices[3:])
        estimate = mf.loo_acv_estimator(high[main], low[main], low[extra], scores[main], scores[extra],
                                        baseline, mode, fixed_gain=gain)[0]
        average += np.prod([p if i else 1-p for i in indices])*estimate
    np.testing.assert_allclose(average, expected, atol=1e-12)


@pytest.mark.parametrize('correlated', [False, True])
def test_history_kernels_match_previous_distribution_primitives(correlated):
    rng = np.random.default_rng(554)
    mean, std = np.array([0.2, -0.1]), np.array([1.3, 0.7])
    chol = np.linalg.cholesky(np.array([[1., 0.4], [0.4, 1.]])) if correlated else None
    x = rng.normal(size=(5, 2))*std+mean
    data = dict(optimizer_samples=x, mean=mean, std=std, correlation_cholesky=chol)
    gradients, hessians = gaussian_history_kernels(data)
    for i, sample in enumerate(x):
        gradient = mf._compute_mfmc_reinforce_gradients(sample[None], sample[None], np.zeros((0, 2)),
                                                       mean, std, np.ones(1), np.ones(1), np.zeros(0),
                                                       'none', False, variational_correlation_cholesky=chol)
        np.testing.assert_allclose(gradients[0][i], gradient[0], atol=1e-12)
        np.testing.assert_allclose(gradients[1][i], gradient[1]-1, atol=1e-12)
        hessian = mf._compute_mfmc_reinforce_hessian_full(sample[None], sample[None], np.zeros((0, 2)),
                                                        mean, std, np.ones(1), np.ones(1), np.zeros(0),
                                                        'none', False, variational_correlation_cholesky=chol)
        expected = np.block([[hessians[0][i], hessians[2][i]], [hessians[2][i].T, hessians[1][i]]])
        np.testing.assert_allclose(expected, hessian, atol=1e-12)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_history_full_newton_hessian_runs_and_restarts(tmp_path, family):
    request = options(tmp_path, family, 3)
    request.update(gradient_gain_estimation_strategy='history', optimizer_method='newton',
                   optimizer_config=workflows.VINewtonOptimizerConfig(
                       max_iterations=3, gradient_norm_tolerance=0.0,
                       newton_hessian_type='full', newton_metric='natural',
                   ),
                   line_search_method='legacy',
                   line_search_config=workflows.VILegacyLineSearchConfig(
                       initial_step_size=1e-3, max_step_size=1e-3, relaxation_parameter=100.0,
                   ))
    full = workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    resumed = workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(),
                                 restart_file=str(tmp_path/'iteration_1/restart.npz'), **request)
    for a, b in zip(full, resumed):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_history_rejected_trials_do_not_replace_gain_source(tmp_path, monkeypatch, family):
    from romtools.workflows.inverse.vi import full_covariance_mf_drivers as fc
    target, name = (mf, '_evaluate_mf_vi_state') if family == 'diagonal' else (fc, '_evaluate_mf_state')
    original = getattr(target, name)
    states = []
    def evaluate(*args, **kwargs):
        state = original(*args, **kwargs)
        if len(states) == 1:
            state['elbo'] = -1e30
        elif len(states) == 2:
            state['elbo'] = 1e30
        states.append(state)
        return state
    monkeypatch.setattr(target, name, evaluate)
    request = options(tmp_path, family, 2)
    request.update(gradient_gain_estimation_strategy='history', optimizer_method='gradient',
                   optimizer_config=workflows.VIGradientOptimizerConfig(max_iterations=2, gradient_norm_tolerance=0),
                   line_search_method='legacy', line_search_config=workflows.VILegacyLineSearchConfig(
                       initial_step_size=1e-3, max_step_size=1e-3, relaxation_parameter=10.,
                       line_search_sample_growth_factor=1.,
                   ))
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    assert len(states) == 3
    for key in ('optimizer_samples', 'high_log_joint', 'low_log_joint'):
        np.testing.assert_array_equal(states[1]['gain_history_data'][key], states[2]['gain_history_data'][key])
    assert len(states[2]['training_dirs']) == 12
    gain_key = 'mfmc_alpha_mean' if family == 'diagonal' else 'mfmc_alpha_packed'
    assert not np.array_equal(states[1]['gain_history_data']['mean'], states[2]['gain_history_data']['mean'])
    assert not np.array_equal(states[1]['gain_history_data']['importance_weights'], states[2]['gain_history_data']['importance_weights'])


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_history_rejects_missing_previous_distribution_metadata(tmp_path, family):
    request = options(tmp_path, family, 2)
    request['gradient_gain_estimation_strategy'] = 'history'
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    filename = tmp_path/'iteration_1/restart.npz'
    with np.load(filename, allow_pickle=True) as saved:
        checkpoint = dict(saved)
    state = dict(checkpoint['mf_vi_state'].item())
    del state['gain_distribution_std']
    checkpoint['mf_vi_state'] = np.array(state, dtype=object)
    broken = tmp_path/'broken.npz'
    np.savez(broken, **checkpoint)
    with pytest.raises(ValueError, match='incomplete MF-VI state'):
        workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(),
                           restart_file=str(broken), **request)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_history_gain_is_available_through_auto_rom_wrapper(tmp_path, family):
    request = options(tmp_path, family, 2)
    request.update(gradient_gain_estimation_strategy='history', rom_type='gp',
                   rom_args={'normalize_parameters': True, 'normalize_targets': True})
    result = workflows.mf_vi_with_auto_rom(model=LinearModel(), **request)
    assert np.all(np.isfinite(result[0]))
    with np.load(tmp_path/'iteration_1/restart.npz', allow_pickle=True) as saved:
        assert saved['mf_vi_state'].item()['gain_history_data'] is not None


@pytest.mark.parametrize('entropy', ['analytic', 'joint'])
def test_reweighted_history_current_gradient_is_unbiased(entropy):
    rng = np.random.default_rng(616)
    old_mean, old_std = np.array([-0.5]), np.array([1.2])
    old_x = old_mean+old_std*rng.normal(size=(8, 1))
    data = dict(optimizer_samples=old_x, mean=old_mean, std=old_std, correlation_cholesky=None,
                high_log_joint=-0.75*old_x[:, 0]**2+old_x[:, 0],
                low_log_joint=-0.8*old_x[:, 0]**2+0.9*old_x[:, 0])
    mean, std = np.array([0.4]), np.array([0.8])
    proposal = dict(parameter_samples=old_x, mean=old_mean, std=old_std, correlation_cholesky=None)
    weights, _, _ = importance_weights(old_x, mean, std, None, [proposal])
    data.update(mean=mean, std=std, importance_weights=weights)
    estimates = []
    for _ in range(500):
        x, z = mean+std*rng.normal(size=(6, 1)), mean+std*rng.normal(size=(48, 1))
        high = -0.75*x[:, 0]**2+x[:, 0]
        low, extra = -0.8*x[:, 0]**2+0.9*x[:, 0], -0.8*z[:, 0]**2+0.9*z[:, 0]
        result = mf._compute_mfmc_reinforce_gradients(
            x, x, z, mean, std, high, low, extra, 'loo', True, 'scalar',
            score_function_entropy_strategy=entropy, gradient_gain_estimation_strategy='history', gain_history_data=data,
        )
        estimates.append([result[0].item(), result[1].item()])
    estimates = np.asarray(estimates)
    expected = np.array([1-1.5*mean.item(), 1-1.5*std.item()**2])
    error = estimates.std(axis=0, ddof=1)/np.sqrt(len(estimates))
    assert np.all(np.abs(estimates.mean(axis=0)-expected) < 6*error)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_history_gain_with_immediate_retraining_uses_updated_rom(tmp_path, monkeypatch, family):
    calls = capture_evaluations(monkeypatch)
    request = options(tmp_path, family, 2)
    request.update(gradient_gain_estimation_strategy='history', rom_retraining_strategy='same_sample')
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    assert calls[1][1]['rom_model'].version == 12
    assert calls[1][1]['gain_history_data']['source_directory'].endswith('iteration_0')


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
@pytest.mark.parametrize('shape', [(3,), (2, 2)])
def test_weighted_history_fit_matches_direct_covariance_reference(mode, baseline, shape):
    rng = np.random.default_rng(288)
    n = 6
    high, low = rng.normal(size=n), rng.normal(size=n)
    scores = rng.normal(size=(n,)+shape)
    flat = scores.reshape(n, -1)
    weights = np.array([0.07, 0.10, 0.13, 0.17, 0.21, 0.32])
    hb = lb = np.zeros_like(flat)
    if baseline == 'loo':
        hb, lb = np.empty((n, 1)), np.empty((n, 1))
        for j in range(n):
            keep = np.arange(n) != j
            hb[j] = np.dot(weights[keep], high[keep])/weights[keep].sum()
            lb[j] = np.dot(weights[keep], low[keep])/weights[keep].sum()
    elif baseline == 'optimal':
        denominator = np.sum(weights[:, None]*flat**2, axis=0)
        hb = np.sum(weights[:, None]*high[:, None]*flat**2, axis=0)/denominator
        lb = np.sum(weights[:, None]*low[:, None]*flat**2, axis=0)/denominator
    h, l = (high[:, None]-hb)*flat, (low[:, None]-lb)*flat
    # Independent moment formula: E_w[LH] - E_w[L]E_w[H], with w once.
    hm, lm = weights@h, weights@l
    variance = l.T@(weights[:, None]*l)-np.outer(lm, lm)
    cross = l.T@(weights[:, None]*h)-np.outer(lm, hm)
    if mode == 'scalar':
        expected = np.trace(cross)/np.trace(variance)
    elif mode == 'componentwise':
        expected = (np.diag(cross)/np.diag(variance)).reshape(shape)
    else:
        expected = np.linalg.solve(variance+1e-12*np.eye(flat.shape[1]), cross)
    actual = fit_history_gain(high, low, scores, baseline, mode, weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    if mode != 'matrix':
        wrong = mf._compute_mfmc_alpha((weights[:, None]*h).reshape(scores.shape),
                                       (weights[:, None]*l).reshape(scores.shape), mode=mode)
        assert not np.allclose(actual, wrong, rtol=1e-4, atol=1e-4)


def test_history_importance_weights_use_sample_count_weighted_proposal_mixture():
    from scipy.stats import multivariate_normal
    samples = np.array([[-1., 0.], [0., 1.], [0.5, -1.], [1., 0.3], [2., 0.2]])
    c1 = np.linalg.cholesky(np.array([[1., 0.4], [0.4, 1.]]))
    ct = np.linalg.cholesky(np.array([[1., -0.2], [-0.2, 1.]]))
    batches = [dict(parameter_samples=samples[:2], mean=np.array([-0.4, 0.3]), std=np.array([0.8, 1.1]), correlation_cholesky=c1),
               dict(parameter_samples=samples[2:], mean=np.array([0.9, -0.2]), std=np.array([1.2, 0.7]), correlation_cholesky=None)]
    mean, std = np.array([0.2, 0.1]), np.array([1.0, 0.9])
    weights, logs, ess = importance_weights(samples, mean, std, ct, batches)
    target_cov = (std[:, None]*ct)@(std[:, None]*ct).T
    target = multivariate_normal.pdf(samples, mean, target_cov)
    proposal = np.zeros(len(samples))
    for batch in batches:
        chol = np.diag(batch['std']) if batch['correlation_cholesky'] is None else batch['std'][:, None]*batch['correlation_cholesky']
        proposal += len(batch['parameter_samples'])/len(samples)*multivariate_normal.pdf(samples, batch['mean'], chol@chol.T)
    raw = target/proposal
    np.testing.assert_allclose(logs, np.log(raw), atol=1e-12)
    np.testing.assert_allclose(weights, raw/raw.sum(), atol=1e-12)
    np.testing.assert_allclose(ess, 1/np.sum(weights**2), atol=1e-12)


def test_history_window_trims_partial_batches_and_preserves_sampling_distributions():
    def batch(directory, count, mean):
        points = np.arange(count, dtype=float)[:, None]+mean
        return dict(parameter_samples=points, optimizer_samples=points, high_log_joint=points[:, 0],
                    log_priors=points[:, 0]*0, log_transform_jacobian=points[:, 0]*0,
                    mean=np.array([mean]), std=np.ones(1), correlation_cholesky=None,
                    sample_ids=[f'{directory}::{j}' for j in range(count)], source_directory=directory)
    first, second = batch('first', 3, 0.), batch('second', 5, 2.)
    retained = append_training_batch([first], second, 6)
    assert [len(item['parameter_samples']) for item in retained] == [1, 5]
    assert retained[0]['sample_ids'] == ['first::2']
    np.testing.assert_array_equal(retained[0]['mean'], first['mean'])
    np.testing.assert_array_equal(retained[0]['optimizer_samples'], first['optimizer_samples'][-1:])
    assert len(first['parameter_samples']) == 3  # No mutation of the accepted source state.
    all_points = np.concatenate([item['optimizer_samples'] for item in retained])
    probabilities, _, _ = importance_weights(all_points, np.array([1.]), np.ones(1), None, retained)
    from scipy.stats import norm
    expected = norm.pdf(all_points[:, 0], 1., 1.)/(norm.pdf(all_points[:, 0], 0., 1.)/6+5*norm.pdf(all_points[:, 0], 2., 1.)/6)
    np.testing.assert_allclose(probabilities, expected/expected.sum(), atol=1e-12)


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
def test_history_effective_sample_size_guard_returns_zero_gain(mode):
    rng = np.random.default_rng(123)
    high, low, scores = rng.normal(size=4), rng.normal(size=4), rng.normal(size=(4, 3))
    for weights in (np.array([0.99, 0.01, 0., 0.]), np.zeros(4)):
        gain = fit_history_gain(high, low, scores, 'loo', mode, weights)
        np.testing.assert_array_equal(gain, np.zeros_like(gain))


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_history_window_evicts_old_batches_and_reuses_all_unchanged_rom_predictions(tmp_path, monkeypatch, family):
    calls = capture_evaluations(monkeypatch)
    request = options(tmp_path, family, 5)
    request.update(gradient_gain_estimation_strategy='history', rom_tolerance=1e10)
    builder = VersionedBuilder()
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=builder, **request)
    assert len(builder.evaluations) == 50
    assert not any('gain_history' in path for _, path in builder.evaluations)
    for index, (_, state) in enumerate(calls):
        batches = state['gain_training_batches']
        assert sum(len(item['parameter_samples']) for item in batches) == min(6*(index+1), 12)
        np.testing.assert_array_equal(np.concatenate([item['parameter_samples'] for item in batches]), state['training_parameters'][-12:])
        if index >= 3:
            assert state['gain_history_data']['source_directories'] == [str(tmp_path/f'iteration_{index-2}'), str(tmp_path/f'iteration_{index-1}')]
            assert len(state['gain_history_data']['optimizer_samples']) == 12


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_history_restart_rejects_missing_proposal_and_changed_window(tmp_path, family):
    request = options(tmp_path, family, 2)
    request['gradient_gain_estimation_strategy'] = 'history'
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    original = tmp_path/'iteration_1/restart.npz'
    with np.load(original, allow_pickle=True) as saved:
        checkpoint = dict(saved)
    state = dict(checkpoint['mf_vi_state'].item())
    del state['gain_training_batches'][0]['mean']
    checkpoint['mf_vi_state'] = np.array(state, dtype=object)
    broken = tmp_path/'missing_proposal.npz'
    np.savez(broken, **checkpoint)
    with pytest.raises(ValueError, match='history-window metadata'):
        workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), restart_file=str(broken), **request)
    request['max_rom_training_history'] = 3
    with pytest.raises(ValueError, match='training-window limit'):
        workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), restart_file=str(original), **request)


def test_reweighted_history_gain_matches_current_gaussian_population_moments():
    rng = np.random.default_rng(100)
    x = np.concatenate([rng.normal(mean, 1.5, size=(5000, 1)) for mean in (-1., 0.5)])
    batches = [dict(parameter_samples=x[:5000], mean=np.array([-1.]), std=np.array([1.5]), correlation_cholesky=None),
               dict(parameter_samples=x[5000:], mean=np.array([0.5]), std=np.array([1.5]), correlation_cholesky=None)]
    weights, _, ess = importance_weights(x, np.array([1.]), np.array([0.8]), None, batches)
    high, low, scores = -x[:, 0]**2, -0.5*x[:, 0]**2+x[:, 0], (x-1)/0.64
    actual = fit_history_gain(high, low, scores, 'none', 'scalar', weights)
    unweighted = fit_history_gain(high, low, scores, 'none', 'scalar')
    # q=N(1, .8^2), H=-X^2*s, L=(-X^2/2+X)*s, s=(X-1)/.8^2.
    # Analytic Gaussian second/fourth/sixth moments give these covariances.
    expected = (-0.5/0.64+7.5*0.64)/(0.25/0.64-1.5+3.75*0.64)
    assert ess > 4000
    assert abs(actual-expected) < 0.15
    assert abs(unweighted-expected) > 1
