import numpy as np
import pytest

import romtools.workflows as workflows
from romtools.workflows.inverse.vi import mf_drivers as mf
from romtools.workflows.inverse.vi import full_covariance_mf_drivers as fc
from tests.romtools.workflows.inverse.test_mf_vi_delayed_adaptation import LinearModel, VersionedBuilder, options


@pytest.mark.parametrize('n,m', [(2, 2), (8, 3), (3, 19)])
@pytest.mark.parametrize('correlation', [-2., 0., 2.])
def test_mf_variance_matches_covariance_expansion(n, m, correlation):
    rng = np.random.default_rng(288)
    l, e = rng.normal(size=n), 3*rng.normal(size=m)
    h = correlation*l+rng.normal(size=n)
    r = m/(n+m)
    expected = (np.var(h, ddof=1)+r*r*np.var(l, ddof=1)
                -2*r*np.cov(h, l, ddof=1)[0, 1])/n + r*r*np.var(e, ddof=1)/m
    np.testing.assert_allclose(mf._compute_mfmc_standard_error(h, l, e)**2, expected, atol=1e-12)


def test_identical_models_have_lf_only_variance_without_phantom_terms():
    paired, extra = np.array([-1., 1.]), np.array([-1., 1.])
    # Both sample variances equal 2; perfect control yields 2/(n+m).
    result = mf._compute_mfmc_standard_error(paired, paired, extra)
    np.testing.assert_allclose(result**2, 0.5)
    assert result < mf._compute_standard_error(paired)


def test_variance_matches_empirical_sampling_distribution():
    rng = np.random.default_rng(67)
    n, m, count = 8, 16, 20000
    l = rng.normal(size=(count, n))
    h = 2*l+0.5*rng.normal(size=(count, n))
    e = rng.normal(size=(count, m))
    r = m/(n+m)
    estimates = h.mean(axis=1)+r*(e.mean(axis=1)-l.mean(axis=1))
    exact = ((2-r)**2+0.25)/n+r*r/m
    np.testing.assert_allclose(estimates.var(ddof=1), exact, rtol=0.03)
    plug_in = [mf._compute_mfmc_standard_error(hh, ll, ee)**2 for hh, ll, ee in zip(h[:3000], l[:3000], e[:3000])]
    np.testing.assert_allclose(np.mean(plug_in), exact, rtol=0.03)


def test_stable_offsets_constant_terms_and_scaling():
    h = np.array([1., 3., -2., 4.])
    l = np.array([2., -1., 3., 5.])
    e = np.array([0., 1., -1., 3.])
    plain = mf._compute_mfmc_standard_error(h, l, e)
    shifted = mf._compute_mfmc_standard_error(h+2**40, l-2**39, e-2**39)
    np.testing.assert_array_equal(plain, shifted)
    assert mf._compute_mfmc_standard_error(np.full(4, 9.), np.full(4, 3.), np.full(5, 2.)) == 0
    state = dict(log_joint_terms_fom=h, log_joint_terms_rom_base=l, log_joint_terms_rom_only=e, entropy=1e100)
    for scale in (-2., 0., 3.):
        assert mf._compute_state_elbo_standard_error(state, scale) == abs(scale)*plain


@pytest.mark.parametrize('n,m', [(1, 1), (1, 3), (3, 1), (4, 0)])
def test_small_sample_fallbacks(n, m):
    h, l, e = np.arange(n, dtype=float), 2*np.arange(n, dtype=float), np.arange(m, dtype=float)
    r = m/(n+m)
    expected = np.hypot(mf._compute_standard_error(h-r*l), r*mf._compute_standard_error(e))
    assert mf._compute_mfmc_standard_error(h, l, e) == expected


def test_mismatched_pairs_rejected():
    with pytest.raises(ValueError, match='matching'):
        mf._compute_mfmc_standard_error(np.ones(5), np.ones(4), np.ones(3))


@pytest.mark.parametrize('window,expected', [(1, (7., 70.)), (2, (10., 2.)), (3, (10., 1.)), (9, (10., 1.))])
def test_reference_selection_and_ties(window, expected):
    assert mf._select_mf_elbo_reference([10., 10., 7.], [1., 2., 70.], window) == expected


def stochastic_options(directory, family, iterations):
    request = options(directory, family, iterations)
    request.update(optimizer_method='gradient',
                   optimizer_config=workflows.VIGradientOptimizerConfig(max_iterations=iterations, gradient_norm_tolerance=0),
                   line_search_method='stochastic_nonmonotone',
                   line_search_config=workflows.VIStochasticNonmonotoneLineSearchConfig(
                       initial_step_size=1e-3, max_step_size=1e-3, step_size_growth_factor=1,
                       line_search_sample_growth_factor=1, line_search_nonmonotone_window=3,
                       line_search_armijo_coefficient=0, line_search_uncertainty_sigma=1))
    return request


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_acceptance_uses_reference_uncertainty_and_rejections_do_not_enter_history(tmp_path, monkeypatch, family):
    target, name = (mf, '_evaluate_mf_vi_state') if family == 'diagonal' else (fc, '_evaluate_mf_state')
    original = getattr(target, name)
    states = []
    def evaluate(*args, **kwargs):
        state = original(*args, **kwargs)
        index = len(states)
        state['elbo'] = [10., 9., 8., 8.8][index]
        state['fixture_se'] = [1., 100., 1., 1.][index]
        states.append(state)
        return state
    monkeypatch.setattr(target, name, evaluate)
    monkeypatch.setattr(mf, '_compute_state_elbo_standard_error', lambda state, scale: state['fixture_se'])
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(),
                       **stochastic_options(tmp_path, family, 3))
    assert len(states) == 4  # The third evaluation is rejected despite the latest state's large SE.
    with np.load(tmp_path/'iteration_2/restart.npz', allow_pickle=True) as saved:
        np.testing.assert_array_equal(saved['accepted_elbo_history'], [10., 9., 8.8])
        np.testing.assert_array_equal(saved['accepted_elbo_standard_error_history'], [1., 100., 1.])


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_stochastic_restart_reproducibility(tmp_path, family):
    full = workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(),
                              **stochastic_options(tmp_path/'full', family, 3))
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(),
                       **stochastic_options(tmp_path/'partial', family, 2))
    resumed = workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(),
                                 restart_file=str(tmp_path/'partial/iteration_1/restart.npz'),
                                 **stochastic_options(tmp_path/'partial', family, 3))
    for a, b in zip(full, resumed):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    with np.load(tmp_path/'full/iteration_2/restart.npz', allow_pickle=True) as a:
        with np.load(tmp_path/'partial/iteration_2/restart.npz', allow_pickle=True) as b:
            np.testing.assert_array_equal(a['accepted_elbo_standard_error_history'], b['accepted_elbo_standard_error_history'])


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
@pytest.mark.parametrize('malformed', ['missing', 'length', 'negative', 'nonfinite'])
def test_stochastic_restart_requires_valid_uncertainty_history(tmp_path, family, malformed):
    request = stochastic_options(tmp_path, family, 1)
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(), **request)
    with np.load(tmp_path/'iteration_0/restart.npz', allow_pickle=True) as saved:
        checkpoint = dict(saved)
    key = 'accepted_elbo_standard_error_history'
    if malformed == 'missing':
        del checkpoint[key]
    else:
        checkpoint[key] = {'length':np.zeros(2), 'negative':np.array([-1.]), 'nonfinite':np.array([np.nan])}[malformed]
    np.savez(tmp_path/'broken.npz', **checkpoint)
    with pytest.raises(ValueError, match='ELBO uncertainty history'):
        workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(),
                           restart_file=str(tmp_path/'broken.npz'), **request)


def test_legacy_uncertainty_history_can_initialize_locally():
    state = dict(elbo=2., log_joint_terms_fom=np.array([1., 2., 3.]),
                 log_joint_terms_rom_base=np.zeros(3), log_joint_terms_rom_only=np.zeros(0))
    values, errors = mf._restore_mf_elbo_uncertainty_history({}, [10., 2.], state, 1.)
    assert values == [2.]
    np.testing.assert_allclose(errors, [mf._compute_standard_error(state['log_joint_terms_fom'])])


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family', ['diagonal', 'full_covariance'])
def test_old_adam_checkpoint_fallback_preserves_initial_convergence_reference(tmp_path, family):
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(),
                       **options(tmp_path, family, 2))
    with np.load(tmp_path/'iteration_1/restart.npz', allow_pickle=True) as saved:
        checkpoint = dict(saved)
        initial = float(saved['accepted_elbo_history'][0])
    del checkpoint['accepted_elbo_standard_error_history']
    if family == 'full_covariance':
        checkpoint.pop('initial_elbo_reference', None)
    np.savez(tmp_path/'older.npz', **checkpoint)
    workflows.run_mf_vi(model=LinearModel(), rom_model_builder=VersionedBuilder(),
                       restart_file=str(tmp_path/'older.npz'), **options(tmp_path, family, 3))
    with np.load(tmp_path/'iteration_2/restart.npz', allow_pickle=True) as saved:
        assert float(saved['initial_elbo_reference']) == initial
        assert len(saved['accepted_elbo_history']) == len(saved['accepted_elbo_standard_error_history']) == 2
