import inspect

import numpy as np
import pytest

import romtools.workflows as workflows
from romtools.workflows.inverse.vi import mf_drivers as mf
from tests.romtools.workflows.inverse.test_mf_vi_delayed_adaptation import LinearModel, VersionedBuilder, options, capture_evaluations


@pytest.mark.parametrize('mode',['scalar','componentwise','matrix'])
@pytest.mark.parametrize('shape',[(3,),(2,2)])
@pytest.mark.parametrize('coefficient',[0.,.3,-.4,2.])
def test_fixed_gain_matches_analytic_correction(mode,shape,coefficient):
    rng=np.random.default_rng(47)
    h,l,e=rng.normal(size=(8,)+shape),rng.normal(size=(8,)+shape),rng.normal(size=(11,)+shape)
    result,gain=mf._mfmc_gradient_estimator(h,l,e,False,mode,mfmc_fixed_gain=coefficient)
    expected=h.mean(axis=0)+coefficient*11/19*(e.mean(axis=0)-l.mean(axis=0))
    np.testing.assert_allclose(result,expected)
    np.testing.assert_array_equal(gain,mf._constant_mfmc_gain(coefficient,shape,mode))
    result,zero=mf._mfmc_gradient_estimator(h,l,np.zeros((0,)+shape),False,mode,mfmc_fixed_gain=coefficient)
    np.testing.assert_allclose(result,h.mean(axis=0))
    np.testing.assert_array_equal(zero,mf._constant_mfmc_gain(0.,shape,mode))


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family',['diagonal','full_covariance'])
@pytest.mark.parametrize('mode',['scalar','componentwise','matrix'])
@pytest.mark.parametrize('strategy',['same_sample','loo','k_fold','history','independent'])
def test_configured_fixed_gain_reaches_gradients_hessians_and_bootstrap(tmp_path,monkeypatch,family,mode,strategy):
    calls=capture_evaluations(monkeypatch)
    request=options(tmp_path,family,2)
    request.update(use_mfmc_control_variate=False,mfmc_fixed_gain=.3,
                   mfmc_control_variate_mode=mode,gradient_gain_estimation_strategy=strategy)
    model=LinearModel()
    workflows.run_mf_vi(model=model,rom_model_builder=VersionedBuilder(),**request)
    assert len(model.evaluations)==12
    for _,state in calls:
        assert state['mfmc_fixed_gain']==.3
        for gain in (*state['gradient_gain_samples'].values(),*state['hessian_gain_samples'].values()):
            shape=() if mode=='scalar' else ((1,1) if mode=='matrix' else (1,))
            # Hessian componentwise blocks have trailing shape (1,1).
            if mode=='componentwise':shape=gain.shape[1:]
            expected=mf._constant_mfmc_gain(.3,shape if mode!='matrix' else (1,),mode)
            np.testing.assert_array_equal(gain,np.broadcast_to(expected,gain.shape))
    if family=='full_covariance':
        with np.load(tmp_path/'iteration_1/restart.npz',allow_pickle=True) as saved:
            gain=saved['mf_vi_state'].item()['mfmc_alpha_packed']
            np.testing.assert_array_equal(gain,mf._constant_mfmc_gain(.3,(2,),mode))


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family',['diagonal','full_covariance'])
def test_fixed_gain_restart_and_configuration_validation(tmp_path,family):
    request=options(tmp_path,family,3)
    request.update(use_mfmc_control_variate=False,mfmc_fixed_gain=.3)
    full=workflows.run_mf_vi(model=LinearModel(),rom_model_builder=VersionedBuilder(),**request)
    request['restart_file']=str(tmp_path/'iteration_1/restart.npz')
    resumed=workflows.run_mf_vi(model=LinearModel(),rom_model_builder=VersionedBuilder(),**request)
    for a,b in zip(full,resumed):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
    request['mfmc_fixed_gain']=.7
    with pytest.raises(ValueError,match='mfmc_fixed_gain'):
        workflows.run_mf_vi(model=LinearModel(),rom_model_builder=VersionedBuilder(),**request)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family',['diagonal','full_covariance'])
def test_auto_wrapper_fixed_gain(tmp_path,family):
    request=options(tmp_path,family,2)
    request.update(use_mfmc_control_variate=False,mfmc_fixed_gain=.7,
                   rom_args={'normalize_parameters':True,'normalize_targets':True})
    workflows.mf_vi_with_auto_rom(model=LinearModel(),**request)
    with np.load(tmp_path/'iteration_1/restart.npz',allow_pickle=True) as saved:
        assert saved['mf_vi_state'].item()['mfmc_fixed_gain']==.7


@pytest.mark.mpi_skip
@pytest.mark.parametrize('invalid',[True,None,np.nan,np.inf,-np.inf,[1.,2.],'1'])
def test_bad_fixed_gain_is_rejected_before_model_evaluations(tmp_path,invalid):
    request=options(tmp_path,iterations=1)
    request['mfmc_fixed_gain']=invalid
    model=LinearModel()
    with pytest.raises(ValueError,match='mfmc_fixed_gain'):
        workflows.run_mf_vi(model=model,rom_model_builder=VersionedBuilder(),**request)
    assert not model.evaluations


def test_fixed_gain_defaults_to_one():
    for f in (workflows.run_mf_vi,workflows.mf_vi_with_auto_rom):
        assert inspect.signature(f).parameters['mfmc_fixed_gain'].default==1.


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family',['diagonal','full_covariance'])
def test_fixed_gain_newton_curvature(tmp_path,family):
    request=options(tmp_path,family,2)
    request.update(use_mfmc_control_variate=False,mfmc_fixed_gain=.4,optimizer_method='newton',
                   optimizer_config=workflows.VINewtonOptimizerConfig(max_iterations=2,gradient_norm_tolerance=0,
                       newton_metric='natural',newton_hessian_type='full'),
                   line_search_method='legacy',line_search_config=workflows.VILegacyLineSearchConfig(
                       initial_step_size=1e-3,max_step_size=1e-3,relaxation_parameter=1e10))
    result=workflows.run_mf_vi(model=LinearModel(),rom_model_builder=VersionedBuilder(),**request)
    assert all(np.all(np.isfinite(value)) for value in result)
    with np.load(tmp_path/'iteration_1/restart.npz',allow_pickle=True) as saved:
        state=saved['mf_vi_state'].item()
        for gain in state['hessian_gain_samples'].values():np.testing.assert_allclose(gain,.4)
        if family=='full_covariance':np.testing.assert_allclose(state['ordinary_hessian_gain_samples'],.4)


@pytest.mark.parametrize('mode',['scalar','componentwise','matrix'])
def test_archive_helpers_obey_fixed_gradient_and_hessian_gain(mode):
    from romtools.workflows.inverse.vi.sample_reuse import _mf_gradient_from_reuse
    from romtools.workflows.inverse.vi.sample_reuse import _mf_hessian_from_reuse
    rng=np.random.default_rng(33)
    x,z=rng.normal(size=(8,2)),rng.normal(size=(9,2))
    h,l,e=rng.normal(size=8),rng.normal(size=8),rng.normal(size=9)
    weights=np.ones(8)
    def evaluate(coefficient):
        gradient=_mf_gradient_from_reuse(x,z,np.zeros(2),np.zeros(2),None,h,l,e,weights,weights,
                                        'none',1.,'analytic',False,mode,mfmc_fixed_gain=coefficient)
        hessian=_mf_hessian_from_reuse(x,z,np.zeros(2),np.zeros(2),None,h,l,e,weights,weights,
                                       'none',1.,False,mode,mfmc_fixed_gain=coefficient)
        return gradient,hessian
    zero,unit,custom=evaluate(0.),evaluate(1.),evaluate(.3)
    for index in (0,1):
        np.testing.assert_allclose(custom[0][index],zero[0][index]+.3*(unit[0][index]-zero[0][index]))
    np.testing.assert_allclose(custom[1],zero[1]+.3*(unit[1]-zero[1]))
    for gain in custom[0][2:4]:np.testing.assert_array_equal(gain,mf._constant_mfmc_gain(.3,(2,),mode))
