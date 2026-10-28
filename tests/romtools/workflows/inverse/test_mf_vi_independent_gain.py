import numpy as np
import pytest

import romtools.workflows as workflows
from romtools.workflows.inverse.vi import mf_drivers as mf
from romtools.workflows.inverse.vi import full_covariance_mf_drivers as fc
from tests.romtools.workflows.inverse.test_mf_vi_delayed_adaptation import (
    LinearModel, VersionedBuilder, options, capture_evaluations,
)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family',['diagonal','full_covariance'])
@pytest.mark.parametrize('mode',['scalar','componentwise','matrix'])
@pytest.mark.parametrize('baseline',['none','loo','optimal'])
def test_independent_pairs_fit_gain_without_entering_estimator_or_training(tmp_path,monkeypatch,family,mode,baseline):
    calls=capture_evaluations(monkeypatch)
    request=options(tmp_path,family,3)
    request.update(gradient_gain_estimation_strategy='independent',gain_sample_size=7,
                   mfmc_control_variate_mode=mode,baseline_method=baseline)
    model,builder=LinearModel(),VersionedBuilder()
    workflows.run_mf_vi(model=model,rom_model_builder=builder,**request)
    assert len(model.evaluations)==18+14
    assert len(builder.evaluations)==30+14
    assert calls[0][1]['gain_independent_data'] is None
    for iteration,(_,state) in enumerate(calls[1:],start=1):
        data=state['gain_independent_data']
        assert len(data['optimizer_samples'])==7
        assert len(state['parameter_samples_fom'])==6
        assert not np.isin(data['parameter_samples'],state['parameter_samples']).any()
        assert not np.isin(data['parameter_samples'],state['training_parameters']).any()
        assert state['hessian_gain_estimation_strategy']=='same_sample'
        for gain in state['hessian_gain_samples'].values():
            np.testing.assert_array_equal(gain,np.broadcast_to(gain[0],gain.shape))
        if family=='diagonal':
            h,l=mf.history_weights(data,1.)
            for name,scores in zip(('mean','log_std'),mf.gaussian_history_kernels(data)[0]):
                expected=mf.fit_history_gain(h,l,scores,baseline,mode)
                np.testing.assert_allclose(state['mfmc_alpha_'+name],expected)
                np.testing.assert_allclose(state['gradient_gain_samples'][name],np.broadcast_to(expected,state['gradient_gain_samples'][name].shape))
        else:
            h,l=mf.history_weights(data,1.)
            scores=np.hstack(fc.natural_score_terms(data['optimizer_samples'],data['mean'],data['cholesky']))
            expected=mf.fit_history_gain(h,l,scores,baseline,mode)
            with np.load(tmp_path/f'iteration_{iteration}/restart.npz',allow_pickle=True) as saved:
                np.testing.assert_allclose(saved['mf_vi_state'].item()['mfmc_alpha_packed'],expected)
    for _,_,parameters in builder.builds:
        assert not np.isin(parameters,np.concatenate([state['gain_independent_data']['parameter_samples'] for _,state in calls[1:]])).any()


@pytest.mark.parametrize('mode',['scalar','componentwise','matrix'])
@pytest.mark.parametrize('baseline',['none','loo','optimal'])
@pytest.mark.parametrize('entropy',['analytic','joint'])
def test_gain_independent_of_current_evaluation_samples(mode,baseline,entropy):
    rng=np.random.default_rng(48)
    data=dict(optimizer_samples=rng.normal(size=(9,2)),mean=np.zeros(2),std=np.ones(2),
              correlation_cholesky=None,high_log_joint=rng.normal(size=9),low_log_joint=rng.normal(size=9))
    x,z=rng.normal(size=(8,2)),rng.normal(size=(6,2))
    h,l,e=rng.normal(size=8),rng.normal(size=8),rng.normal(size=6)
    def evaluate(x,z,h,l,e):
        return mf._compute_mfmc_reinforce_gradients(
            x,x,z,np.zeros(2),np.ones(2),h,l,e,baseline,True,mode,
            gradient_gain_estimation_strategy='independent',gain_independent_data=data,
            score_function_entropy_strategy=entropy,return_gain_samples=True)
    before=evaluate(x,z,h,l,e)
    after=evaluate(x*10,z*20,h*30,l*40,e*50)
    for index in (4,5):np.testing.assert_array_equal(before[index],after[index])
    for name in ('mean','log_std'):np.testing.assert_array_equal(before[8][name],after[8][name])


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family',['diagonal','full_covariance'])
@pytest.mark.parametrize('checkpoint',[0,1])
def test_independent_restart_no_refit_and_same_results(tmp_path,family,checkpoint):
    request=options(tmp_path/'full',family,4)
    request.update(gradient_gain_estimation_strategy='independent',gain_sample_size=7)
    full=workflows.run_mf_vi(model=LinearModel(),rom_model_builder=VersionedBuilder(),**request)
    request=options(tmp_path/'partial',family,checkpoint+1)
    request.update(gradient_gain_estimation_strategy='independent',gain_sample_size=7)
    workflows.run_mf_vi(model=LinearModel(),rom_model_builder=VersionedBuilder(),**request)
    request=options(tmp_path/'partial',family,4)
    request.update(gradient_gain_estimation_strategy='independent',gain_sample_size=7,
                   restart_file=str(tmp_path/f'partial/iteration_{checkpoint}/restart.npz'))
    model=LinearModel()
    resumed=workflows.run_mf_vi(model=model,rom_model_builder=VersionedBuilder(),**request)
    assert len(model.evaluations)==(3-checkpoint)*13
    for a,b in zip(full,resumed):np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family',['diagonal','full_covariance'])
@pytest.mark.parametrize('extra,control',[(0,True),(4,False)])
def test_skip_unused_gain_batches(tmp_path,family,extra,control):
    request=options(tmp_path,family,2)
    request.update(gradient_gain_estimation_strategy='independent',rom_extra_sample_size=extra,use_mfmc_control_variate=control)
    model=LinearModel()
    workflows.run_mf_vi(model=model,rom_model_builder=VersionedBuilder(),**request)
    assert len(model.evaluations)==12
    with np.load(tmp_path/'iteration_1/restart.npz',allow_pickle=True) as saved:
        assert saved['mf_vi_state'].item()['gain_independent_data'] is None


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family',['diagonal','full_covariance'])
def test_auto_wrapper_and_default_batch_size(tmp_path,family):
    request=options(tmp_path,family,2)
    request.update(gradient_gain_estimation_strategy='independent',rom_args={'normalize_parameters':True,'normalize_targets':True})
    workflows.mf_vi_with_auto_rom(model=LinearModel(),**request)
    with np.load(tmp_path/'iteration_1/restart.npz',allow_pickle=True) as saved:
        assert len(saved['mf_vi_state'].item()['gain_independent_data']['optimizer_samples'])==6


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family',['diagonal','full_covariance'])
@pytest.mark.parametrize('problem',['count','missing','data'])
def test_independent_restart_metadata_validation(tmp_path,family,problem):
    request=options(tmp_path,family,2)
    request.update(gradient_gain_estimation_strategy='independent',gain_sample_size=7)
    workflows.run_mf_vi(model=LinearModel(),rom_model_builder=VersionedBuilder(),**request)
    with np.load(tmp_path/'iteration_1/restart.npz',allow_pickle=True) as saved:checkpoint=dict(saved)
    state=checkpoint['mf_vi_state'].item()
    if problem=='count':request['gain_sample_size']=8
    if problem=='missing':del state['gain_independent_data']
    if problem=='data':state['gain_independent_data']=None
    checkpoint['mf_vi_state']=np.array(state,dtype=object)
    np.savez(tmp_path/'broken.npz',**checkpoint)
    with pytest.raises(ValueError,match='gain_sample_size|independent gain'):
        workflows.run_mf_vi(model=LinearModel(),rom_model_builder=VersionedBuilder(),
                           restart_file=str(tmp_path/'broken.npz'),**request)


@pytest.mark.parametrize('value',[True,False,0,-1,1.5,'6'])
def test_gain_sample_count_validation(value):
    with pytest.raises(ValueError,match='gain_sample_size'):mf._validate_gain_sample_size(value)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family',['diagonal','full_covariance'])
@pytest.mark.parametrize('gain_size',[None,7])
def test_retries_draw_new_gain_pairs_and_default_count_tracks_trial(tmp_path,monkeypatch,family,gain_size):
    target,name=(mf,'_evaluate_mf_vi_state') if family=='diagonal' else (fc,'_evaluate_mf_state')
    original=getattr(target,name);states=[]
    def evaluate(*args,**kwargs):
        state=original(*args,**kwargs)
        if len(states)==1:state['elbo']=-1e30
        if len(states)==2:state['elbo']=1e30
        states.append(state)
        return state
    monkeypatch.setattr(target,name,evaluate)
    request=options(tmp_path,family,2)
    request.update(gradient_gain_estimation_strategy='independent',gain_sample_size=gain_size,
                   optimizer_method='gradient',optimizer_config=workflows.VIGradientOptimizerConfig(max_iterations=2,gradient_norm_tolerance=0),
                   line_search_method='legacy',line_search_config=workflows.VILegacyLineSearchConfig(
                       initial_step_size=1e-3,max_step_size=1e-3,relaxation_parameter=10,line_search_sample_growth_factor=2))
    workflows.run_mf_vi(model=LinearModel(),rom_model_builder=VersionedBuilder(),**request)
    assert len(states)==3
    a,b=(state['gain_independent_data'] for state in states[1:])
    assert len(a['optimizer_samples'])==(6 if gain_size is None else gain_size)
    assert len(b['optimizer_samples'])==(12 if gain_size is None else gain_size)
    assert not np.isin(a['parameter_samples'],b['parameter_samples']).any()
    assert not np.isin(a['parameter_samples'],states[2]['training_parameters']).any()


@pytest.mark.parametrize('entropy',['analytic','joint'])
@pytest.mark.parametrize('baseline',['none','loo','optimal'])
def test_independent_fixed_rom_monte_carlo_unbiasedness(entropy,baseline):
    rng=np.random.default_rng(461)
    mean,std=np.array([.4]),np.array([.8])
    estimates=[]
    for _ in range(1000):
        gx=mean+std*rng.normal(size=(9,1))
        x,z=mean+std*rng.normal(size=(6,1)),mean+std*rng.normal(size=(12,1))
        high=lambda points:-.75*points[:,0]**2+points[:,0]
        low=lambda points:-.8*points[:,0]**2+.9*points[:,0]
        data=dict(optimizer_samples=gx,high_log_joint=high(gx),low_log_joint=low(gx),mean=mean,std=std,correlation_cholesky=None)
        result=mf._compute_mfmc_reinforce_gradients(
            x,x,z,mean,std,high(x),low(x),low(z),baseline,True,'scalar',
            gradient_gain_estimation_strategy='independent',gain_independent_data=data,
            score_function_entropy_strategy=entropy)
        estimates.append([result[0].item(),result[1].item()])
    estimates=np.array(estimates)
    expected=[1-1.5*mean.item(),1-1.5*std.item()**2]
    error=estimates.std(axis=0,ddof=1)/np.sqrt(len(estimates))
    assert np.all(np.abs(estimates.mean(axis=0)-expected)<5*error)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family',['diagonal','full_covariance'])
def test_independent_requires_mc(tmp_path,family):
    request=options(tmp_path,family,1)
    request.update(gradient_gain_estimation_strategy='independent',sampling_method='rqmc')
    model=LinearModel()
    with pytest.raises(ValueError,match='independent paired samples'):
        workflows.run_mf_vi(model=model,rom_model_builder=VersionedBuilder(),**request)
    assert not model.evaluations


@pytest.mark.mpi_skip
def test_hessian_only_auxiliary_does_not_draw_gradient_gain_pairs(tmp_path,monkeypatch):
    calls=capture_evaluations(monkeypatch)
    request=options(tmp_path,iterations=2)
    request.update(gradient_gain_estimation_strategy='independent',gain_sample_size=7,optimizer_method='newton',
                   optimizer_config=workflows.VINewtonOptimizerConfig(max_iterations=2,gradient_norm_tolerance=0,
                       newton_curvature_strategy='independent',newton_hessian_num_samples=6),
                   line_search_method='legacy',line_search_config=workflows.VILegacyLineSearchConfig(
                       initial_step_size=1e-3,max_step_size=1e-3,relaxation_parameter=1e10))
    workflows.run_mf_vi(model=LinearModel(),rom_model_builder=VersionedBuilder(),**request)
    auxiliary=[state for args,state in calls if not args['allow_rom_retraining']]
    assert len(auxiliary)==1
    assert auxiliary[0]['gain_independent_data'] is None


@pytest.mark.mpi_skip
@pytest.mark.parametrize('family',['diagonal','full_covariance'])
def test_newton_independent_gain_runs(tmp_path,family):
    request=options(tmp_path,family,2)
    request.update(gradient_gain_estimation_strategy='independent',gain_sample_size=7,optimizer_method='newton',
                   optimizer_config=workflows.VINewtonOptimizerConfig(max_iterations=2,gradient_norm_tolerance=0,
                       newton_metric='natural',newton_hessian_type='full'),
                   line_search_method='legacy',line_search_config=workflows.VILegacyLineSearchConfig(
                       initial_step_size=1e-3,max_step_size=1e-3,relaxation_parameter=1e10))
    result=workflows.run_mf_vi(model=LinearModel(),rom_model_builder=VersionedBuilder(),**request)
    assert all(np.all(np.isfinite(value)) for value in result)
