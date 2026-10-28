import numpy as np
import pytest

import romtools.workflows as workflows
from romtools.workflows.inverse.vi import drivers as vi
from romtools.workflows.inverse.vi import mf_drivers as mf
from romtools.workflows.inverse.vi import full_covariance_drivers as fc_vi
from romtools.workflows.inverse.vi import full_covariance_mf_drivers as fc_mf
from tests.romtools.workflows.inverse.test_mf_vi_delayed_adaptation import LinearModel, VersionedBuilder, options


def request(directory, workflow, family, iterations, method='legacy', optimizer='gradient', adaptive=False):
    result = options(directory, family, iterations)
    if workflow == 'vi':
        for key in ('fom_sample_size','rom_extra_sample_size','max_rom_training_history','rom_tolerance',
                    'rom_retraining_strategy','gradient_gain_estimation_strategy','rom_evaluation_concurrency'):
            result.pop(key)
        result['sample_size'] = 6
        result['evaluation_concurrency'] = result.pop('fom_evaluation_concurrency')
    result['optimizer_method'] = optimizer
    if optimizer == 'gradient':
        result['optimizer_config'] = workflows.VIGradientOptimizerConfig(max_iterations=iterations,gradient_norm_tolerance=0)
    else:
        result['optimizer_config'] = workflows.VINewtonOptimizerConfig(
            max_iterations=iterations,gradient_norm_tolerance=0,
            newton_additive_regularization=.01,newton_adaptive_regularization=adaptive,
            newton_regularization_increase_factor=3,newton_regularization_max_multiplier=5,
            newton_regularization_decrease_factor=2,
            newton_metric='natural' if family=='full_covariance' else 'standard')
    line_type = workflows.VILegacyLineSearchConfig if method=='legacy' else workflows.VIStochasticNonmonotoneLineSearchConfig
    line_kwargs = dict(initial_step_size=.008,max_step_size=.004,step_size_decay_factor=2,
                       step_size_growth_factor=1.5,line_search_sample_growth_factor=1,relaxation_parameter=1.1)
    if method != 'legacy':
        line_kwargs.update(line_search_armijo_coefficient=0,line_search_uncertainty_sigma=0)
    result.update(line_search_method=method,line_search_config=line_type(**line_kwargs))
    return result


def run(request, workflow):
    if workflow=='vi':
        return workflows.run_vi(model=LinearModel(),**request)
    return workflows.run_mf_vi(model=LinearModel(),rom_model_builder=VersionedBuilder(),**request)


def force_two_rejections(monkeypatch, workflow, family, method):
    if workflow=='vi' and family=='diagonal':
        module,name = vi,'_build_vi_state_from_results'
    elif workflow=='vi':
        module,name = fc_vi,'_evaluate_single_fidelity_state'
    elif family=='diagonal':
        module,name = mf,'_evaluate_mf_vi_state'
    else:
        module,name = fc_mf,'_evaluate_mf_state'
    original = getattr(module,name)
    states=[]
    objectives = [10.,0.,-5.,9.,8.5] if method=='legacy' else [10.,0.,-5.,11.,12.]
    def evaluate(*args,**kwargs):
        state=original(*args,**kwargs)
        state['elbo']=objectives[len(states)]
        states.append(state)
        return state
    monkeypatch.setattr(module,name,evaluate)
    return states


@pytest.mark.mpi_skip
@pytest.mark.parametrize('workflow',['vi','mf'])
@pytest.mark.parametrize('family',['diagonal','full_covariance'])
@pytest.mark.parametrize('method',['legacy','stochastic_nonmonotone'])
def test_step_size_resets_after_acceptance_but_retries_backtrack(tmp_path,monkeypatch,workflow,family,method):
    states=force_two_rejections(monkeypatch,workflow,family,method)
    run(request(tmp_path,workflow,family,3,method),workflow)
    assert len(states)==5
    with np.load(tmp_path/'history.npz',allow_pickle=True) as history:
        np.testing.assert_allclose(history['vi_history_accepted_step_size'][-2:],[.001,.004])
    with np.load(tmp_path/'iteration_2/restart.npz',allow_pickle=True) as saved:
        assert float(saved['step_size'])==.004


@pytest.mark.mpi_skip
@pytest.mark.parametrize('workflow',['vi','mf'])
@pytest.mark.parametrize('adaptive',[False,True])
@pytest.mark.parametrize('method',['legacy','stochastic_nonmonotone'])
def test_newton_damping_resets_and_records_actual_accepted_value(tmp_path,monkeypatch,workflow,adaptive,method):
    states=force_two_rejections(monkeypatch,workflow,'diagonal',method)
    module=vi if workflow=='vi' else mf
    lambdas=[]
    def step(*args,**kwargs):
        lambdas.append(kwargs['newton_additive_regularization'])
        return np.ones(1),np.zeros(1)
    monkeypatch.setattr(module,'_compute_newton_step',step)
    run(request(tmp_path,workflow,'diagonal',3,method,'newton',adaptive),workflow)
    assert len(states)==5
    np.testing.assert_allclose(lambdas,[.01,.03,.05,.01] if adaptive else [.01]*4)
    with np.load(tmp_path/'iteration_2/restart.npz',allow_pickle=True) as saved:
        assert float(saved['current_newton_additive_regularization'])==.01
        assert float(saved['step_size'])==.004
        np.testing.assert_allclose(saved['vi_history_newton_additive_regularization'][-2:],
                                   [.05,.01] if adaptive else [.01,.01])
        np.testing.assert_allclose(saved['vi_history_accepted_step_size'][-2:],[.001,.004])


@pytest.mark.mpi_skip
@pytest.mark.parametrize('workflow',['vi','mf'])
@pytest.mark.parametrize('family',['diagonal','full_covariance'])
def test_restart_ignores_saved_adapted_step_size(tmp_path,workflow,family):
    complete=run(request(tmp_path/'full',workflow,family,4),workflow)
    run(request(tmp_path/'partial',workflow,family,2),workflow)
    with np.load(tmp_path/'partial/iteration_1/restart.npz',allow_pickle=True) as saved:
        checkpoint=dict(saved)
    checkpoint['step_size']=np.array(.000001)
    np.savez(tmp_path/'partial/changed.npz',**checkpoint)
    resumed_request=request(tmp_path/'partial',workflow,family,4)
    resumed_request['restart_file']=str(tmp_path/'partial/changed.npz')
    resumed=run(resumed_request,workflow)
    for a,b in zip(complete,resumed):
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('workflow',['vi','mf'])
def test_adaptive_newton_restart_uses_base_damping(tmp_path,workflow):
    complete=run(request(tmp_path/'full',workflow,'diagonal',4,optimizer='newton',adaptive=True),workflow)
    run(request(tmp_path/'partial',workflow,'diagonal',2,optimizer='newton',adaptive=True),workflow)
    with np.load(tmp_path/'partial/iteration_1/restart.npz',allow_pickle=True) as saved:
        checkpoint=dict(saved)
    checkpoint['step_size']=np.array(.000001)
    checkpoint['current_newton_additive_regularization']=np.array(.04)
    np.savez(tmp_path/'partial/changed.npz',**checkpoint)
    resumed_request=request(tmp_path/'partial',workflow,'diagonal',4,optimizer='newton',adaptive=True)
    resumed_request['restart_file']=str(tmp_path/'partial/changed.npz')
    resumed=run(resumed_request,workflow)
    for a,b in zip(complete,resumed):
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('workflow',['vi','mf'])
@pytest.mark.parametrize('method',['legacy','stochastic_nonmonotone'])
def test_full_covariance_natural_newton_resets_step(tmp_path,monkeypatch,workflow,method):
    states=force_two_rejections(monkeypatch,workflow,'full_covariance',method)
    run(request(tmp_path,workflow,'full_covariance',3,method,optimizer='newton'),workflow)
    assert len(states)==5
    with np.load(tmp_path/'history.npz',allow_pickle=True) as history:
        np.testing.assert_allclose(history['vi_history_accepted_step_size'][-2:],[.001,.004])
    with np.load(tmp_path/'iteration_2/restart.npz',allow_pickle=True) as saved:
        assert float(saved['step_size'])==.004
