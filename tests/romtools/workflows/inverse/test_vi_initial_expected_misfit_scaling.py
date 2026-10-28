"""Initial-misfit scaling uses only initial HF data and is fixed on restart."""
from pathlib import Path
import numpy as np
import pytest
import romtools.workflows as workflows
from romtools.workflows.inverse.vi import drivers as vi
from romtools.workflows.parameter_spaces import GaussianParameterSpace, MultivariateGaussianParameterSpace, MonteCarloSampler

class Model:
    def __init__(self, factor=1.0):
        self.calls = 0
        self.factor = factor
    def populate_run_directory(self, run_directory, parameter_sample):
        pass
    def run_model(self, run_directory, parameter_sample):
        self.calls += 1
        return 0
    def compute_qoi(self, run_directory, parameter_sample):
        return self.factor * np.array([2*parameter_sample['x'], parameter_sample['y']])

class Builder:
    def build_from_training_dirs(self, *args):
        return Model(factor=0.8)

def space(full):
    common=dict(parameter_names=['x','y'],means=np.array([.4,-.2]),sampler=MonteCarloSampler)
    return (MultivariateGaussianParameterSpace(covariance=np.array([[.5,.1],[.1,.3]]),**common)
            if full else GaussianParameterSpace(stds=np.array([.7,.5]),**common))

def run(path, *, mf, full, scale, entropy='analytic', iterations=2, restart=None):
    model=Model();s=space(full)
    kw=dict(model=model,prior_parameter_space=s,initial_variational_parameter_space=s,
        observations=np.zeros(2),observations_covariance=np.eye(2)*.01,
        absolute_work_dir=str(path),random_seed=12,elbo_scaling_factor=scale,
        parameter_mins=np.full(2,-3.0),parameter_maxes=np.full(2,3.0),
        score_function_entropy_strategy=entropy,restart_file=restart,
        optimizer_method='newton',optimizer_config=workflows.VINewtonOptimizerConfig(
            max_iterations=iterations,gradient_norm_tolerance=0.0),
        line_search_method='legacy',line_search_config=workflows.VILegacyLineSearchConfig(
            initial_step_size=.001,max_step_size=.001,step_size_growth_factor=1.0),
        restart_files_to_keep=10)
    if mf:
        kw.update(rom_model_builder=Builder(),fom_sample_size=8,rom_extra_sample_size=8,
                  fom_evaluation_concurrency=1,rom_evaluation_concurrency=1)
    else:
        kw.update(sample_size=8,evaluation_concurrency=1)
    (workflows.run_mf_vi if mf else workflows.run_vi)(**kw)
    return model.calls

def checkpoints(path):
    return sorted(Path(path).glob('iteration_*/restart.npz'),key=lambda p:int(p.parent.name.split('_')[-1]))

@pytest.mark.parametrize('mf',[False,True])
@pytest.mark.parametrize('full',[False,True])
@pytest.mark.parametrize('entropy',['analytic','joint'])
def test_matches_explicit_scale_and_adds_no_hf_calls(tmp_path,mf,full,entropy):
    a=tmp_path/'auto';b=tmp_path/'explicit'
    calls=run(a,mf=mf,full=full,scale='initial_expected_misfit',entropy=entropy)
    files=checkpoints(a);assert len(files)>=2
    with np.load(files[0],allow_pickle=True) as initial:
        state=initial['mf_vi_state'].item() if full and mf else initial
        likelihood=state['log_likelihoods_fom'] if 'log_likelihoods_fom' in state else state['log_likelihoods']
        factor=1/max(1,float(-likelihood.mean()))
        assert float(initial['elbo_scaling_factor'])==pytest.approx(factor)
    assert run(b,mf=mf,full=full,scale=factor,entropy=entropy)==calls
    for pa,pb in zip(files,checkpoints(b)):
        with np.load(pa,allow_pickle=True) as da,np.load(pb,allow_pickle=True) as db:
            assert float(da['elbo_scaling_factor'])==pytest.approx(factor)
            for key in ['elbo','entropy','gradient_mean','gradient_log_std','hessian_diagonal_mean','hessian_diagonal_log_std','hessian_full','variational_mean','variational_log_std','variational_cholesky_optimizer']:
                if key in da and da[key].dtype!=object:
                    np.testing.assert_allclose(da[key],db[key],rtol=1e-10,atol=1e-12,err_msg=key)
            if full and mf:
                sa,sb=da['mf_vi_state'].item(),db['mf_vi_state'].item()
                for key in ['elbo','entropy','gradient_mean','gradient_covariance_svec','newton_hessian_full']:
                    np.testing.assert_allclose(sa[key],sb[key],rtol=1e-10,atol=1e-12,err_msg=key)

@pytest.mark.parametrize('mf',[False,True])
@pytest.mark.parametrize('full',[False,True])
def test_restart_preserves_initial_scale(tmp_path,mf,full):
    path=tmp_path/'resume'
    run(path,mf=mf,full=full,scale='initial_expected_misfit',iterations=1)
    initial=checkpoints(path)[0]
    with np.load(initial) as data:factor=float(data['elbo_scaling_factor'])
    run(path,mf=mf,full=full,scale='initial_expected_misfit',iterations=2,restart=str(initial))
    files=checkpoints(path);assert len(files)>=2
    for f in files:
        with np.load(f) as d:assert float(d['elbo_scaling_factor'])==factor
    direct=tmp_path/'direct'
    run(direct,mf=mf,full=full,scale='initial_expected_misfit',iterations=2)
    with np.load(files[-1],allow_pickle=True) as da,np.load(checkpoints(direct)[-1],allow_pickle=True) as db:
        np.testing.assert_allclose(da['variational_mean'],db['variational_mean'],rtol=1e-10,atol=1e-12)
        key='variational_cholesky_optimizer' if full else 'variational_log_std'
        np.testing.assert_allclose(da[key],db[key],rtol=1e-10,atol=1e-12)

@pytest.mark.parametrize('misfits,expected',[([0,0],1),([.2,.4],1),([2,6],.25),([1e308,1e308],1e-308)])
def test_scale_guard_and_overflow_safe_mean(misfits,expected):
    assert vi._initial_expected_misfit_scale(misfits)==pytest.approx(expected,abs=0)

@pytest.mark.parametrize('misfits',[[],[np.nan],[np.inf],[-1],[[1,2]]])
def test_reject_invalid_misfits(misfits):
    with pytest.raises(ValueError):vi._initial_expected_misfit_scale(misfits)

def test_restart_requires_a_valid_saved_scale():
    for saved in [{},{'elbo_scaling_factor':np.nan},{'elbo_scaling_factor':0}]:
        with pytest.raises(ValueError):vi._restore_initial_expected_misfit_scale(None,saved)
