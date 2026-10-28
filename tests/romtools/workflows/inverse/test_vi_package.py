"""Public imports and per-run state survive VI package consolidation."""

import subprocess
import sys

import numpy as np
import pytest

from romtools.workflows.inverse import vi
from romtools.workflows.inverse.vi import drivers, evaluation, mf_drivers
from romtools.workflows.inverse.vi.newton import FullCovarianceNewton
from tests.romtools.workflows.inverse.test_vi_initial_expected_misfit_scaling import Model, space


@pytest.mark.parametrize('first', [
    'romtools.workflows.inverse.vi',
    'romtools.workflows.inverse.vi.sample_reuse',
    'romtools.workflows.inverse.vi.newton',
])
def test_public_exports_in_fresh_process(first):
    subprocess.run([sys.executable, '-c', f'''
import importlib, inspect
importlib.import_module({first!r})
from romtools.workflows import inverse
from romtools.workflows.inverse import vi
for name in vi.__all__:
    assert getattr(inverse, name) is getattr(vi, name), name
for name in ('run_vi', 'run_mf_vi', 'mf_vi_with_auto_rom'):
    signature = inspect.signature(getattr(vi, name))
    assert not any(p.startswith('_') for p in signature.parameters)
    assert signature.parameters['initial_variational_parameter_space'].default is inspect.Parameter.empty
'''], check=True)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('full', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_evaluators_remain_unmodified_during_runs_and_after_failure(tmp_path, full, fail):
    original_vi = drivers._evaluate_vi_state
    original_mf = mf_drivers._evaluate_mf_vi_state
    original_iteration = drivers.run_vi_iteration

    class InspectingModel(Model):
        def run_model(self, *args):
            assert drivers._evaluate_vi_state is original_vi
            assert mf_drivers._evaluate_mf_vi_state is original_mf
            assert drivers.run_vi_iteration is original_iteration
            assert evaluation._ACTIVE_CREATE_RUN_DIRECTORIES.get() is False
            if fail:
                raise RuntimeError('model failed')
            return super().run_model(*args)

    parameter_space = space(full)
    kwargs = dict(
        model=InspectingModel(), prior_parameter_space=parameter_space,
        initial_variational_parameter_space=parameter_space,
        observations=np.zeros(2), observations_covariance=np.eye(2),
        optimizer_method='newton', optimizer_config=vi.VINewtonOptimizerConfig(max_iterations=1),
        sample_size=6, absolute_work_dir=str(tmp_path), bounded_parameter_handling='clip',
        create_run_directories=False,
    )
    if not full:
        kwargs['sample_reuse_config'] = vi.VISampleReuseConfig(
            use_score_diagnostic=False, use_hessian_score_diagnostic=False,
            hessian_relative_standard_error_threshold=None,
        )
    if fail:
        with pytest.raises(RuntimeError, match='model failed'):
            vi.run_vi(**kwargs)
    else:
        vi.run_vi(**kwargs)
    assert evaluation._ACTIVE_CREATE_RUN_DIRECTORIES.get() is None
    assert drivers._evaluate_vi_state is original_vi
    assert mf_drivers._evaluate_mf_vi_state is original_mf


def test_newton_runs_own_curvature_and_restore_saved_accepted_hessian():
    config = vi.VINewtonOptimizerConfig(newton_curvature_strategy=' LAGGED ')
    first = FullCovarianceNewton(config)
    second = FullCovarianceNewton(config)
    saved = {
        'newton_hessian_full': 4.0 * np.eye(2),
        'natural_gradient_mean': np.zeros(1),
        'natural_gradient_covariance_svec': np.zeros(1),
    }
    first.gradient_norm(saved)
    np.testing.assert_allclose(first.average_hessian(np.eye(2)), 1.75 * np.eye(2))
    np.testing.assert_allclose(second.average_hessian(np.eye(2)), np.eye(2))
    assert config.newton_curvature_strategy == ' LAGGED '


@pytest.mark.mpi_skip
def test_nested_public_call_inherits_directory_policy(tmp_path):
    parameter_space = space(False)

    class InspectingModel(Model):
        def run_model(self, *args):
            assert evaluation._ACTIVE_CREATE_RUN_DIRECTORIES.get() is False
            return super().run_model(*args)

    with evaluation._run_directory_policy(False):
        vi.run_vi(
            model=InspectingModel(), prior_parameter_space=parameter_space,
            initial_variational_parameter_space=parameter_space,
            observations=np.zeros(2), observations_covariance=np.eye(2),
            optimizer_config=vi.VIGradientOptimizerConfig(max_iterations=1),
            sample_size=6, absolute_work_dir=str(tmp_path), bounded_parameter_handling='clip',
        )
        assert evaluation._ACTIVE_CREATE_RUN_DIRECTORIES.get() is False
    assert evaluation._ACTIVE_CREATE_RUN_DIRECTORIES.get() is None
    assert not list(tmp_path.glob('iteration_*/run_[0-9]*'))
