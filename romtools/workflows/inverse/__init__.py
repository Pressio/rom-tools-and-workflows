"""
Inverse workflows estimate unknown model parameters from observed data.

An inverse problem starts from a forward model, experimental or synthetic
observations, and a parameterized uncertainty model. The goal is to identify
parameter values, or a posterior distribution over parameter values, that make
the model predictions consistent with the observations.

romtools currently supports:

- Ensemble Kalman inversion (EKI) for derivative-free parameter calibration.
- Multifidelity EKI with control variates and adaptive reduced-order model
  refresh strategies.
- Variational inference (VI) with gradient, Adam, and Newton optimizers for
  Gaussian variational families, including diagonal and full-covariance VI.
- Multifidelity VI with control variates and adaptive reduced-order model
  updates, including full-covariance gradient/Adam optimization.
"""

from importlib import import_module as _import_module
import inspect as _inspect

#from romtools.workflows.inverse.bfgs_drivers import *
from romtools.workflows.inverse.eki_drivers import *
from romtools.workflows.inverse.mf_eki_drivers import *
from romtools.workflows.inverse.mf_vi_drivers import *
from romtools.workflows.inverse.vi_drivers import *
from romtools.workflows.inverse.vi_optimization_methods import *
from romtools.workflows.inverse.ego_drivers import *
from romtools.workflows.inverse.ego_optimization_methods import *

# Install the bound-handling wrappers after the legacy EKI modules are loaded.
# The wrappers delegate the historical ``clip`` path unchanged and adapt the
# same numerical engines to unconstrained coordinates for ``transform``.
from romtools.workflows.inverse.eki_bound_transforms import (
    run_eki as _bounded_run_eki,
    run_mf_eki as _bounded_run_mf_eki,
    mf_eki_with_auto_rom as _bounded_mf_eki_with_auto_rom,
)

run_eki = _bounded_run_eki
run_mf_eki = _bounded_run_mf_eki
mf_eki_with_auto_rom = _bounded_mf_eki_with_auto_rom

# Also update the defining modules so direct imports such as
# ``from ...eki_drivers import run_eki`` receive the same public behavior after
# the inverse package has initialized.
_eki_drivers_module = _import_module("romtools.workflows.inverse.eki_drivers")
_mf_eki_drivers_module = _import_module("romtools.workflows.inverse.mf_eki_drivers")
_eki_drivers_module.run_eki = run_eki
_mf_eki_drivers_module.run_mf_eki = run_mf_eki
_mf_eki_drivers_module.mf_eki_with_auto_rom = mf_eki_with_auto_rom

# Install the opt-in VI/MFVI sample-reuse layer after the original drivers have
# loaded. With sample_reuse_config=None the wrappers delegate directly to the
# original implementations, preserving the existing behavior and API.
from romtools.workflows.inverse.vi_sample_reuse import VISampleReuseConfig

_sample_reuse_module = _import_module("romtools.workflows.inverse.vi_sample_reuse")
# Add score-function Hessian reuse for the existing Newton optimizer and keep a
# separate archive when the user requests independent curvature samples.
_import_module("romtools.workflows.inverse.vi_sample_reuse_hessian")

# Add the run-directory policy outside the sample-reuse layer so the same
# create_run_directories option applies to ordinary VI/MFVI, sample reuse, and
# auto-ROM MFVI without changing the underlying driver signatures.
from romtools.workflows.inverse.vi_run_directory_policy import (
    run_vi as _directory_policy_run_vi,
    run_mf_vi as _directory_policy_run_mf_vi,
    mf_vi_with_auto_rom as _directory_policy_mf_vi_with_auto_rom,
)


def _add_vi_wrapper_signature(wrapper, original):
    signature = _inspect.signature(original)
    parameters = list(signature.parameters.values())
    insertion_index = next(
        (
            index for index, parameter in enumerate(parameters)
            if parameter.kind == _inspect.Parameter.VAR_KEYWORD
        ),
        len(parameters),
    )
    directory_parameter = _inspect.Parameter(
        "create_run_directories",
        kind=_inspect.Parameter.KEYWORD_ONLY,
        default=True,
        annotation=bool,
    )
    reuse_parameter = _inspect.Parameter(
        "sample_reuse_config",
        kind=_inspect.Parameter.KEYWORD_ONLY,
        default=None,
        annotation=VISampleReuseConfig,
    )
    parameters.insert(insertion_index, directory_parameter)
    parameters.insert(insertion_index + 1, reuse_parameter)
    wrapper.__signature__ = signature.replace(parameters=parameters)


run_vi = _directory_policy_run_vi
run_mf_vi = _directory_policy_run_mf_vi
mf_vi_with_auto_rom = _directory_policy_mf_vi_with_auto_rom
_add_vi_wrapper_signature(run_vi, _sample_reuse_module._ORIGINAL_RUN_VI)
_add_vi_wrapper_signature(run_mf_vi, _sample_reuse_module._ORIGINAL_RUN_MF_VI)
_add_vi_wrapper_signature(
    mf_vi_with_auto_rom,
    _sample_reuse_module._ORIGINAL_MF_VI_WITH_AUTO_ROM,
)

_vi_drivers_module = _import_module("romtools.workflows.inverse.vi_drivers")
_mf_vi_drivers_module = _import_module("romtools.workflows.inverse.mf_vi_drivers")
_vi_drivers_module.run_vi = run_vi
_mf_vi_drivers_module.run_mf_vi = run_mf_vi
_mf_vi_drivers_module.mf_vi_with_auto_rom = mf_vi_with_auto_rom

# Add the true full-covariance Gaussian family as an outer routing layer. The
# required ``initial_variational_parameter_space`` now defines both the
# variational family and its initial moments. GaussianParameterSpace routes to
# diagonal VI; MultivariateGaussianParameterSpace routes to true freely
# evolving full-covariance VI. The prior family is independent.
from romtools.workflows.inverse.full_covariance_router import (
    run_vi as _full_covariance_run_vi,
    run_mf_vi as _full_covariance_run_mf_vi,
    mf_vi_with_auto_rom as _full_covariance_auto_mf_vi,
)


def _add_full_covariance_signature(wrapper, legacy_wrapper):
    signature = _inspect.signature(legacy_wrapper)
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.name != "initial_variational_parameter_space"
    ]
    insertion_index = next(
        (
            index
            for index, parameter in enumerate(parameters)
            if parameter.kind
            in (_inspect.Parameter.KEYWORD_ONLY, _inspect.Parameter.VAR_KEYWORD)
        ),
        len(parameters),
    )
    parameters.insert(
        insertion_index,
        _inspect.Parameter(
            "initial_variational_parameter_space",
            kind=_inspect.Parameter.KEYWORD_ONLY,
            default=_inspect.Parameter.empty,
        ),
    )
    parameters.insert(
        insertion_index + 1,
        _inspect.Parameter(
            "max_covariance_log_step",
            kind=_inspect.Parameter.KEYWORD_ONLY,
            default=1.0,
            annotation=float,
        ),
    )
    wrapper.__signature__ = signature.replace(parameters=parameters)


_add_full_covariance_signature(_full_covariance_run_vi, _directory_policy_run_vi)
_add_full_covariance_signature(_full_covariance_run_mf_vi, _directory_policy_run_mf_vi)
_add_full_covariance_signature(
    _full_covariance_auto_mf_vi, _directory_policy_mf_vi_with_auto_rom
)

run_vi = _full_covariance_run_vi
run_mf_vi = _full_covariance_run_mf_vi
mf_vi_with_auto_rom = _full_covariance_auto_mf_vi

# Keep direct imports from the historical defining modules aligned with the
# public package-level API.
_vi_drivers_module.run_vi = run_vi
_mf_vi_drivers_module.run_mf_vi = run_mf_vi
_mf_vi_drivers_module.mf_vi_with_auto_rom = mf_vi_with_auto_rom

# Expose submodules as attributes so `import romtools.workflows.inverse.mf_eki_drivers as m`
# binds to the module instead of the function imported above.
#run_eki = _import_module("romtools.workflows.inverse.eki_drivers")
#run_mf_eki = _import_module("romtools.workflows.inverse.mf_eki_drivers")
