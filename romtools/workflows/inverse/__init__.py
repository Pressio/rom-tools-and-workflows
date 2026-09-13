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
- Variational inference (VI) with gradient and Newton optimizers for Gaussian
  variational families.
- Multifidelity VI with control variates and adaptive reduced-order model
  updates.
"""

from importlib import import_module as _import_module

#from romtools.workflows.inverse.bfgs_drivers import *
from romtools.workflows.inverse.eki_drivers import *
from romtools.workflows.inverse.mf_eki_drivers import *
from romtools.workflows.inverse.mf_vi_drivers import *
from romtools.workflows.inverse.vi_drivers import *
from romtools.workflows.inverse.vi_optimization_methods import *
from romtools.workflows.inverse.ego_drivers import *
from romtools.workflows.inverse.ego_optimization_methods import *
from romtools.workflows._work_dir_compat import patch_work_dir_argument as _patch_work_dir_argument

_eki_drivers = _import_module("romtools.workflows.inverse.eki_drivers")
_mf_eki_drivers = _import_module("romtools.workflows.inverse.mf_eki_drivers")
_vi_drivers = _import_module("romtools.workflows.inverse.vi_drivers")
_mf_vi_drivers = _import_module("romtools.workflows.inverse.mf_vi_drivers")
_ego_drivers = _import_module("romtools.workflows.inverse.ego_drivers")

run_eki = _patch_work_dir_argument(
    _eki_drivers, "run_eki", "absolute_eki_directory"
)
run_mf_eki = _patch_work_dir_argument(
    _mf_eki_drivers, "run_mf_eki", "absolute_eki_directory"
)
mf_eki_with_auto_rom = _patch_work_dir_argument(
    _mf_eki_drivers, "mf_eki_with_auto_rom", "absolute_eki_directory"
)
run_vi = _patch_work_dir_argument(
    _vi_drivers, "run_vi", "absolute_vi_directory"
)
run_mf_vi = _patch_work_dir_argument(
    _mf_vi_drivers, "run_mf_vi", "absolute_vi_directory"
)
mf_vi_with_auto_rom = _patch_work_dir_argument(
    _mf_vi_drivers, "mf_vi_with_auto_rom", "absolute_vi_directory"
)
run_ego = _patch_work_dir_argument(
    _ego_drivers, "run_ego", "absolute_ego_directory"
)
