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
  Gaussian variational families.
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

# Install the stateful Adam update on the existing VI/MF-VI gradient paths.
# This keeps sampling, variance reduction, bounds handling, logging, and model
# evaluation in the original drivers while exposing optimizer_method='adam'.
_vi_drivers_module = _import_module("romtools.workflows.inverse.vi_drivers")
_mf_vi_drivers_module = _import_module("romtools.workflows.inverse.mf_vi_drivers")
from romtools.workflows.inverse.vi_adam_integration import (
    install_adam_optimizer_support as _install_adam_optimizer_support,
)
_install_adam_optimizer_support(_vi_drivers_module, _mf_vi_drivers_module)
run_vi = _vi_drivers_module.run_vi
run_mf_vi = _mf_vi_drivers_module.run_mf_vi

# Expose submodules as attributes so `import romtools.workflows.inverse.mf_eki_drivers as m`
# binds to the module instead of the function imported above.
#run_eki = _import_module("romtools.workflows.inverse.eki_drivers")
#run_mf_eki = _import_module("romtools.workflows.inverse.mf_eki_drivers")
