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

# Expose submodules as attributes so `import romtools.workflows.inverse.mf_eki_drivers as m`
# binds to the module instead of the function imported above.
#run_eki = _import_module("romtools.workflows.inverse.eki_drivers")
#run_mf_eki = _import_module("romtools.workflows.inverse.mf_eki_drivers")
