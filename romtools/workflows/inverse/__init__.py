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

The names in ``__all__`` are the stable high-level workflow entry points.
Optimization helpers and implementation utilities remain available from their
defining modules but are not part of the public compatibility contract.
"""

from romtools.workflows.inverse.eki_drivers import *
from romtools.workflows.inverse.mf_eki_drivers import *
from romtools.workflows.inverse.mf_vi_drivers import *
from romtools.workflows.inverse.vi_drivers import *
from romtools.workflows.inverse.vi_optimization_methods import *
from romtools.workflows.inverse.ego_drivers import *
from romtools.workflows.inverse.ego_optimization_methods import *

__all__ = [
    "run_eki",
    "run_mf_eki",
    "mf_eki_with_auto_rom",
    "run_vi",
    "run_mf_vi",
    "mf_vi_with_auto_rom",
    "run_ego",
    "run_batch_ego",
]
