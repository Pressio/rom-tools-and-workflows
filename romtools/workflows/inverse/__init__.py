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

from . import (
    ego_drivers,
    ego_optimization_methods,
    eki_drivers,
    mf_eki_drivers,
    mf_vi_drivers,
    vi_drivers,
    vi_optimization_methods,
)
from .ego_drivers import run_batch_ego, run_ego
from .eki_drivers import run_eki
from .mf_eki_drivers import mf_eki_with_auto_rom, run_mf_eki
from .mf_vi_drivers import mf_vi_with_auto_rom, run_mf_vi
from .vi_drivers import run_vi

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
