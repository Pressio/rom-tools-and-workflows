"""Gaussian variational inference workflows and optimizer configurations.

Use ``run_vi``, ``run_mf_vi``, and ``mf_vi_with_auto_rom`` here or through
``romtools.workflows.inverse``. Implementation modules are private to this package.
"""

from .api import run_vi, run_mf_vi, mf_vi_with_auto_rom
from .optimization import (
    VIGradientOptimizerConfig, VIAdamOptimizerConfig, VINewtonOptimizerConfig,
    VINoLineSearchConfig, VILegacyLineSearchConfig,
    VIStochasticNonmonotoneLineSearchConfig,
    SteepestDescentSolver, AdamSolver, NewtonSolver,
)
from .sample_reuse import VISampleReuseConfig

__all__ = [
    "run_vi", "run_mf_vi", "mf_vi_with_auto_rom", "VISampleReuseConfig",
    "VIGradientOptimizerConfig", "VIAdamOptimizerConfig", "VINewtonOptimizerConfig",
    "VINoLineSearchConfig", "VILegacyLineSearchConfig",
    "VIStochasticNonmonotoneLineSearchConfig",
    "SteepestDescentSolver", "AdamSolver", "NewtonSolver",
]
