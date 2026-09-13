"""Monte Carlo uncertainty-quantification workflows."""

from importlib import import_module

from romtools.workflows._work_dir_compat import patch_work_dir_argument

_monte_carlo = import_module("romtools.workflows.uq.monte_carlo")

MonteCarloResult = _monte_carlo.MonteCarloResult
MultifidelityMonteCarloResult = _monte_carlo.MultifidelityMonteCarloResult
run_monte_carlo = patch_work_dir_argument(
    _monte_carlo, "run_monte_carlo", "absolute_uq_directory"
)
run_multifidelity_monte_carlo = patch_work_dir_argument(
    _monte_carlo, "run_multifidelity_monte_carlo", "absolute_uq_directory"
)

__all__ = [
    "MonteCarloResult",
    "MultifidelityMonteCarloResult",
    "run_monte_carlo",
    "run_multifidelity_monte_carlo",
]
