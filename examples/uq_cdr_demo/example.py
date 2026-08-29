"""Monte Carlo UQ for the steady convection-diffusion-reaction example."""

import argparse
import os
from pathlib import Path
import sys
from typing import Optional

import numpy as np

from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.sampling_methods import MonteCarloSampler
from romtools.workflows.uq import run_monte_carlo, run_multifidelity_monte_carlo


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
MODEL_DIRECTORY = EXAMPLE_DIRECTORY.parent / "models"
if str(MODEL_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODEL_DIRECTORY))

from steady_cdr_model import SteadyCdrQoiModel  # noqa: E402


class IntegratedFluxCdrModel(SteadyCdrQoiModel):
    """Return the integrated right-boundary flux as a scalar QoI."""

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        with np.load(os.path.join(run_directory, "solution.npz")) as solution:
            boundary_flux = solution["qoi"]
        return np.array([np.sum(boundary_flux) * self.system.dy])


def build_parameter_space() -> UniformParameterSpace:
    """Construct independent uniform distributions for the CDR parameters."""
    return UniformParameterSpace(
        parameter_names=["bmag", "theta", "nu", "sigma"],
        lower_bounds=[0.5, np.pi / 6.0, 0.02, 0.1],
        upper_bounds=[1.5, np.pi / 3.0, 0.08, 0.6],
        sampler=MonteCarloSampler,
    )


def main(output_directory: Optional[Path] = None) -> None:
    """Run single- and multifidelity CDR mean-estimation examples."""
    output_directory = (
        EXAMPLE_DIRECTORY / "uq_cdr_output"
        if output_directory is None
        else Path(output_directory).resolve()
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    parameter_space = build_parameter_space()

    mc_result = run_monte_carlo(
        model=IntegratedFluxCdrModel(nx=21, ny=21),
        parameter_space=parameter_space,
        absolute_uq_directory=str(output_directory / "monte_carlo"),
        number_of_samples=12,
        random_seed=7,
    )

    mfmc_result = run_multifidelity_monte_carlo(
        high_fidelity_model=IntegratedFluxCdrModel(nx=21, ny=21),
        low_fidelity_model=IntegratedFluxCdrModel(nx=9, ny=9),
        parameter_space=parameter_space,
        absolute_uq_directory=str(output_directory / "multifidelity"),
        pilot_sample_count=4,
        high_fidelity_equivalent_budget=12.0,
        low_to_high_fidelity_cost_ratio=0.05,
        allocation_qoi_index=0,
        random_seed=7,
    )

    print("CDR integrated right-boundary flux")
    print(
        f"MC:   mean={mc_result.mean[0]:.6e}, "
        f"standard error={mc_result.standard_error[0]:.3e}"
    )
    print(
        f"MFMC: mean={mfmc_result.mean[0]:.6e}, "
        f"standard error={mfmc_result.standard_error[0]:.3e}, "
        f"N_H={mfmc_result.high_fidelity_sample_count}, "
        f"N_L={mfmc_result.low_fidelity_sample_count}"
    )
    print(f"Results written to {output_directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Output path (default: examples/uq_cdr_demo/uq_cdr_output)",
    )
    arguments = parser.parse_args()
    main(arguments.output_directory)
