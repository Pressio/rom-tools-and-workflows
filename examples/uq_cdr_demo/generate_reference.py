"""Regenerate the high-fidelity reference mean for the CDR UQ example."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from example import EXAMPLE_DIRECTORY, MODEL_DIRECTORY, build_parameter_space

if str(MODEL_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODEL_DIRECTORY))
import steady_cdr as cdr  # noqa: E402


def integrated_flux(system, parameter_sample):
    bmag, theta, nu, sigma = parameter_sample
    b = np.array([bmag * np.cos(theta), bmag * np.sin(theta)])
    state = cdr.solveFom(system, b, nu, sigma)
    state_grid = np.zeros((system.Nx + 2, system.Ny + 2))
    state_grid[1:-1, 1:-1] = state.reshape(system.Nx, system.Ny)
    boundary_flux = -state_grid[:, -2] / system.dx
    return float(np.sum(boundary_flux) * system.dy)


def main(number_of_samples: int, random_seed: int, output: Path) -> None:
    samples = build_parameter_space().generate_samples(
        number_of_samples, seed=random_seed
    )
    system = cdr.AdvectionDiffusionSystem(21, 21)
    values = np.fromiter(
        (integrated_flux(system, sample) for sample in samples),
        dtype=float,
        count=number_of_samples,
    )
    variance = float(np.var(values, ddof=1))
    result = {
        "method": "high-fidelity Monte Carlo",
        "grid": [21, 21],
        "number_of_samples": number_of_samples,
        "random_seed": random_seed,
        "mean": float(np.mean(values)),
        "variance": variance,
        "standard_error": float(np.sqrt(variance / number_of_samples)),
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--number-of-samples", type=int, default=20000)
    parser.add_argument("--random-seed", type=int, default=314159)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXAMPLE_DIRECTORY / "reference_stats.json",
    )
    arguments = parser.parse_args()
    main(arguments.number_of_samples, arguments.random_seed, arguments.output)
