from pathlib import Path

import numpy as np

from romtools.workflows import sampling
from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.sampling_methods import MonteCarloSampler


class ToyModel:
    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        run_dir = Path(run_directory)
        run_dir.joinpath("params.txt").write_text(
            f"alpha={parameter_sample['alpha']}\n"
            f"beta={parameter_sample['beta']}\n",
            encoding="utf-8",
        )

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        run_dir = Path(run_directory)
        alpha = float(parameter_sample["alpha"])
        beta = float(parameter_sample["beta"])
        result = alpha + 2.0 * beta
        np.savez(run_dir / "solution", value=result)
        return 0


def main() -> None:
    parameter_space = UniformParameterSpace(
        parameter_names=["alpha", "beta"],
        lower_bounds=[0.0, 0.0],
        upper_bounds=[1.0, 1.0],
        sampler=MonteCarloSampler,
    )
    output_dir = Path(__file__).resolve().parent / "sampling_output"
    sample_dirs = sampling.run_sampling(
        model=ToyModel(),
        parameter_space=parameter_space,
        absolute_sampling_directory=str(output_dir),
        number_of_samples=3,
        random_seed=1,
    )
    print(f"Generated {len(sample_dirs)} samples in {output_dir}")


if __name__ == "__main__":
    main()
