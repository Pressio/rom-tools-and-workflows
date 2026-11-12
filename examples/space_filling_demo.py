"""Train a GP ROM with concurrent maximin FOM batches on a disk.

Run from the checkout with:
    python examples/space_filling_demo.py --work-dir /tmp/space-filling-demo

The output directory must be absent or empty. Model classes are module-level
and the entry point is guarded so multiprocessing spawn can import them.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from romtools.workflows import run_space_filling
from romtools.workflows.inverse.mf_eki_drivers import GaussianProcessQoiModelBuilderWithTrainingData
from romtools.workflows.parameter_spaces import ParameterSpace


class DiskParameterSpace(ParameterSpace):
    """A constrained domain supplied solely through the sampling interface."""

    def get_names(self):
        return ["x", "y"]

    def get_dimensionality(self):
        return 2

    def generate_samples(self, number_of_samples, seed=None):
        rng = np.random.default_rng(seed)
        radius = np.sqrt(rng.uniform(size=number_of_samples))
        angle = rng.uniform(0, 2 * np.pi, size=number_of_samples)
        return np.c_[radius * np.cos(angle), radius * np.sin(angle)]


class DiskQoiModel:
    """A small file-backed FOM illustrating the full execution contract."""

    def populate_run_directory(self, run_directory, parameter_sample):
        np.savetxt(Path(run_directory) / "parameters.txt",
                   [parameter_sample["x"], parameter_sample["y"]])

    def run_model(self, run_directory, parameter_sample):
        x, y = parameter_sample["x"], parameter_sample["y"]
        sensors = np.linspace(0, 1, 20)
        state = 2 + x * sensors + y * sensors**2 + 0.1 * np.sin((x + y) * sensors)
        qoi = np.array([state.mean(), state[-1]])
        np.savez(Path(run_directory) / "solution.npz", state=state, qoi=qoi)
        return 0

    def compute_qoi(self, run_directory, parameter_sample):
        with np.load(Path(run_directory) / "solution.npz") as data:
            return data["qoi"]


def main(work_dir, max_concurrent_fom_runs=3):
    result = run_space_filling(
        fom_model=DiskQoiModel(),
        rom_model_builder=GaussianProcessQoiModelBuilderWithTrainingData(
            parameter_names=["x", "y"],
        ),
        parameter_space=DiskParameterSpace(),
        absolute_work_dir=str(Path(work_dir).resolve()),
        tolerance=0.02,
        initial_sample_size=10,
        batch_size=5,
        candidate_pool_size=500,
        max_concurrent_fom_runs=max_concurrent_fom_runs,
        random_seed=1,
        max_iterations=6,
        max_fom_samples=30,
    )
    for count, error in zip(result.sample_counts, result.validation_errors):
        print(f"Training samples: {count:2d}; maximum held-out relative L2 error: {error:.6g}")
    print(f"Stopped: {result.termination_reason}; converged: {result.converged}")
    print(f"Results: {Path(work_dir).resolve() / 'space_filling_stats.npz'}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=str(Path(__file__).parent / "space_filling_work"))
    parser.add_argument("--max-concurrent-fom-runs", type=int, default=3)
    arguments = parser.parse_args()
    main(arguments.work_dir, arguments.max_concurrent_fom_runs)
