"""Train a GP surrogate with greedy uncertainty and maximin batch enrichment.

Run: python examples/hybrid_greedy_demo.py --work-dir /tmp/hybrid-greedy-demo
Use a fresh output directory. The main guard supports multiprocessing spawn.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from romtools.rom.qoi_surrogates import GaussianProcessKernel
from romtools.workflows import run_hybrid_greedy
from romtools.workflows.inverse.mf_eki_drivers import GaussianProcessQoiModelBuilderWithTrainingData
from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.sampling_methods import LatinHypercubeSampler


class ScalarFom:
    """File-backed stand-in for an expensive scalar-output simulation."""

    def populate_run_directory(self, directory, parameters):
        np.savetxt(Path(directory) / "parameter.txt", [parameters["x"]])

    def run_model(self, directory, parameters):
        x = parameters["x"]
        value = 2 + np.sin(6 * x) + .3 * np.cos(15 * x)
        np.save(Path(directory) / "qoi.npy", [value])
        return 0

    def compute_qoi(self, directory, parameters):
        return np.load(Path(directory) / "qoi.npy")


class UncertaintyRom:
    """Adapt the existing scalar GP's predictive standard deviation to an indicator.

    Uncertainty ranks candidates; it is not the measured validation error.
    """

    def __init__(self, model):
        self.model = model

    def populate_run_directory(self, directory, parameters):
        return self.model.populate_run_directory(directory, parameters)

    def run_model(self, directory, parameters):
        return self.model.run_model(directory, parameters)

    def compute_qoi(self, directory, parameters):
        return self.model.compute_qoi(directory, parameters)

    def compute_error_estimate(self, directory, parameters):
        _, std = self.model.compute_qoi_and_var(directory, parameters)
        return float(np.linalg.norm(std))


class UncertaintyBuilder(GaussianProcessQoiModelBuilderWithTrainingData):
    """Reuse the existing builder, including independent fitting in each fold."""

    def __init__(self):
        super().__init__(parameter_names=["x"],
                         kernel=GaussianProcessKernel(length_scale=.15),
                         noise_variance=1e-10)

    def build_from_training_dirs(self, offline, directories, parameters, qois):
        return UncertaintyRom(super().build_from_training_dirs(offline, directories, parameters, qois))


def main(work_dir, max_concurrent_fom_runs=3):
    result = run_hybrid_greedy(
        fom_model=ScalarFom(),
        rom_model_builder=UncertaintyBuilder(),
        parameter_space=UniformParameterSpace(["x"], [0], [1], LatinHypercubeSampler),
        absolute_work_dir=str(Path(work_dir).resolve()),
        tolerance=.02,
        initial_sample_size=5,
        batch_size=3,
        candidate_pool_size=256,
        max_concurrent_fom_runs=max_concurrent_fom_runs,
        random_seed=1,
        max_iterations=6,
        max_fom_samples=20,
        create_rom_run_directories=False,
    )
    for count, error in zip(result.sample_counts, result.validation_errors):
        print(f"Training samples: {count:2d}; maximum held-out relative L2 error: {error:.6g}")
    print(f"Greedy points: {result.greedy_parameters[:, 0]}")
    print(f"Stopped: {result.termination_reason}; converged: {result.converged}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=str(Path(__file__).parent / "hybrid_greedy_work"))
    parser.add_argument("--max-concurrent-fom-runs", type=int, default=3)
    arguments = parser.parse_args()
    main(arguments.work_dir, arguments.max_concurrent_fom_runs)
