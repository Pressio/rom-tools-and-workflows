"""Trusted reference adapter used to validate the benchmark, not supplied to agents."""
from pathlib import Path
import numpy as np
from benchmarks.agent_skills.tasks.cdr_model_adapter.legacy_cdr import LegacyCDR
from romtools.workflows.sampling.sampling import run_sampling
from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.sampling_methods import MonteCarloSampler


class CDRModel:
    def __init__(self, nx=8, ny=8):
        self.legacy = LegacyCDR(nx, ny)

    def populate_run_directory(self, run_directory, parameter_sample):
        Path(run_directory).mkdir(parents=True, exist_ok=True)

    def run_model(self, run_directory, parameter_sample):
        state = self.legacy.solve(parameter_sample["nu"], parameter_sample["sigma"])
        np.save(Path(run_directory) / "solution.npy", state)
        return 0

    def compute_qoi(self, run_directory, parameter_sample):
        state = np.load(Path(run_directory) / "solution.npy")
        return np.array([self.legacy.boundary_flux(state)])


def run_experiment(work_dir, number_of_samples=4):
    space = UniformParameterSpace(["nu", "sigma"], [.01, .1], [.05, .5],
                                  sampler=MonteCarloSampler)
    return run_sampling(CDRModel(), space, absolute_work_dir=str(work_dir),
                        number_of_samples=number_of_samples, random_seed=7,
                        evaluation_concurrency=1)
