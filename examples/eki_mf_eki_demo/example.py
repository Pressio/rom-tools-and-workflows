import os
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt

from romtools.workflows.parameters import UniformParameter
from romtools.workflows.parameter_spaces import HeterogeneousParameterSpace
from romtools.workflows.models import QoiModel
from romtools.workflows.model_builders import QoiModelBuilder
from romtools.workflows.inverse.eki_drivers import run_eki
from romtools.workflows.inverse.mf_eki_drivers import run_mf_eki

EXAMPLE_DIR = os.path.abspath(os.path.dirname(__file__))
CDR_PATH = os.path.join(EXAMPLE_DIR, "convection_diffusion_reaction_system_code")
if CDR_PATH not in sys.path:
    sys.path.insert(0, CDR_PATH)

import cdr  # noqa: E402
import cdr_rom  # noqa: E402


class CdrFomQoiModel:
    def __init__(self, system: cdr.AdvectionDiffusionSystem, b_vec: np.ndarray):
        self._system = system
        self._b_vec = b_vec

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        with open(os.path.join(run_directory, "params.txt"), "w", encoding="utf-8") as handle:
            for key, value in parameter_sample.items():
                handle.write(f"{key}: {value}\n")

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        nu = float(parameter_sample["nu"])
        sigma = float(parameter_sample["sigma"])
        u = cdr.solveFom(self._system, self._b_vec, nu, sigma)
        np.savez(os.path.join(run_directory, "solution.npz"), u=u)
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        data = np.load(os.path.join(run_directory, "solution.npz"))
        u = data["u"]
        qoi = np.dot(self._system.C, u)
        return np.array([qoi])


class CdrRomQoiModel:
    def __init__(self, system: cdr.AdvectionDiffusionSystem, basis: np.ndarray, b_vec: np.ndarray):
        self._system = system
        self._basis = basis
        self._b_vec = b_vec
        self._rom = cdr_rom.primalGalerkinROM(system, basis)

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        with open(os.path.join(run_directory, "params.txt"), "w", encoding="utf-8") as handle:
            for key, value in parameter_sample.items():
                handle.write(f"{key}: {value}\n")

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        nu = float(parameter_sample["nu"])
        sigma = float(parameter_sample["sigma"])
        u_hat = cdr_rom.solveRom(self._rom, self._b_vec, nu, sigma)
        u = self._basis @ u_hat
        np.savez(os.path.join(run_directory, "solution.npz"), u=u, u_hat=u_hat)
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        data = np.load(os.path.join(run_directory, "solution.npz"))
        u = data["u"]
        qoi = np.dot(self._system.C, u)
        return np.array([qoi])


class CdrRomBuilder:
    def __init__(self, system: cdr.AdvectionDiffusionSystem, b_vec: np.ndarray, rom_dim: int = 12):
        self._system = system
        self._b_vec = b_vec
        self._rom_dim = rom_dim

    def build_from_training_dirs(self, offline_data_dir: str, training_data_dirs):
        snapshots = []
        for run_dir in training_data_dirs:
            solution_path = os.path.join(run_dir, "solution.npz")
            if os.path.exists(solution_path):
                data = np.load(solution_path)
                snapshots.append(data["u"])
        if not snapshots:
            raise RuntimeError("No training snapshots found for ROM construction.")
        snapshot_matrix = np.column_stack(snapshots)
        u, _, _ = np.linalg.svd(snapshot_matrix, full_matrices=False)
        basis = u[:, : min(self._rom_dim, u.shape[1])]
        return CdrRomQoiModel(self._system, basis, self._b_vec)


def _collect_error_history(work_dir: str, mf: bool) -> list:
    history = []
    iteration = 0
    while True:
        restart_path = os.path.join(work_dir, f"iteration_{iteration}", "restart.npz")
        if not os.path.exists(restart_path):
            break
        data = np.load(restart_path, allow_pickle=True)
        if mf:
            sample_one_fom_results = data["sample_one_fom_results"].item()
            errors = sample_one_fom_results["errors"]
        else:
            errors = data["errors"]
        history.append(float(np.mean(np.linalg.norm(errors, axis=0))))
        iteration += 1
    return history


def main():
    np.random.seed(1)

    system = cdr.AdvectionDiffusionSystem(Nx=25, Ny=25)
    b_vec = np.array([1.0, 1.0])

    nu_true = 0.04
    sigma_true = 0.3
    u_true = cdr.solveFom(system, b_vec, nu_true, sigma_true)
    observations = np.array([np.dot(system.C, u_true)])
    observations_covariance = np.eye(1) * 1e-5

    parameter_space = HeterogeneousParameterSpace(
        [
            UniformParameter("nu", 0.01, 0.08),
            UniformParameter("sigma", 0.1, 0.6),
        ]
    )

    base_dir = os.path.join(EXAMPLE_DIR, "eki_mf_eki_work")
    eki_dir = os.path.join(base_dir, "eki")
    mf_dir = os.path.join(base_dir, "mf_eki")
    shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(base_dir, exist_ok=True)

    fom_model = CdrFomQoiModel(system, b_vec)
    rom_builder = CdrRomBuilder(system, b_vec, rom_dim=12)

    run_eki(
        model=fom_model,
        parameter_space=parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        absolute_eki_directory=eki_dir,
        ensemble_size=18,
        max_iterations=8,
        evaluation_concurrency=1,
    )

    run_mf_eki(
        model=fom_model,
        rom_model_builder=rom_builder,
        parameter_space=parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        absolute_eki_directory=mf_dir,
        fom_ensemble_size=8,
        rom_extra_ensemble_size=12,
        rom_tolerance=0.1,
        max_iterations=8,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
    )

    eki_history = _collect_error_history(eki_dir, mf=False)
    mf_history = _collect_error_history(mf_dir, mf=True)

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(eki_history, marker="o", label="EKI (FOM)")
    plt.plot(mf_history, marker="s", label="MF-EKI (FOM+ROM)")
    plt.xlabel("Iteration")
    plt.ylabel("Mean observation error")
    plt.title("EKI vs MF-EKI on a convection-diffusion-reaction model")
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = os.path.join(EXAMPLE_DIR, "eki_mf_eki_demo.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
