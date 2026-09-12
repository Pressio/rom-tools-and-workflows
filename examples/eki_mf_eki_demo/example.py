import argparse
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

from romtools.rom import (
    LipschitzConfig,
    NeuralNetworkConfig,
    NeuralNetworkQoiModelBuilderWithTrainingData,
)
from romtools.workflows.inverse.eki_drivers import run_eki
from romtools.workflows.inverse.mf_eki_drivers import mf_eki_with_auto_rom, run_mf_eki
from romtools.workflows.parameter_spaces import HeterogeneousParameterSpace
from romtools.workflows.parameters import UniformParameter


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

    def build_from_training_dirs(
        self,
        offline_data_dir: str,
        training_data_dirs,
        training_parameters=None,
        training_qois=None,
    ):
        del offline_data_dir, training_parameters, training_qois
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


def main(smoke: bool = False, work_dir: str = None, output_path: str = None) -> None:
    np.random.seed(1)

    grid_size = 10 if smoke else 25
    fom_ensemble_size = 3 if smoke else 4
    rom_extra_ensemble_size = 4 if smoke else 12
    max_iterations = 2 if smoke else 20
    rom_dim = 4 if smoke else 12
    rom_substep_start_iteration = 0 if smoke else 1
    rom_substep_end_iteration = 1 if smoke else 15
    num_rom_substeps = 1 if smoke else 4
    max_rom_training_history = 2 if smoke else 3
    nn_training_iterations = 100 if smoke else 5000

    # Use the same EKI controls for every curve so differences are due to the
    # surrogate strategy rather than hidden workflow defaults.
    eki_solver_args = {
        "initial_step_size": 0.05,
        "regularization_parameter": 1.0e-4,
        "step_size_growth_factor": 1.25,
        "step_size_decay_factor": 2.0,
        "max_step_size_decrease_trys": 5,
        "relaxation_parameter": 1.05,
        "error_norm_tolerance": 1.0e-5,
        "delta_params_tolerance": 1.0e-6,
        "random_seed": 1,
    }
    mf_eki_solver_args = {
        **eki_solver_args,
        "use_updated_rom_in_update_on_rebuild": False,
    }

    system = cdr.AdvectionDiffusionSystem(Nx=grid_size, Ny=grid_size)
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

    base_dir = os.path.abspath(
        work_dir if work_dir is not None else os.path.join(EXAMPLE_DIR, "eki_mf_eki_work")
    )
    eki_dir = os.path.join(base_dir, "eki")
    mf_dir = os.path.join(base_dir, "mf_eki")
    mf_auto_rom_dir = os.path.join(base_dir, "mf_eki_auto_rom")
    mf_lipschitz_nn_dir = os.path.join(base_dir, "mf_eki_lipschitz_nn")
    shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(base_dir, exist_ok=True)

    fom_model = CdrFomQoiModel(system, b_vec)
    rom_builder = CdrRomBuilder(system, b_vec, rom_dim=rom_dim)
    lipschitz_nn_builder = NeuralNetworkQoiModelBuilderWithTrainingData(
        parameter_names=parameter_space.get_names(),
        network_config=NeuralNetworkConfig(
            training_iterations=nn_training_iterations,
        ),
        lipschitz_config=LipschitzConfig(
            enabled=True,
            safety_factor=1.1,
        ),
        normalize_parameters=True,
        normalize_targets=True,
    )

    run_eki(
        model=fom_model,
        parameter_space=parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        absolute_eki_directory=eki_dir,
        ensemble_size=fom_ensemble_size,
        max_iterations=max_iterations,
        evaluation_concurrency=1,
        **eki_solver_args,
    )

    run_mf_eki(
        model=fom_model,
        rom_model_builder=rom_builder,
        parameter_space=parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        absolute_eki_directory=mf_dir,
        rom_substep_start_iteration=rom_substep_start_iteration,
        rom_substep_end_iteration=rom_substep_end_iteration,
        num_rom_substeps=num_rom_substeps,
        fom_ensemble_size=fom_ensemble_size,
        rom_extra_ensemble_size=rom_extra_ensemble_size,
        rom_tolerance=0.001,
        max_iterations=max_iterations,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        max_rom_training_history=max_rom_training_history,
        **mf_eki_solver_args,
    )

    mf_eki_with_auto_rom(
        model=fom_model,
        parameter_space=parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        absolute_eki_directory=mf_auto_rom_dir,
        rom_substep_start_iteration=rom_substep_start_iteration,
        rom_substep_end_iteration=rom_substep_end_iteration,
        num_rom_substeps=num_rom_substeps,
        fom_ensemble_size=fom_ensemble_size,
        rom_extra_ensemble_size=rom_extra_ensemble_size,
        rom_tolerance=0.001,
        max_iterations=max_iterations,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        rom_type="gp",
        rom_args={
            "normalize_parameters": True,
            "normalize_targets": True,
        },
        max_rom_training_history=max_rom_training_history,
        **mf_eki_solver_args,
    )

    run_mf_eki(
        model=fom_model,
        rom_model_builder=lipschitz_nn_builder,
        parameter_space=parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        absolute_eki_directory=mf_lipschitz_nn_dir,
        rom_substep_start_iteration=rom_substep_start_iteration,
        rom_substep_end_iteration=rom_substep_end_iteration,
        num_rom_substeps=num_rom_substeps,
        fom_ensemble_size=fom_ensemble_size,
        rom_extra_ensemble_size=rom_extra_ensemble_size,
        rom_tolerance=0.001,
        max_iterations=max_iterations,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        max_rom_training_history=max_rom_training_history,
        **mf_eki_solver_args,
    )

    eki_history = _collect_error_history(eki_dir, mf=False)
    mf_history = _collect_error_history(mf_dir, mf=True)
    mf_auto_rom_history = _collect_error_history(mf_auto_rom_dir, mf=True)
    mf_lipschitz_nn_history = _collect_error_history(mf_lipschitz_nn_dir, mf=True)

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(eki_history, marker="o", label="EKI (FOM)")
    plt.plot(mf_history, marker="s", label="MF-EKI (FOM+ROM)")
    plt.plot(mf_auto_rom_history, marker="^", label="MF-EKI (FOM+GP auto-ROM)")
    plt.plot(
        mf_lipschitz_nn_history,
        marker="D",
        label="MF-EKI (FOM+Lipschitz NN auto-ROM)",
    )
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Mean observation error")
    plt.title("EKI vs MF-EKI on a convection-diffusion-reaction model")
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = os.path.abspath(
        output_path
        if output_path is not None
        else os.path.join(EXAMPLE_DIR, "eki_mf_eki_demo.png")
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Wrote {output_path}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the EKI/MF-EKI CDR demo.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a reduced configuration intended for CI validation.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Override the directory used for EKI iteration data.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override the output figure path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(smoke=args.smoke, work_dir=args.work_dir, output_path=args.output)
