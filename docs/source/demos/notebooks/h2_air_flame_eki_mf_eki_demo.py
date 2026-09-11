"""Compare FOM EKI and GP auto-ROM MF-EKI for the H2-air flame model."""

import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from romtools.workflows.inverse.eki_drivers import run_eki
from romtools.workflows.inverse.mf_eki_drivers import mf_eki_with_auto_rom
from romtools.workflows.parameter_spaces import HeterogeneousParameterSpace
from romtools.workflows.parameters import UniformParameter


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_MODELS_PATH = PROJECT_ROOT / "examples" / "models"
if str(EXAMPLE_MODELS_PATH) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_MODELS_PATH))

from h2_air_flame_model import H2AirFlameQoiModel  # noqa: E402


def _collect_error_history(work_dir: Path, mf: bool) -> list[float]:
    """Return the mean observation-error norm recorded at each EKI iteration."""
    history = []
    iteration = 0
    while True:
        restart_path = work_dir / f"iteration_{iteration}" / "restart.npz"
        if not restart_path.exists():
            break
        with np.load(restart_path, allow_pickle=True) as data:
            if mf:
                errors = data["sample_one_fom_results"].item()["errors"]
            else:
                errors = data["errors"]
        history.append(float(np.mean(np.linalg.norm(errors, axis=0))))
        iteration += 1
    return history


def _collect_parameter_history(work_dir: Path, mf: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return the parameter-ensemble mean and standard deviation per iteration."""
    means = []
    standard_deviations = []
    iteration = 0
    while True:
        restart_path = work_dir / f"iteration_{iteration}" / "restart.npz"
        if not restart_path.exists():
            break
        with np.load(restart_path, allow_pickle=True) as data:
            samples = (
                data["parameter_samples_one"]
                if mf
                else data["parameter_samples"]
            )
        means.append(np.mean(samples, axis=0))
        standard_deviations.append(np.std(samples, axis=0))
        iteration += 1
    return np.asarray(means), np.asarray(standard_deviations)


def main() -> None:
    np.random.seed(1)

    model = H2AirFlameQoiModel(
        nx=64,
        ny=32,
        dt=1.0e-3,
        t_end=6.0e-2,
        snapshot_stride=10,
    )
    truth = {
        "kappa": 2.0,
        "scaled_activation_energy": 8.0,
        "beta_x": 40.0,
        "beta_y": 7.0,
    }
    parameter_space = HeterogeneousParameterSpace(
        [
            UniformParameter("kappa", 0.5, 4.0),
            UniformParameter("scaled_activation_energy", 4.0, 12.0),
            UniformParameter("beta_x", 20.0, 60.0),
            UniformParameter("beta_y", 1.0, 20.0),
        ]
    )
    parameter_mins = np.array([0.5, 4.0, 20.0, 1.0])
    parameter_maxes = np.array([4.0, 12.0, 60.0, 20.0])

    base_dir = Path(__file__).resolve().parent / "h2_air_flame_eki_mf_eki_work"
    eki_dir = base_dir / "eki"
    mf_auto_rom_dir = base_dir / "mf_eki_auto_rom"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    truth_dir = base_dir / "truth"
    model.populate_run_directory(str(truth_dir), truth)
    model.run_model(str(truth_dir), truth)
    observations = model.compute_qoi(str(truth_dir), truth)
    observations_covariance = np.eye(observations.size) * 1.0e-4


    mf_eki_with_auto_rom(
        model=model,
        parameter_space=parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=parameter_mins,
        parameter_maxes=parameter_maxes,
        absolute_eki_directory=str(mf_auto_rom_dir),
        fom_ensemble_size=4,
        rom_extra_ensemble_size=32,
        max_rom_training_history=3,
        rom_substep_start_iteration=3,
        rom_substep_end_iteration=15,
        num_rom_substeps=3,
        max_iterations=10,
        fom_evaluation_concurrency=4,
        rom_type="gp",
        rom_args={
            "normalize_parameters": True,
            "normalize_targets": True,
        },
    )

    run_eki(
        model=model,
        parameter_space=parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=parameter_mins,
        parameter_maxes=parameter_maxes,
        absolute_eki_directory=str(eki_dir),
        ensemble_size=4,
        max_iterations=10,
        evaluation_concurrency=4,
    )

    eki_history = _collect_error_history(eki_dir, mf=False)
    mf_auto_rom_history = _collect_error_history(mf_auto_rom_dir, mf=True)
    eki_parameter_means, eki_parameter_stds = _collect_parameter_history(
        eki_dir, mf=False
    )
    mf_auto_rom_parameter_means, mf_auto_rom_parameter_stds = (
        _collect_parameter_history(mf_auto_rom_dir, mf=True)
    )

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(eki_history, marker="o", label="EKI (FOM)")
    plt.plot(
        mf_auto_rom_history,
        marker="^",
        label="MF-EKI (FOM+GP auto-ROM)",
    )
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Mean observation error")
    plt.title("EKI vs GP auto-ROM MF-EKI on the H2-air flame model")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = Path(__file__).resolve().parent / "h2_air_flame_eki_mf_eki_demo.png"
    plt.savefig(output_path, dpi=180)
    print(f"Wrote {output_path}")

    parameter_names = parameter_space.get_names()
    truth_values = np.array([truth[name] for name in parameter_names])
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), sharex=True)
    for parameter_index, axis in enumerate(axes.flat):
        eki_iterations = np.arange(eki_parameter_means.shape[0])
        mf_iterations = np.arange(mf_auto_rom_parameter_means.shape[0])
        axis.plot(
            eki_iterations,
            eki_parameter_means[:, parameter_index],
            marker="o",
            label="EKI (FOM)",
        )
        axis.fill_between(
            eki_iterations,
            eki_parameter_means[:, parameter_index] - eki_parameter_stds[:, parameter_index],
            eki_parameter_means[:, parameter_index] + eki_parameter_stds[:, parameter_index],
            alpha=0.2,
        )
        axis.plot(
            mf_iterations,
            mf_auto_rom_parameter_means[:, parameter_index],
            marker="^",
            label="MF-EKI (FOM+GP auto-ROM)",
        )
        axis.fill_between(
            mf_iterations,
            mf_auto_rom_parameter_means[:, parameter_index]
            - mf_auto_rom_parameter_stds[:, parameter_index],
            mf_auto_rom_parameter_means[:, parameter_index]
            + mf_auto_rom_parameter_stds[:, parameter_index],
            alpha=0.2,
        )
        axis.axhline(
            truth_values[parameter_index],
            color="black",
            linestyle="--",
            label="Truth",
        )
        axis.set_title(parameter_names[parameter_index])
        axis.set_ylabel("Parameter value")
        axis.grid(True, alpha=0.3)

    for axis in axes[-1, :]:
        axis.set_xlabel("Iteration")
    axes[0, 0].legend()
    figure.suptitle("H2-air flame parameter convergence")
    figure.tight_layout()
    parameter_output_path = (
        Path(__file__).resolve().parent / "h2_air_flame_parameter_convergence.png"
    )
    figure.savefig(parameter_output_path, dpi=180)
    print(f"Wrote {parameter_output_path}")


if __name__ == "__main__":
    main()
