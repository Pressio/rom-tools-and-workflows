"""EKI and GP auto-ROM MF-EKI benchmark for the solid-dynamics cantilever.

The inverse problem estimates Young's modulus, Poisson's ratio, and transient
load amplitude from noisy transverse-displacement histories at three sensors.
Synthetic truth and the inference FOM intentionally use the same spatial and
temporal discretization. MF-EKI uses romtools' on-the-fly Gaussian-process QoI
surrogate as its low-fidelity model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from romtools.workflows.inverse.eki_drivers import run_eki
from romtools.workflows.inverse.mf_eki_drivers import mf_eki_with_auto_rom
from romtools.workflows.parameter_spaces import HeterogeneousParameterSpace
from romtools.workflows.parameters import UniformParameter


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
MODELS = REPO_ROOT / "examples" / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from solid_dynamics import cantilever_model  # noqa: E402


E_RANGE = (0.6e6, 1.4e6)
NU_RANGE = (0.20, 0.40)
F0_RANGE = (700.0, 1300.0)
TRUTH = {"E": 1.15e6, "nu": 0.33, "F0": 900.0}
SENSOR_X_OVER_L = np.array([0.50, 0.75, 1.00])
PULSE_DURATION = 0.15
DENSITY = 1050.0


@dataclass(frozen=True)
class Discretization:
    nx: int
    ny: int
    dt: float
    t_end: float

    @property
    def num_steps(self) -> int:
        return int(round(self.t_end / self.dt))


@dataclass(frozen=True)
class BenchmarkConfig:
    fom: Discretization
    observation_times: np.ndarray
    eki_ensemble_size: int
    mf_fom_ensemble_size: int
    mf_extra_gp_ensemble_size: int
    max_iterations: int
    gp_substep_start_iteration: int
    gp_substep_end_iteration: int
    num_gp_substeps: int
    max_gp_training_history: int


def configuration(smoke: bool) -> BenchmarkConfig:
    """Return the production benchmark or a reduced CI-only configuration."""
    if smoke:
        return BenchmarkConfig(
            fom=Discretization(4, 2, 0.05, 0.20),
            observation_times=np.array([0.05, 0.10, 0.15, 0.20]),
            eki_ensemble_size=3,
            mf_fom_ensemble_size=3,
            mf_extra_gp_ensemble_size=3,
            max_iterations=2,
            gp_substep_start_iteration=0,
            gp_substep_end_iteration=2,
            num_gp_substeps=1,
            max_gp_training_history=2,
        )
    return BenchmarkConfig(
        # Truth and inference HF deliberately use this same discretization.
        fom=Discretization(33, 16, 0.005, 1.0),
        observation_times=np.arange(0.05, 1.001, 0.05),
        eki_ensemble_size=8,
        mf_fom_ensemble_size=8,
        mf_extra_gp_ensemble_size=24,
        max_iterations=15,
        gp_substep_start_iteration=1,
        gp_substep_end_iteration=15,
        num_gp_substeps=4,
        max_gp_training_history=5,
    )


def transformed_sample(E: float, nu: float, F0: float) -> dict[str, float]:
    return {
        "log_E": float(np.log(E)),
        "nu": float(nu),
        "log_F0": float(np.log(F0)),
    }


def physical_parameters(parameter_sample: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(np.exp(parameter_sample["log_E"])),
        float(parameter_sample["nu"]),
        float(np.exp(parameter_sample["log_F0"])),
    )


class CantileverInverseQoiModel:
    """Parameterized FOM returning concatenated transverse sensor histories."""

    def __init__(
        self,
        discretization: Discretization,
        observation_times: np.ndarray,
        fidelity: str = "high",
    ) -> None:
        self.discretization = discretization
        self.observation_times = np.asarray(observation_times, dtype=float)
        self.fidelity = fidelity
        time_indices = np.rint(self.observation_times / discretization.dt).astype(int)
        represented_times = time_indices * discretization.dt
        if not np.allclose(
            represented_times,
            self.observation_times,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("Observation times must lie on the FOM time grid")
        self._time_indices = time_indices

    def _solve(self, parameter_sample: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        young_modulus, poisson_ratio, load_amplitude = physical_parameters(parameter_sample)
        model = cantilever_model(
            nx=self.discretization.nx,
            ny=self.discretization.ny,
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
            density=DENSITY,
            mass_type="consistent",
            material_model="neo_hookean",
        )
        edge_shape = model.boundary_force_x(
            model.mesh.length,
            np.array([0.0, -1.0]),
        )

        def external_force(current_time: float) -> np.ndarray:
            if 0.0 <= current_time <= PULSE_DURATION:
                pulse = np.sin(np.pi * current_time / PULSE_DURATION)
                return load_amplitude * pulse * edge_shape
            return np.zeros(model.ndof)

        state = model.initial_state(external_force=external_force)
        times, displacements, _velocities = model.solve_implicit(
            state,
            dt=self.discretization.dt,
            num_steps=self.discretization.num_steps,
            external_force=external_force,
            newton_tolerance=1.0e-8,
            snapshot_stride=1,
        )

        sensor_dofs = []
        for x_fraction in SENSOR_X_OVER_L:
            node = model.mesh.node_nearest(
                x_fraction * model.mesh.length,
                0.5 * model.mesh.height,
            )
            sensor_dofs.append(2 * node + 1)
        sensor_dofs = np.asarray(sensor_dofs, dtype=int)
        sampled = displacements[np.ix_(self._time_indices, sensor_dofs)]
        qoi = sampled.T.reshape(-1)
        return times, qoi

    def evaluate_qoi(self, parameter_sample: dict[str, float]) -> np.ndarray:
        return self._solve(parameter_sample)[1]

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        run_path = Path(run_directory)
        run_path.mkdir(parents=True, exist_ok=True)
        with (run_path / "params.json").open("w", encoding="utf-8") as handle:
            json.dump(parameter_sample, handle, indent=2)

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        start = time.perf_counter()
        times, qoi = self._solve(parameter_sample)
        elapsed = time.perf_counter() - start
        np.savez(
            Path(run_directory) / "solution.npz",
            times=times,
            qoi=qoi,
            fidelity=self.fidelity,
            elapsed_seconds=elapsed,
        )
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        del parameter_sample
        with np.load(Path(run_directory) / "solution.npz") as data:
            return np.array(data["qoi"], copy=True)


def parameter_space() -> HeterogeneousParameterSpace:
    return HeterogeneousParameterSpace(
        [
            UniformParameter("log_E", np.log(E_RANGE[0]), np.log(E_RANGE[1])),
            UniformParameter("nu", NU_RANGE[0], NU_RANGE[1]),
            UniformParameter("log_F0", np.log(F0_RANGE[0]), np.log(F0_RANGE[1])),
        ]
    )


def parameter_bounds() -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([np.log(E_RANGE[0]), NU_RANGE[0], np.log(F0_RANGE[0])])
    upper = np.array([np.log(E_RANGE[1]), NU_RANGE[1], np.log(F0_RANGE[1])])
    return lower, upper


def _restart_files(work_dir: Path) -> list[Path]:
    files = []
    iteration = 0
    while True:
        path = work_dir / f"iteration_{iteration}" / "restart.npz"
        if not path.exists():
            break
        files.append(path)
        iteration += 1
    return files


def _parameter_history(work_dir: Path, multifidelity: bool) -> np.ndarray:
    history = []
    for restart in _restart_files(work_dir):
        with np.load(restart, allow_pickle=True) as data:
            samples = (
                data["parameter_samples_one"]
                if multifidelity
                else data["parameter_samples"]
            )
            E = np.mean(np.exp(samples[:, 0]))
            nu = np.mean(samples[:, 1])
            F0 = np.mean(np.exp(samples[:, 2]))
            history.append([E, nu, F0])
    return np.asarray(history, dtype=float)


def _error_history(work_dir: Path, multifidelity: bool) -> np.ndarray:
    history = []
    for restart in _restart_files(work_dir):
        with np.load(restart, allow_pickle=True) as data:
            if multifidelity:
                errors = data["sample_one_fom_results"].item()["errors"]
            else:
                errors = data["errors"]
            history.append(float(np.mean(np.linalg.norm(errors, axis=0))))
    return np.asarray(history)


def _hf_evaluation_history(work_dir: Path) -> np.ndarray:
    """Cumulative HF solves; GP evaluations are treated as negligible cost."""
    cumulative = []
    running = 0
    iteration = 0
    while True:
        iteration_dir = work_dir / f"iteration_{iteration}"
        if not iteration_dir.exists():
            break
        for solution in iteration_dir.rglob("solution.npz"):
            with np.load(solution) as data:
                if "fidelity" in data and str(data["fidelity"]) == "high":
                    running += 1
        cumulative.append(float(running))
        iteration += 1
    return np.asarray(cumulative)


def _hf_wallclock(work_dir: Path) -> float:
    total = 0.0
    for solution in work_dir.rglob("solution.npz"):
        with np.load(solution) as data:
            if (
                "fidelity" in data
                and str(data["fidelity"]) == "high"
                and "elapsed_seconds" in data
            ):
                total += float(data["elapsed_seconds"])
    return total


def _plot_truth(
    observation_times: np.ndarray,
    clean: np.ndarray,
    noisy: np.ndarray,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    n_times = observation_times.size
    for i, x_fraction in enumerate(SENSOR_X_OVER_L):
        sl = slice(i * n_times, (i + 1) * n_times)
        ax.plot(
            observation_times,
            clean[sl],
            label=f"truth x/L={x_fraction:.2f}",
        )
        ax.plot(observation_times, noisy[sl], "o", ms=3, alpha=0.65)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Transverse displacement [m]")
    ax.set_title("Synthetic cantilever observations")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_parameter_history(
    eki_history: np.ndarray,
    mf_history: np.ndarray,
    output: Path,
) -> None:
    truth = np.array([TRUTH["E"], TRUTH["nu"], TRUTH["F0"]])
    labels = [
        "Young's modulus E [MPa]",
        "Poisson ratio nu",
        "Load amplitude F0 [N]",
    ]
    scales = [1.0e-6, 1.0, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for j, ax in enumerate(axes):
        if eki_history.size:
            ax.plot(
                np.arange(len(eki_history)),
                eki_history[:, j] * scales[j],
                "o-",
                label="EKI",
            )
        if mf_history.size:
            ax.plot(
                np.arange(len(mf_history)),
                mf_history[:, j] * scales[j],
                "s-",
                label="MF-EKI (GP auto-ROM)",
            )
        ax.axhline(
            truth[j] * scales[j],
            ls="--",
            color="0.25",
            label="truth" if j == 0 else None,
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(labels[j])
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle("Solid-dynamics parameter estimates")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_error_vs_cost(
    eki_error: np.ndarray,
    mf_error: np.ndarray,
    eki_cost: np.ndarray,
    mf_cost: np.ndarray,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    n = min(len(eki_error), len(eki_cost))
    if n:
        ax.plot(eki_cost[:n], eki_error[:n], "o-", label="EKI")
    n = min(len(mf_error), len(mf_cost))
    if n:
        ax.plot(
            mf_cost[:n],
            mf_error[:n],
            "s-",
            label="MF-EKI (GP auto-ROM)",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative high-fidelity model evaluations")
    ax.set_ylabel("Mean observation error")
    ax.set_title("Convergence versus high-fidelity cost")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(
    smoke: bool = False,
    work_dir: str | None = None,
    output_dir: str | None = None,
) -> None:
    cfg = configuration(smoke)
    base_dir = Path(work_dir) if work_dir is not None else EXAMPLE_DIR / "work"
    figures_dir = Path(output_dir) if output_dir is not None else EXAMPLE_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    high_model = CantileverInverseQoiModel(
        cfg.fom,
        cfg.observation_times,
        fidelity="high",
    )

    # Synthetic truth intentionally uses the exact same FOM discretization as
    # the HF inference model for this controlled algorithmic benchmark.
    truth_sample = transformed_sample(**TRUTH)
    clean_observations = high_model.evaluate_qoi(truth_sample)
    noise_sigma = 0.01 * max(
        float(np.max(np.abs(clean_observations))),
        1.0e-8,
    )
    rng = np.random.default_rng(17)
    observations = clean_observations + rng.normal(
        0.0,
        noise_sigma,
        clean_observations.shape,
    )
    observations_covariance = np.eye(observations.size) * noise_sigma**2

    lower, upper = parameter_bounds()
    space = parameter_space()
    eki_dir = base_dir / "eki"
    mf_dir = base_dir / "mf_eki_gp_auto_rom"

    solver_args = dict(
        initial_step_size=0.25,
        regularization_parameter=1.0e-8,
        step_size_growth_factor=1.25,
        step_size_decay_factor=2.0,
        max_step_size_decrease_trys=5,
        relaxation_parameter=1.05,
        error_norm_tolerance=1.0e-5,
        delta_params_tolerance=1.0e-6,
        max_iterations=cfg.max_iterations,
        random_seed=5,
    )

    run_eki(
        model=high_model,
        parameter_space=space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=lower,
        parameter_maxes=upper,
        absolute_work_dir=str(eki_dir),
        ensemble_size=cfg.eki_ensemble_size,
        evaluation_concurrency=1,
        **solver_args,
    )

    mf_eki_with_auto_rom(
        model=high_model,
        parameter_space=space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=lower,
        parameter_maxes=upper,
        absolute_work_dir=str(mf_dir),
        fom_ensemble_size=cfg.mf_fom_ensemble_size,
        rom_extra_ensemble_size=cfg.mf_extra_gp_ensemble_size,
        rom_tolerance=0.005,
        max_rom_training_history=cfg.max_gp_training_history,
        rom_substep_start_iteration=cfg.gp_substep_start_iteration,
        rom_substep_end_iteration=cfg.gp_substep_end_iteration,
        num_rom_substeps=cfg.num_gp_substeps,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        use_updated_rom_in_update_on_rebuild=False,
        rom_type="gp",
        rom_args={
            "normalize_parameters": True,
            "normalize_targets": True,
            "noise_variance_fraction": 1.0e-6,
        },
        **solver_args,
    )

    eki_parameters = _parameter_history(eki_dir, multifidelity=False)
    mf_parameters = _parameter_history(mf_dir, multifidelity=True)
    eki_error = _error_history(eki_dir, multifidelity=False)
    mf_error = _error_history(mf_dir, multifidelity=True)
    eki_cost = _hf_evaluation_history(eki_dir)
    mf_cost = _hf_evaluation_history(mf_dir)

    _plot_truth(
        cfg.observation_times,
        clean_observations,
        observations,
        figures_dir / "solid_dynamics_observations.png",
    )
    _plot_parameter_history(
        eki_parameters,
        mf_parameters,
        figures_dir / "solid_dynamics_parameter_convergence.png",
    )
    _plot_error_vs_cost(
        eki_error,
        mf_error,
        eki_cost,
        mf_cost,
        figures_dir / "solid_dynamics_error_vs_cost.png",
    )

    def final_estimate(history: np.ndarray) -> dict[str, float] | None:
        if not history.size:
            return None
        return {
            "E": float(history[-1, 0]),
            "nu": float(history[-1, 1]),
            "F0": float(history[-1, 2]),
        }

    summary = {
        "truth": TRUTH,
        "noise_sigma": noise_sigma,
        "fom_discretization": vars(cfg.fom),
        "observation_times": cfg.observation_times.tolist(),
        "num_sensors": int(SENSOR_X_OVER_L.size),
        "qoi_dimension": int(clean_observations.size),
        "eki_settings": {
            "initial_step_size": solver_args["initial_step_size"],
            "regularization_parameter": solver_args["regularization_parameter"],
        },
        "truth_and_hf_share_discretization": True,
        "low_fidelity_model": "Gaussian-process automatic QoI ROM",
        "gp_settings": {
            "normalize_parameters": True,
            "normalize_targets": True,
            "noise_variance_fraction": 1.0e-6,
            "rom_tolerance": 0.005,
            "max_training_history": cfg.max_gp_training_history,
            "extra_ensemble_size": cfg.mf_extra_gp_ensemble_size,
            "num_rom_substeps": cfg.num_gp_substeps,
        },
        "eki_final_estimate": final_estimate(eki_parameters),
        "mf_eki_final_estimate": final_estimate(mf_parameters),
        "eki_final_error": None if not eki_error.size else float(eki_error[-1]),
        "mf_eki_final_error": None if not mf_error.size else float(mf_error[-1]),
        "eki_hf_evaluations": None if not eki_cost.size else float(eki_cost[-1]),
        "mf_eki_hf_evaluations": None if not mf_cost.size else float(mf_cost[-1]),
        "eki_hf_wallclock_seconds": _hf_wallclock(eki_dir),
        "mf_eki_hf_wallclock_seconds": _hf_wallclock(mf_dir),
    }
    with (
        figures_dir / "solid_dynamics_eki_mf_eki_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a reduced CI configuration.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory for EKI/MF-EKI iteration data.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for figures and summary JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        smoke=args.smoke,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
    )
