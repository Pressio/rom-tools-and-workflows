"""Benchmark MF-EKI ensemble rejuvenation on the H2-air flame model."""

from __future__ import annotations

import argparse
import json
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MODELS_PATH = REPOSITORY_ROOT / "examples" / "models"
if str(EXAMPLE_MODELS_PATH) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_MODELS_PATH))

from h2_air_flame_model import H2AirFlameQoiModel  # noqa: E402


TRUTH = np.array([2.0, 8.0, 40.0, 7.0])
PARAMETER_MINS = np.array([0.5, 4.0, 20.0, 1.0])
PARAMETER_MAXES = np.array([4.0, 12.0, 60.0, 20.0])
PARAMETER_RANGES = PARAMETER_MAXES - PARAMETER_MINS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare EKI, baseline GP auto-ROM MF-EKI, and adaptive-rejuvenation "
            "MF-EKI on the H2-air flame inverse problem."
        )
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "work",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    return parser.parse_args()


def _parameter_space() -> HeterogeneousParameterSpace:
    return HeterogeneousParameterSpace(
        [
            UniformParameter("kappa", PARAMETER_MINS[0], PARAMETER_MAXES[0]),
            UniformParameter(
                "scaled_activation_energy", PARAMETER_MINS[1], PARAMETER_MAXES[1]
            ),
            UniformParameter("beta_x", PARAMETER_MINS[2], PARAMETER_MAXES[2]),
            UniformParameter("beta_y", PARAMETER_MINS[3], PARAMETER_MAXES[3]),
        ]
    )


def _load_history(work_dir: Path, mf: bool) -> dict[str, list]:
    residual = []
    parameter_relative_error = []
    scaled_ensemble_spread = []
    rejuvenation_count = []
    iteration = 0
    while True:
        restart_path = work_dir / f"iteration_{iteration}" / "restart.npz"
        if not restart_path.exists():
            break
        with np.load(restart_path, allow_pickle=True) as data:
            if mf:
                errors = data["sample_one_fom_results"].item()["errors"]
                primary_samples = data["parameter_samples_one"]
                all_samples = np.vstack(
                    [data["parameter_samples_one"], data["parameter_samples_two"]]
                )
            else:
                errors = data["errors"]
                primary_samples = data["parameter_samples"]
                all_samples = primary_samples

            mean_parameters = np.mean(primary_samples, axis=0)
            relative_error = np.linalg.norm(
                (mean_parameters - TRUTH) / TRUTH
            ) / np.sqrt(TRUTH.size)
            spread = np.linalg.norm(
                np.std(all_samples, axis=0, ddof=0) / PARAMETER_RANGES
            ) / np.sqrt(TRUTH.size)

            residual.append(float(np.mean(np.linalg.norm(errors, axis=0))))
            parameter_relative_error.append(float(relative_error))
            scaled_ensemble_spread.append(float(spread))
            rejuvenation_count.append(
                int(data["rejuvenation_count"])
                if "rejuvenation_count" in data
                else 0
            )
        iteration += 1

    return {
        "residual": residual,
        "parameter_relative_error": parameter_relative_error,
        "scaled_ensemble_spread": scaled_ensemble_spread,
        "rejuvenation_count": rejuvenation_count,
    }


def _summary(history: dict[str, list]) -> dict[str, float | int]:
    if not history["residual"]:
        raise RuntimeError("Benchmark produced no restart history.")
    return {
        "recorded_iterations": len(history["residual"]),
        "final_residual": history["residual"][-1],
        "minimum_residual": min(history["residual"]),
        "final_parameter_relative_error": history["parameter_relative_error"][-1],
        "final_scaled_ensemble_spread": history["scaled_ensemble_spread"][-1],
        "rejuvenations": history["rejuvenation_count"][-1],
    }


def _plot_histories(histories: dict[str, dict[str, list]], output_dir: Path) -> None:
    labels = {
        "eki": "Single-fidelity EKI",
        "mf_eki": "MF-EKI",
        "mf_eki_rejuvenated": "MF-EKI + adaptive rejuvenation",
    }
    markers = {"eki": "o", "mf_eki": "s", "mf_eki_rejuvenated": "^"}

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    for key, history in histories.items():
        iterations = np.arange(len(history["residual"]))
        axes[0].plot(
            iterations,
            history["residual"],
            marker=markers[key],
            label=labels[key],
        )
        axes[1].plot(
            iterations,
            history["parameter_relative_error"],
            marker=markers[key],
            label=labels[key],
        )
        axes[2].plot(
            iterations,
            history["scaled_ensemble_spread"],
            marker=markers[key],
            label=labels[key],
        )

    for axis in axes:
        axis.set_yscale("log")
        axis.set_xlabel("Outer iteration")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean observation residual")
    axes[1].set_ylabel("RMS relative parameter error")
    axes[2].set_ylabel("Range-scaled ensemble spread")
    axes[0].legend()
    figure.suptitle("H2-air flame: effect of adaptive MF-EKI rejuvenation")
    figure.tight_layout()
    figure.savefig(output_dir / "h2_air_flame_eki_rejuvenation_benchmark.svg")
    figure.savefig(
        output_dir / "h2_air_flame_eki_rejuvenation_benchmark.png", dpi=180
    )
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    np.random.seed(1)

    if args.smoke:
        nx, ny = 20, 10
        dt, t_end = 1.0e-3, 8.0e-3
        fom_ensemble_size, rom_extra_ensemble_size = 4, 4
        max_iterations = args.max_iterations or 2
        concurrency = 1
        rom_substep_start_iteration, rom_substep_end_iteration = 1, 2
        num_rom_substeps = 1
        max_rom_training_history = 2
        adaptive_delta_tolerance = 1.0e12
        max_rejuvenations = 1
    else:
        nx, ny = 64, 32
        dt, t_end = 1.0e-3, 6.0e-2
        fom_ensemble_size, rom_extra_ensemble_size = 4, 32
        max_iterations = args.max_iterations or 30
        concurrency = min(4, os.cpu_count() or 1)
        rom_substep_start_iteration, rom_substep_end_iteration = 3, 15
        num_rom_substeps = 3
        max_rom_training_history = 3
        adaptive_delta_tolerance = 1.0e-4
        max_rejuvenations = 3

    model = H2AirFlameQoiModel(
        nx=nx,
        ny=ny,
        dt=dt,
        t_end=t_end,
        snapshot_stride=10,
    )
    truth = {
        "kappa": TRUTH[0],
        "scaled_activation_energy": TRUTH[1],
        "beta_x": TRUTH[2],
        "beta_y": TRUTH[3],
    }
    parameter_space = _parameter_space()

    work_dir = args.work_dir.resolve()
    output_dir = args.output_dir.resolve()
    shutil.rmtree(work_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    truth_dir = work_dir / "truth"
    model.populate_run_directory(str(truth_dir), truth)
    model.run_model(str(truth_dir), truth)
    observations = model.compute_qoi(str(truth_dir), truth)
    observations_covariance = np.eye(observations.size) * 1.0e-4

    eki_dir = work_dir / "eki"
    mf_dir = work_dir / "mf_eki"
    mf_rejuvenated_dir = work_dir / "mf_eki_rejuvenated"

    common_mf_args = dict(
        model=model,
        parameter_space=parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=PARAMETER_MINS,
        parameter_maxes=PARAMETER_MAXES,
        fom_ensemble_size=fom_ensemble_size,
        rom_extra_ensemble_size=rom_extra_ensemble_size,
        max_rom_training_history=max_rom_training_history,
        rom_substep_start_iteration=rom_substep_start_iteration,
        rom_substep_end_iteration=rom_substep_end_iteration,
        num_rom_substeps=num_rom_substeps,
        max_iterations=max_iterations,
        random_seed=1,
        fom_evaluation_concurrency=concurrency,
        rom_type="gp",
        rom_args={
            "normalize_parameters": True,
            "normalize_targets": True,
        },
    )

    print("\n=== Baseline MF-EKI ===", flush=True)
    mf_eki_with_auto_rom(
        **common_mf_args,
        absolute_eki_directory=str(mf_dir),
        rejuvenation_strategy="none",
    )

    print("\n=== MF-EKI with adaptive rejuvenation ===", flush=True)
    mf_eki_with_auto_rom(
        **common_mf_args,
        absolute_eki_directory=str(mf_rejuvenated_dir),
        rejuvenation_strategy="adaptive",
        delta_params_tolerance=adaptive_delta_tolerance,
        max_rejuvenations=max_rejuvenations,
    )

    print("\n=== Single-fidelity EKI reference ===", flush=True)
    run_eki(
        model=model,
        parameter_space=parameter_space,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=PARAMETER_MINS,
        parameter_maxes=PARAMETER_MAXES,
        absolute_eki_directory=str(eki_dir),
        ensemble_size=fom_ensemble_size,
        max_iterations=max_iterations,
        random_seed=1,
        evaluation_concurrency=concurrency,
        rejuvenation_strategy="none",
    )

    histories = {
        "eki": _load_history(eki_dir, mf=False),
        "mf_eki": _load_history(mf_dir, mf=True),
        "mf_eki_rejuvenated": _load_history(mf_rejuvenated_dir, mf=True),
    }
    summaries = {key: _summary(history) for key, history in histories.items()}
    results = {
        "configuration": {
            "smoke": args.smoke,
            "nx": nx,
            "ny": ny,
            "dt": dt,
            "t_end": t_end,
            "max_iterations": max_iterations,
            "fom_ensemble_size": fom_ensemble_size,
            "rom_extra_ensemble_size": rom_extra_ensemble_size,
            "adaptive_delta_params_tolerance": adaptive_delta_tolerance,
            "rejuvenation_prior_weight": 0.0025,
            "rejuvenation_fraction_when_collapsed": 0.05,
            "random_seed": 1,
        },
        "summary": summaries,
        "history": histories,
    }

    _plot_histories(histories, output_dir)
    results_path = output_dir / "h2_air_flame_eki_rejuvenation_benchmark.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("\nBenchmark summary:", flush=True)
    for key, summary in summaries.items():
        print(f"  {key}: {summary}", flush=True)
    print(
        "BENCHMARK_SUMMARY_JSON=" + json.dumps(summaries, sort_keys=True),
        flush=True,
    )
    print(f"Wrote {results_path}", flush=True)


if __name__ == "__main__":
    main()
