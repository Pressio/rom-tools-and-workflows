"""Compare single-fidelity VI with GP-based automatic-ROM MF-VI."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from romtools.workflows.inverse.mf_vi_drivers import mf_vi_with_auto_rom
from romtools.workflows.inverse.vi_drivers import run_vi
from romtools.workflows.inverse.vi_optimization_methods import (
    VINewtonOptimizerConfig,
    VIStochasticNonmonotoneLineSearchConfig,
)
from romtools.workflows.parameter_spaces import GaussianParameterSpace, MonteCarloSampler

from examples.eki_mf_eki_demo.example import CdrFomQoiModel, cdr  # noqa: E402


def _collect_vi_history(work_dir: Path):
    """Return the ELBO, variational-mean, and standard-deviation histories."""
    elbos = []
    iteration = 0
    while True:
        restart_path = work_dir / f"iteration_{iteration}" / "restart.npz"
        if not restart_path.exists():
            break
        with np.load(restart_path, allow_pickle=True) as data:
            elbos.append(float(data["elbo"]))
        iteration += 1

    # history.npz stores moments in physical parameter coordinates. In
    # transformed bounded problems, exp(variational_log_std) from a restart is
    # an optimizer-coordinate quantity and must not be plotted as a physical
    # posterior standard deviation.
    with np.load(work_dir / "history.npz") as history:
        means = np.asarray(history["vi_history_variational_mean"], dtype=float)
        covariances = np.asarray(
            history["vi_history_variational_covariance"], dtype=float
        )
    standard_deviations = np.sqrt(
        np.maximum(np.diagonal(covariances, axis1=1, axis2=2), 0.0)
    )
    return np.asarray(elbos), means, standard_deviations


def _write_plots(
    output_dir: Path,
    parameter_names,
    truth: np.ndarray,
    vi_history,
    mf_vi_history,
) -> None:
    """Write ELBO and posterior-parameter convergence figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vi_elbo, vi_means, vi_stds = vi_history
    mf_elbo, mf_means, mf_stds = mf_vi_history

    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    axis.plot(vi_elbo, marker="o", label="Single-fidelity VI")
    axis.plot(mf_elbo, marker="^", label="MF-VI with automatic GP ROM")
    axis.set(xlabel="Iteration", ylabel="ELBO", title="VI ELBO convergence")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "vi_mf_vi_elbo_convergence.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(
        1,
        len(parameter_names),
        figsize=(4.5 * len(parameter_names), 4.0),
        squeeze=False,
    )
    histories = (
        (vi_means, vi_stds, "Single-fidelity VI", "o"),
        (mf_means, mf_stds, "MF-VI with automatic GP ROM", "^"),
    )
    for parameter_index, axis in enumerate(axes.flat):
        for means, standard_deviations, label, marker in histories:
            iterations = np.arange(means.shape[0])
            axis.plot(iterations, means[:, parameter_index], marker=marker, label=label)
            axis.fill_between(
                iterations,
                means[:, parameter_index] - standard_deviations[:, parameter_index],
                means[:, parameter_index] + standard_deviations[:, parameter_index],
                alpha=0.2,
            )
        axis.axhline(truth[parameter_index], color="black", linestyle="--", label="Truth")
        axis.set(
            title=parameter_names[parameter_index],
            xlabel="Iteration",
            ylabel="Parameter value",
        )
        axis.grid(True, alpha=0.3)
    axes.flat[0].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "vi_mf_vi_parameter_convergence.png", dpi=180)
    plt.close(figure)


def main(
    smoke: bool = False,
    work_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Run the VI/MF-VI comparison."""
    grid_size = 10 if smoke else 25
    sample_size = 4 if smoke else 8
    rom_extra_sample_size = 4 if smoke else 64
    max_iterations = 2 if smoke else 50

    system = cdr.AdvectionDiffusionSystem(Nx=grid_size, Ny=grid_size)
    model = CdrFomQoiModel(system, np.array([1.0, 1.0]))
    truth_parameters = {"nu": 0.04, "sigma": 0.3}

    root = Path(work_dir).resolve() if work_dir else Path(__file__).parent / "vi_mf_vi_work"
    shutil.rmtree(root, ignore_errors=True)
    truth_dir = root / "truth"
    truth_dir.mkdir(parents=True)
    model.populate_run_directory(str(truth_dir), truth_parameters)
    model.run_model(str(truth_dir), truth_parameters)
    observations = model.compute_qoi(str(truth_dir), truth_parameters)

    parameter_mins = np.array([0.01, 0.1])
    parameter_maxes = np.array([0.08, 0.6])
    prior = GaussianParameterSpace(
        ["nu", "sigma"],
        (parameter_mins + parameter_maxes) / 2.0,
        (parameter_maxes - parameter_mins) / 4.0,
        MonteCarloSampler,
    )
    optimizer = VINewtonOptimizerConfig(
        max_iterations=max_iterations,
        gradient_norm_tolerance=0.0,
        newton_hessian_type="full",
        newton_curvature_strategy="lagged",
        newton_hessian_averaging_factor=0.5,
    )
    line_search = VIStochasticNonmonotoneLineSearchConfig()
    common_arguments = {
        "model": model,
        "prior_parameter_space": prior,
        "observations": observations,
        "observations_covariance": np.eye(observations.size) * 1.0e-5,
        "parameter_mins": parameter_mins,
        "parameter_maxes": parameter_maxes,
        "optimizer_method": "newton",
        "optimizer_config": optimizer,
        "line_search_method": "stochastic_nonmonotone",
        "line_search_config": line_search,
        "restart_files_to_keep": max_iterations,
        "baseline_method": "loo",
        "random_seed": 1,
    }

    vi_dir = root / "vi"
    mf_vi_dir = root / "mf_vi"
    run_vi(
        **common_arguments,
        absolute_vi_directory=str(vi_dir),
        sample_size=sample_size,
        evaluation_concurrency=1,
    )
    mf_vi_with_auto_rom(
        **common_arguments,
        absolute_vi_directory=str(mf_vi_dir),
        fom_sample_size=sample_size,
        rom_extra_sample_size=rom_extra_sample_size,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        rom_type="gp",
        rom_args={
            "normalize_parameters": True,
            "normalize_targets": True,
        },
    )

    output_path = Path(output_dir).resolve() if output_dir else Path(__file__).parent
    truth = np.array([truth_parameters[name] for name in prior.get_names()])
    _write_plots(
        output_path,
        prior.get_names(),
        truth,
        _collect_vi_history(vi_dir),
        _collect_vi_history(mf_vi_dir),
    )
    print(f"Wrote VI/MF-VI figures to {output_path}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the reduced configuration used for CI validation.",
    )
    parser.add_argument("--work-dir", help="Override the workflow output directory.")
    parser.add_argument("--output-dir", help="Override the figure output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    main(
        smoke=arguments.smoke,
        work_dir=arguments.work_dir,
        output_dir=arguments.output_dir,
    )
