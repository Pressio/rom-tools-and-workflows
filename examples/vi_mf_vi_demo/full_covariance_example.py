"""Full- and mean-field VI/MF-VI for non-equispaced sine observations."""

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

import romtools.workflows as workflows
from romtools.workflows.inverse.vi_optimization_methods import (
    VINewtonOptimizerConfig,
    VIStochasticNonmonotoneLineSearchConfig,
)
from romtools.workflows.parameter_spaces import (
    GaussianParameterSpace,
    MonteCarloSampler,
    MultivariateGaussianParameterSpace,
)

from examples.vi_mf_vi_demo.analytic_sine_support import (
    AnalyticSineQoiModel,
    PRIOR_STD,
    build_problem,
    collect_history,
    covariance_to_correlation,
    gaussian_kl,
    write_convergence_plot,
    write_correlated_posterior_plot,
    write_mean_plot,
)


def _mean_field_optimum(problem) -> tuple[np.ndarray, float]:
    """Return the reverse-KL optimal diagonal covariance and its KL floor."""
    posterior_precision = np.linalg.inv(problem.posterior_covariance)
    covariance = np.diag(1.0 / np.diag(posterior_precision))
    kl_floor = gaussian_kl(
        problem.posterior_mean,
        covariance,
        problem.posterior_mean,
        problem.posterior_covariance,
    )
    return covariance, kl_floor


def _write_family_convergence_plot(
    output_path: Path,
    problem,
    full_vi_history: dict[str, np.ndarray],
    full_mf_history: dict[str, np.ndarray],
    mean_field_vi_history: dict[str, np.ndarray],
    mean_field_mf_history: dict[str, np.ndarray],
) -> None:
    """Compare optimization error and the irreducible mean-field KL gap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, mean_field_kl_floor = _mean_field_optimum(problem)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    histories = (
        ("Full-covariance VI", full_vi_history, "o"),
        ("Full-covariance MF-VI", full_mf_history, "^"),
        ("Mean-field VI", mean_field_vi_history, "s"),
        ("Mean-field MF-VI", mean_field_mf_history, "D"),
    )
    for label, history, marker in histories:
        axis.semilogy(
            np.arange(history["kl"].size),
            np.maximum(history["kl"], 1.0e-14),
            marker=marker,
            markevery=max(1, history["kl"].size // 12),
            label=label,
        )
    axis.axhline(
        max(mean_field_kl_floor, 1.0e-14),
        linestyle="--",
        linewidth=1.5,
        label="Optimal mean-field KL floor",
    )
    axis.set(
        xlabel="Accepted variational state",
        ylabel=r"$D_{\mathrm{KL}}(q\,\|\,p_{\mathrm{post}})$",
        title="Correlated posterior: variational-family comparison",
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _write_family_correlation_plot(
    output_path: Path,
    problem,
    full_vi_history: dict[str, np.ndarray],
    full_mf_history: dict[str, np.ndarray],
    mean_field_vi_history: dict[str, np.ndarray],
    mean_field_mf_history: dict[str, np.ndarray],
) -> None:
    """Compare exact, full-covariance, and mean-field posterior correlations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_field_covariance, _ = _mean_field_optimum(problem)
    panels = (
        ("Exact posterior", problem.posterior_covariance),
        ("Full-covariance VI", full_vi_history["covariance"][-1]),
        ("Full-covariance MF-VI", full_mf_history["covariance"][-1]),
        ("Optimal mean-field", mean_field_covariance),
        ("Mean-field VI", mean_field_vi_history["covariance"][-1]),
        ("Mean-field MF-VI", mean_field_mf_history["covariance"][-1]),
    )

    figure, axes = plt.subplots(2, 3, figsize=(11.2, 7.0))
    image = None
    for axis, (title, covariance) in zip(axes.flat, panels):
        image = axis.imshow(
            covariance_to_correlation(covariance),
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )
        axis.set(
            title=title,
            xticks=np.arange(problem.posterior_mean.size),
            yticks=np.arange(problem.posterior_mean.size),
            xticklabels=np.arange(1, problem.posterior_mean.size + 1),
            yticklabels=np.arange(1, problem.posterior_mean.size + 1),
        )
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    figure.suptitle("Posterior correlation: full covariance versus mean field")
    figure.subplots_adjust(left=0.06, right=0.91, bottom=0.06, top=0.91, wspace=0.25, hspace=0.30)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main(
    smoke: bool = False,
    work_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Run the non-equispaced correlated-posterior example."""
    problem = build_problem(equispaced=False)
    model = AnalyticSineQoiModel(problem)

    sample_size = 8 if smoke else 16
    rom_extra_sample_size = 8 if smoke else 64
    max_iterations = 2 if smoke else 80

    prior = GaussianParameterSpace(
        parameter_names=list(problem.parameter_names),
        means=problem.prior_mean,
        stds=np.full(problem.prior_mean.size, PRIOR_STD),
        sampler=MonteCarloSampler,
    )
    full_covariance_initializer = MultivariateGaussianParameterSpace(
        parameter_names=list(problem.parameter_names),
        means=problem.prior_mean,
        covariance=problem.prior_covariance,
        sampler=MonteCarloSampler,
    )
    mean_field_initializer = GaussianParameterSpace(
        parameter_names=list(problem.parameter_names),
        means=problem.prior_mean,
        stds=np.full(problem.prior_mean.size, PRIOR_STD),
        sampler=MonteCarloSampler,
    )
    vi_optimizer = VINewtonOptimizerConfig(
        newton_metric="natural",
        newton_hessian_type="full",
        newton_curvature_strategy="lagged",
        newton_hessian_averaging_factor=0.25,
        newton_regularization=5e-4,
        gradient_norm_tolerance=0.0,
        max_iterations=max_iterations,
    )
    mf_vi_optimizer = VINewtonOptimizerConfig(
        newton_metric="natural",
        newton_hessian_type="full",
        newton_curvature_strategy="lagged",
        newton_hessian_averaging_factor=0.25,
        newton_regularization=1e-4,
        gradient_norm_tolerance=0.0,
        max_iterations=max_iterations,
    )
    line_search = VIStochasticNonmonotoneLineSearchConfig(
        initial_step_size=0.25,
        max_step_size=1.0,
    )

    root = (
        Path(work_dir).resolve()
        if work_dir
        else Path(__file__).parent / "correlated_work"
    )
    shutil.rmtree(root, ignore_errors=True)
    full_vi_dir = root / "full_vi"
    full_mf_vi_dir = root / "full_mf_vi"
    mean_field_vi_dir = root / "mean_field_vi"
    mean_field_mf_vi_dir = root / "mean_field_mf_vi"

    common_arguments = dict(
        model=model,
        prior_parameter_space=prior,
        observations=problem.observations,
        observations_covariance=problem.observation_covariance,
        optimizer_method="newton",
        line_search_method="stochastic_nonmonotone",
        line_search_config=line_search,
        baseline_method="loo",
        score_function_entropy_strategy="joint",
        bounded_parameter_handling="clip",
        covariance_regularization=0.0,
        random_seed=7,
        restart_files_to_keep=max_iterations + 1,
    )

    workflows.run_vi(
        **common_arguments,
        initial_variational_parameter_space=full_covariance_initializer,
        absolute_work_dir=str(full_vi_dir),
        sample_size=sample_size,
        evaluation_concurrency=1,
        optimizer_config=vi_optimizer,
    )
    workflows.mf_vi_with_auto_rom(
        **common_arguments,
        initial_variational_parameter_space=full_covariance_initializer,
        absolute_work_dir=str(full_mf_vi_dir),
        fom_sample_size=sample_size,
        rom_extra_sample_size=rom_extra_sample_size,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        max_rom_training_history=4,
        mfmc_control_variate_mode="scalar",
        rom_type="gp",
        rom_args={
            "normalize_parameters": True,
            "normalize_targets": True,
        },
        optimizer_config=mf_vi_optimizer,
    )
    workflows.run_vi(
        **common_arguments,
        initial_variational_parameter_space=mean_field_initializer,
        absolute_work_dir=str(mean_field_vi_dir),
        sample_size=sample_size,
        evaluation_concurrency=1,
        optimizer_config=vi_optimizer,
    )
    workflows.mf_vi_with_auto_rom(
        **common_arguments,
        initial_variational_parameter_space=mean_field_initializer,
        absolute_work_dir=str(mean_field_mf_vi_dir),
        fom_sample_size=sample_size,
        rom_extra_sample_size=rom_extra_sample_size,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        max_rom_training_history=4,
        mfmc_control_variate_mode="scalar",
        rom_type="gp",
        rom_args={
            "normalize_parameters": True,
            "normalize_targets": True,
        },
        optimizer_config=mf_vi_optimizer,
    )

    full_vi_history = collect_history(full_vi_dir, problem)
    full_mf_history = collect_history(full_mf_vi_dir, problem)
    mean_field_vi_history = collect_history(mean_field_vi_dir, problem)
    mean_field_mf_history = collect_history(mean_field_mf_vi_dir, problem)
    output = (
        Path(output_dir).resolve()
        if output_dir
        else Path(__file__).parent
    )
    write_convergence_plot(
        output / "analytic_sine_correlated_convergence.png",
        full_vi_history,
        full_mf_history,
        "Non-equispaced observations: full-covariance convergence",
    )
    write_mean_plot(
        output / "analytic_sine_correlated_mean.png",
        problem,
        full_vi_history,
        full_mf_history,
        "Non-equispaced observations: posterior mean",
    )
    write_correlated_posterior_plot(
        output / "analytic_sine_correlated_posterior.png",
        problem,
        full_vi_history,
        full_mf_history,
    )
    _write_family_convergence_plot(
        output / "analytic_sine_correlated_family_convergence.png",
        problem,
        full_vi_history,
        full_mf_history,
        mean_field_vi_history,
        mean_field_mf_history,
    )
    _write_family_correlation_plot(
        output / "analytic_sine_correlated_family_correlation.png",
        problem,
        full_vi_history,
        full_mf_history,
        mean_field_vi_history,
        mean_field_mf_history,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--work-dir")
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    main(
        smoke=arguments.smoke,
        work_dir=arguments.work_dir,
        output_dir=arguments.output_dir,
    )
