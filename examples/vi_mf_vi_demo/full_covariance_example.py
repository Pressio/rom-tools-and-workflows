"""Full-covariance VI versus MF-VI for non-equispaced sine observations."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Optional

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
    write_convergence_plot,
    write_correlated_posterior_plot,
    write_mean_plot,
)


def main(
    smoke: bool = False,
    work_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Run the non-equispaced full-covariance example."""
    problem = build_problem(equispaced=False)
    model = AnalyticSineQoiModel(problem)

    sample_size = 8 if smoke else 16
    rom_extra_sample_size = 8 if smoke else 64
    max_iterations = 2 if smoke else 40

    prior = GaussianParameterSpace(
        parameter_names=list(problem.parameter_names),
        means=problem.prior_mean,
        stds=np.full(problem.prior_mean.size, PRIOR_STD),
        sampler=MonteCarloSampler,
    )
    initial_variational_parameter_space = MultivariateGaussianParameterSpace(
        parameter_names=list(problem.parameter_names),
        means=problem.prior_mean,
        covariance=problem.prior_covariance,
        sampler=MonteCarloSampler,
    )
    optimizer = VINewtonOptimizerConfig(
        newton_metric="natural",
        newton_regularization=1e-4,
        gradient_norm_tolerance=0.0,
        max_iterations=max_iterations,
    )
    line_search = VIStochasticNonmonotoneLineSearchConfig(
        max_step_size=1.0,
    )

    root = (
        Path(work_dir).resolve()
        if work_dir
        else Path(__file__).parent / "correlated_work"
    )
    shutil.rmtree(root, ignore_errors=True)
    vi_dir = root / "vi"
    mf_vi_dir = root / "mf_vi"

    common_arguments = dict(
        model=model,
        prior_parameter_space=prior,
        initial_variational_parameter_space=initial_variational_parameter_space,
        observations=problem.observations,
        observations_covariance=problem.observation_covariance,
        optimizer_method="newton",
        optimizer_config=optimizer,
        line_search_method="stochastic_nonmonotone",
        line_search_config=line_search,
        baseline_method="loo",
        score_function_entropy_strategy="joint",
        bounded_parameter_handling="clip",
        random_seed=19,
        restart_files_to_keep=max_iterations + 1,
    )

    workflows.run_vi(
        **common_arguments,
        absolute_work_dir=str(vi_dir),
        sample_size=sample_size,
        evaluation_concurrency=1,
    )
    workflows.mf_vi_with_auto_rom(
        **common_arguments,
        absolute_work_dir=str(mf_vi_dir),
        fom_sample_size=sample_size,
        rom_extra_sample_size=rom_extra_sample_size,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        max_rom_training_history=4,
        rom_type="gp",
        rom_args={
            "normalize_parameters": True,
            "normalize_targets": True,
        },
    )

    vi_history = collect_history(vi_dir, problem)
    mf_history = collect_history(mf_vi_dir, problem)
    output = (
        Path(output_dir).resolve()
        if output_dir
        else Path(__file__).parent
    )
    write_convergence_plot(
        output / "analytic_sine_correlated_convergence.png",
        vi_history,
        mf_history,
        "Non-equispaced observations: full-covariance convergence",
    )
    write_mean_plot(
        output / "analytic_sine_correlated_mean.png",
        problem,
        vi_history,
        mf_history,
        "Non-equispaced observations: posterior mean",
    )
    write_correlated_posterior_plot(
        output / "analytic_sine_correlated_posterior.png",
        problem,
        vi_history,
        mf_history,
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
