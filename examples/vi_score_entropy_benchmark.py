"""Compare analytic-entropy and full-ELBO score-function VI estimators.

The target is a standard Gaussian, so the variational optimum and KL divergence
are analytic. The benchmark writes direct optimum-gradient statistics and
iteration histories, including the step size that was actually accepted.

Run the full study with::

    python -m examples.vi_score_entropy_benchmark --output-dir work/score_entropy
"""

import argparse
import contextlib
import csv
import io
from pathlib import Path

import numpy as np

from romtools.workflows.inverse.vi_drivers import (
    _average_hessians,
    _compute_reinforce_gradients,
    _compute_reinforce_hessian_full,
    _compute_variational_log_densities,
)
from romtools.workflows.inverse.mf_vi_drivers import _compute_mfmc_reinforce_gradients
from romtools.workflows.inverse.vi_optimization_methods import NewtonSolver


def _estimate(rng, sample_count, mean, log_std, entropy_strategy):
    std = np.exp(log_std)
    samples = mean + std * rng.standard_normal((sample_count, 1))
    log_joint = -0.5 * (samples[:, 0] ** 2 + np.log(2.0 * np.pi))
    gradient_mean, gradient_log_std, _, _, _, _ = _compute_reinforce_gradients(
        samples,
        np.array([mean]),
        np.array([std]),
        log_joint,
        baseline_method="loo",
        score_function_entropy_strategy=entropy_strategy,
    )
    hessian = _compute_reinforce_hessian_full(
        samples,
        np.array([mean]),
        np.array([std]),
        log_joint,
        baseline_method="loo",
    )
    return np.array([gradient_mean[0], gradient_log_std[0]]), hessian


def _kl(mean, log_std):
    variance = np.exp(2.0 * log_std)
    return max(0.0, 0.5 * (mean ** 2 + variance - 1.0 - 2.0 * log_std))


def direct_statistics(sample_counts, repetitions, seed):
    rows = []
    for sample_count in sample_counts:
        for entropy_strategy in ("analytic", "joint"):
            estimates = np.empty((repetitions, 2))
            for repetition in range(repetitions):
                rng = np.random.default_rng(seed + sample_count * 100000 + repetition)
                estimates[repetition], _ = _estimate(
                    rng, sample_count, 0.0, 0.0, entropy_strategy
                )
            for component, name in enumerate(("mean", "log_std")):
                rows.append({
                    "sample_count": sample_count,
                    "entropy_strategy": entropy_strategy,
                    "component": name,
                    "gradient_mean": np.mean(estimates[:, component]),
                    "gradient_variance": np.var(estimates[:, component], ddof=1),
                })
    return rows


def convergence_histories(sample_counts, seeds, iterations, beta):
    rows = []
    solver = NewtonSolver(regularization=1e-2, hessian_type="full")
    for sample_count in sample_counts:
        for entropy_strategy in ("analytic", "joint"):
            for curvature_strategy in ("same_sample", "lagged"):
                for seed in seeds:
                    rng = np.random.default_rng(seed + sample_count * 1000)
                    mean, log_std = 1.0, np.log(1.8)
                    running_hessian = None
                    proposed_step = 1.0
                    for iteration in range(iterations):
                        gradient, current_hessian = _estimate(
                            rng, sample_count, mean, log_std, entropy_strategy
                        )
                        if running_hessian is None:
                            running_hessian = current_hessian.copy()
                        metric = (
                            current_hessian
                            if curvature_strategy == "same_sample"
                            else running_hessian
                        )
                        with contextlib.redirect_stdout(io.StringIO()):
                            direction = solver.step(gradient, metric)
                        old_kl = _kl(mean, log_std)
                        accepted_step = proposed_step
                        while accepted_step >= 2.0 ** -20:
                            candidate_mean = mean + accepted_step * direction[0]
                            candidate_log_std = np.clip(
                                log_std + accepted_step * direction[1], -8.0, 8.0
                            )
                            if _kl(candidate_mean, candidate_log_std) <= old_kl + 1e-15:
                                mean, log_std = candidate_mean, candidate_log_std
                                break
                            accepted_step *= 0.5
                        else:
                            accepted_step = 0.0
                        running_hessian = _average_hessians(
                            running_hessian, current_hessian, beta
                        )
                        proposed_step = min(
                            1.0, 1.2 * accepted_step if accepted_step > 0.0 else proposed_step
                        )
                        rows.append({
                            "sample_count": sample_count,
                            "seed": seed,
                            "iteration": iteration,
                            "entropy_strategy": entropy_strategy,
                            "curvature_strategy": curvature_strategy,
                            "kl": _kl(mean, log_std),
                            "gradient_norm": np.linalg.norm(gradient),
                            "accepted_step_size": accepted_step,
                            "mean_error": abs(mean),
                            "variance_error": abs(np.exp(2.0 * log_std) - 1.0),
                        })
    return rows


def mfmc_optimum_statistics(sample_counts, repetitions, seed):
    rows = []
    for sample_count in sample_counts:
        for rom_case in ("exact", "perturbed"):
            for coefficient in ("fitted", "fixed"):
                gradients = []
                alphas = []
                for repetition in range(repetitions):
                    rng = np.random.default_rng(
                        seed + 700000 + sample_count * 10000 + repetition
                    )
                    fom_samples = rng.standard_normal((sample_count, 1))
                    extra_samples = rng.standard_normal((4 * sample_count, 1))
                    fom_log_q = _compute_variational_log_densities(
                        fom_samples, np.zeros(1), np.ones(1)
                    )
                    extra_log_q = _compute_variational_log_densities(
                        extra_samples, np.zeros(1), np.ones(1)
                    )
                    if rom_case == "exact":
                        rom_base_joint = fom_log_q
                        rom_extra_joint = extra_log_q
                    else:
                        rom_base_joint = fom_log_q + 0.15 * fom_samples[:, 0]
                        rom_extra_joint = extra_log_q + 0.15 * extra_samples[:, 0]
                    result = _compute_mfmc_reinforce_gradients(
                        fom_samples,
                        fom_samples,
                        extra_samples,
                        np.zeros(1),
                        np.ones(1),
                        fom_log_q,
                        rom_base_joint,
                        rom_extra_joint,
                        baseline_method="loo",
                        use_mfmc_control_variate=coefficient == "fitted",
                        score_function_entropy_strategy="joint",
                    )
                    gradients.append(np.concatenate(result[:2]))
                    alphas.append(np.concatenate([
                        np.asarray(result[4]).reshape(-1),
                        np.asarray(result[5]).reshape(-1),
                    ]))
                rows.append({
                    "sample_count": sample_count,
                    "rom_case": rom_case,
                    "coefficient": coefficient,
                    "gradient_mean_norm": np.linalg.norm(np.mean(gradients, axis=0)),
                    "gradient_variance": np.mean(np.var(gradients, axis=0, ddof=1)),
                    "alpha_mean_norm": np.mean(np.linalg.norm(alphas, axis=1)),
                })
    return rows


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path, direct_rows, mfmc_rows, history_rows, args):
    final_rows = [row for row in history_rows if row["iteration"] == args.iterations - 1]
    lines = [
        "# Score-function entropy benchmark",
        "",
        f"Direct repetitions: {args.repetitions}; convergence seeds: {args.seeds}; "
        f"iterations: {args.iterations}; lagged beta: {args.beta}.",
        "",
        "| Samples | Entropy | Component | Optimum gradient variance |",
        "|---:|---|---|---:|",
    ]
    for row in direct_rows:
        lines.append(
            f"| {row['sample_count']} | {row['entropy_strategy']} | {row['component']} | "
            f"{row['gradient_variance']:.6e} |"
        )
    lines.extend([
        "",
        "| Samples | ROM | Coefficient | MF gradient variance | Mean alpha norm |",
        "|---:|---|---|---:|---:|",
    ])
    for row in mfmc_rows:
        lines.append(
            f"| {row['sample_count']} | {row['rom_case']} | {row['coefficient']} | "
            f"{row['gradient_variance']:.6e} | {row['alpha_mean_norm']:.6e} |"
        )
    lines.extend([
        "",
        "| Samples | Entropy | Curvature | Median final KL | Median accepted step |",
        "|---:|---|---|---:|---:|",
    ])
    keys = sorted({
        (row["sample_count"], row["entropy_strategy"], row["curvature_strategy"])
        for row in final_rows
    })
    for key in keys:
        selected = [
            row for row in final_rows
            if (row["sample_count"], row["entropy_strategy"], row["curvature_strategy"]) == key
        ]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | "
            f"{np.median([row['kl'] for row in selected]):.6e} | "
            f"{np.median([row['accepted_step_size'] for row in selected]):.6e} |"
        )
    lines.extend([
        "",
        "The joint estimator uses a leave-one-out baseline. Its variance collapses "
        "at the representable Gaussian optimum; early-iteration behavior and noisy "
        "line-search decisions remain empirical properties of each run.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plots(output_dir, history_rows, sample_counts, iterations):
    import matplotlib.pyplot as plt

    for metric, filename, logarithmic in (
        ("kl", "kl_history.png", True),
        ("accepted_step_size", "accepted_step_history.png", False),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        for axis, curvature_strategy in zip(axes, ("same_sample", "lagged")):
            for sample_count in sample_counts:
                for entropy_strategy, linestyle in (("analytic", "--"), ("joint", "-")):
                    medians = []
                    for iteration in range(iterations):
                        values = [
                            row[metric] for row in history_rows
                            if row["sample_count"] == sample_count
                            and row["entropy_strategy"] == entropy_strategy
                            and row["curvature_strategy"] == curvature_strategy
                            and row["iteration"] == iteration
                        ]
                        medians.append(np.median(values))
                    axis.plot(
                        range(iterations), medians, linestyle,
                        label=f"{entropy_strategy}, N={sample_count}",
                    )
            axis.set_title(curvature_strategy.replace("_", " "))
            axis.set_xlabel("iteration")
            axis.set_ylabel(metric.replace("_", " "))
            if logarithmic:
                axis.set_yscale("log")
            axis.grid(alpha=0.25)
        axes[-1].legend(fontsize="small")
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-counts", nargs="+", type=int, default=[16, 64, 256])
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=350)
    parser.add_argument("--output-dir", type=Path, default=Path("work/score_entropy"))
    args = parser.parse_args()
    if min(args.sample_counts) < 2 or args.repetitions < 2 or args.seeds < 1:
        parser.error("sample counts/repetitions must be at least 2 and seeds at least 1")
    if args.iterations < 1 or not 0.0 <= args.beta < 1.0:
        parser.error("iterations must be positive and beta must be in [0, 1)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    direct_rows = direct_statistics(args.sample_counts, args.repetitions, args.seed)
    mfmc_rows = mfmc_optimum_statistics(
        args.sample_counts, args.repetitions, args.seed
    )
    history_rows = convergence_histories(
        args.sample_counts, range(args.seed, args.seed + args.seeds), args.iterations, args.beta
    )
    _write_csv(args.output_dir / "gradient_statistics.csv", direct_rows)
    _write_csv(args.output_dir / "mfmc_optimum_statistics.csv", mfmc_rows)
    _write_csv(args.output_dir / "convergence_history.csv", history_rows)
    _write_report(args.output_dir / "README.md", direct_rows, mfmc_rows, history_rows, args)
    _write_plots(args.output_dir, history_rows, args.sample_counts, args.iterations)


if __name__ == "__main__":
    main()
