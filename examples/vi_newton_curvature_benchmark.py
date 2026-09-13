"""Diagnose stochastic gradient/curvature coupling at an analytic VI optimum.

Run, for example::

    python -m examples.vi_newton_curvature_benchmark --repetitions 5000

The target and variational distribution are the same one-dimensional Gaussian,
so the exact ELBO gradient is zero.  The reported Newton direction is therefore
entirely estimator error; independent and lagged curvature expose how much of
that error is caused by same-batch gradient/Hessian coupling.
"""

import argparse
import contextlib
import io

import numpy as np

from romtools.workflows.inverse.vi_drivers import (
    _average_hessians,
    _compute_reinforce_gradients,
    _compute_reinforce_hessian_full,
)
from romtools.workflows.inverse.vi_optimization_methods import NewtonSolver


def _estimate(rng, sample_count):
    samples = rng.standard_normal((sample_count, 1))
    log_joint = -0.5 * samples[:, 0] ** 2
    gradient_mean, gradient_log_std, _, _, _, _ = _compute_reinforce_gradients(
        samples,
        np.zeros(1),
        np.ones(1),
        log_joint,
        baseline_method="loo",
    )
    hessian = _compute_reinforce_hessian_full(
        samples,
        np.zeros(1),
        np.ones(1),
        log_joint,
        baseline_method="loo",
    )
    return np.concatenate([gradient_mean, gradient_log_std]), hessian


def run_benchmark(sample_counts, repetitions, beta, seed):
    solver = NewtonSolver(regularization=1e-2, hessian_type="full")

    def quiet_step(gradient, hessian):
        with contextlib.redirect_stdout(io.StringIO()):
            return solver.step(gradient, hessian)

    rows = []
    for sample_count in sample_counts:
        rng = np.random.default_rng(seed + sample_count)
        gradients = np.empty((repetitions, 2))
        directions = {
            "same_sample": np.empty((repetitions, 2)),
            "independent": np.empty((repetitions, 2)),
            "lagged": np.empty((repetitions, 2)),
        }
        running_hessian = None
        for repetition in range(repetitions):
            gradient, same_hessian = _estimate(rng, sample_count)
            _, independent_hessian = _estimate(rng, sample_count)
            gradients[repetition] = gradient
            directions["same_sample"][repetition] = quiet_step(
                gradient, same_hessian
            )
            directions["independent"][repetition] = quiet_step(
                gradient, independent_hessian
            )
            if running_hessian is None:
                running_hessian = same_hessian.copy()
            directions["lagged"][repetition] = quiet_step(
                gradient, running_hessian
            )
            running_hessian = _average_hessians(
                running_hessian, same_hessian, beta
            )

        gradient_mean = gradients.mean(axis=0)
        for strategy, values in directions.items():
            direction_mean = values.mean(axis=0)
            direction_se = values.std(axis=0, ddof=1) / np.sqrt(repetitions)
            rows.append((sample_count, strategy, gradient_mean, direction_mean, direction_se))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-counts", nargs="+", type=int, default=[16, 64, 256])
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=348)
    args = parser.parse_args()
    if args.repetitions < 2 or any(count < 2 for count in args.sample_counts):
        parser.error("sample counts and repetitions must be at least 2")
    if not 0.0 <= args.beta < 1.0:
        parser.error("beta must be in [0, 1)")

    print("samples,strategy,gradient_mean,direction_mean,direction_95ci")
    for sample_count, strategy, gradient, direction, standard_error in run_benchmark(
        args.sample_counts, args.repetitions, args.beta, args.seed
    ):
        gradient_text = ";".join(f"{value:.6e}" for value in gradient)
        direction_text = ";".join(f"{value:.6e}" for value in direction)
        confidence_text = ";".join(f"{1.96 * value:.6e}" for value in standard_error)
        print(
            f'{sample_count},{strategy},"{gradient_text}",'
            f'"{direction_text}","{confidence_text}"'
        )


if __name__ == "__main__":
    main()
