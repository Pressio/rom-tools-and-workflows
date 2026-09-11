"""Plotting helpers shared by the VI documentation demos."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def collect_vi_history(work_dir: Path):
    """Return ELBO, variational mean, and standard-deviation histories."""
    elbos, means, stds = [], [], []
    iteration = 0
    while True:
        restart = work_dir / f"iteration_{iteration}" / "restart.npz"
        if not restart.exists():
            break
        with np.load(restart, allow_pickle=True) as data:
            elbos.append(float(data["elbo"]))
            means.append(np.asarray(data["variational_mean"], dtype=float))
            # Restarts persist log standard deviations rather than a separate
            # ``variational_std`` array.
            stds.append(np.exp(np.asarray(data["variational_log_std"], dtype=float)))
        iteration += 1
    return np.asarray(elbos), np.asarray(means), np.asarray(stds)


def write_vi_plots(output_dir, prefix, names, truth, vi_history, mf_history):
    """Write ELBO and posterior-parameter convergence figures."""
    output_dir = Path(output_dir)
    vi_elbo, vi_means, vi_stds = vi_history
    mf_elbo, mf_means, mf_stds = mf_history
    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    axis.plot(vi_elbo, marker="o", label="VI (FOM)")
    axis.plot(mf_elbo, marker="^", label="MF-VI (FOM+GP auto-ROM)")
    axis.set(xlabel="Iteration", ylabel="ELBO", title="VI ELBO convergence")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"{prefix}_elbo_convergence.png", dpi=180)

    figure, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4.0), squeeze=False)
    for index, axis in enumerate(axes.flat):
        for means, stds, label, marker in ((vi_means, vi_stds, "VI (FOM)", "o"), (mf_means, mf_stds, "MF-VI (FOM+GP auto-ROM)", "^")):
            steps = np.arange(means.shape[0])
            axis.plot(steps, means[:, index], marker=marker, label=label)
            axis.fill_between(steps, means[:, index] - stds[:, index], means[:, index] + stds[:, index], alpha=0.2)
        axis.axhline(truth[index], color="black", linestyle="--", label="Truth")
        axis.set(title=names[index], xlabel="Iteration", ylabel="Parameter value")
        axis.grid(True, alpha=0.3)
    axes.flat[0].legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"{prefix}_parameter_convergence.png", dpi=180)
