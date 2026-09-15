"""Shared analytic sine inverse problem and plotting utilities for VI examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


DIMENSION = 7
N_OBSERVATIONS = 31
OBSERVATION_NOISE_STD = 5.0e-3
PRIOR_STD = 0.75
RANDOM_SEED = 7
TRUTH = np.array([1.00, -0.80, 0.65, -0.50, 0.35, -0.25, 0.15], dtype=float)


@dataclass(frozen=True)
class AnalyticSineProblem:
    """Linear-Gaussian sine-series inverse problem with an exact posterior."""

    parameter_names: tuple[str, ...]
    observation_locations: np.ndarray
    observation_matrix: np.ndarray
    truth: np.ndarray
    observations: np.ndarray
    observation_covariance: np.ndarray
    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray


def _observation_locations(equispaced: bool) -> np.ndarray:
    indices = np.arange(1, N_OBSERVATIONS + 1, dtype=float)
    if equispaced:
        return indices / (N_OBSERVATIONS + 1)

    # Keep the same number of observations and nearly the same spatial extent,
    # but cluster points toward x=0. This breaks the discrete sine
    # orthogonality that makes the equispaced posterior diagonal.
    scaled = indices / (N_OBSERVATIONS + 1)
    warped = scaled**3 / scaled[-1]**3
    return 0.02 + 0.96 * warped


def build_problem(equispaced: bool) -> AnalyticSineProblem:
    """Build the sine-series inverse problem and its exact Gaussian posterior."""
    parameter_names = tuple(f"theta_{j}" for j in range(1, DIMENSION + 1))
    x = _observation_locations(equispaced)
    modes = np.arange(1, DIMENSION + 1, dtype=float)
    observation_matrix = -np.sin(np.pi * np.outer(x, modes)) / (
        np.pi * modes[None, :]
    ) ** 2

    prior_mean = np.zeros(DIMENSION, dtype=float)
    prior_covariance = np.eye(DIMENSION) * PRIOR_STD**2
    observation_covariance = np.eye(N_OBSERVATIONS) * OBSERVATION_NOISE_STD**2

    rng = np.random.default_rng(RANDOM_SEED)
    observations = (
        observation_matrix @ TRUTH
        + OBSERVATION_NOISE_STD * rng.standard_normal(N_OBSERVATIONS)
    )

    prior_precision = np.linalg.inv(prior_covariance)
    observation_precision = np.linalg.inv(observation_covariance)
    posterior_precision = (
        prior_precision
        + observation_matrix.T @ observation_precision @ observation_matrix
    )
    posterior_covariance = np.linalg.inv(posterior_precision)
    posterior_mean = posterior_covariance @ (
        prior_precision @ prior_mean
        + observation_matrix.T @ observation_precision @ observations
    )

    off_diagonal = posterior_covariance - np.diag(np.diag(posterior_covariance))
    if equispaced:
        scale = max(float(np.max(np.diag(posterior_covariance))), 1.0e-16)
        if np.max(np.abs(off_diagonal)) / scale > 1.0e-10:
            raise RuntimeError("Expected the equispaced posterior covariance to be diagonal.")
    else:
        correlation = covariance_to_correlation(posterior_covariance)
        correlation_off_diagonal = correlation - np.eye(DIMENSION)
        if np.max(np.abs(correlation_off_diagonal)) < 0.1:
            raise RuntimeError("Non-equispaced observations did not create meaningful correlation.")

    return AnalyticSineProblem(
        parameter_names=parameter_names,
        observation_locations=x,
        observation_matrix=observation_matrix,
        truth=TRUTH.copy(),
        observations=observations,
        observation_covariance=observation_covariance,
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
    )


class AnalyticSineQoiModel:
    """ROMTools QoI model evaluating the exact sine-series solution."""

    def __init__(self, problem: AnalyticSineProblem):
        self.problem = problem

    def _theta(self, parameter_sample: dict) -> np.ndarray:
        return np.array(
            [float(parameter_sample[name]) for name in self.problem.parameter_names],
            dtype=float,
        )

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        del parameter_sample
        Path(run_directory).mkdir(parents=True, exist_ok=True)

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        qoi = self.problem.observation_matrix @ self._theta(parameter_sample)
        np.save(Path(run_directory) / "qoi.npy", qoi)
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        del parameter_sample
        return np.load(Path(run_directory) / "qoi.npy")


def gaussian_kl(
    mean: np.ndarray,
    covariance: np.ndarray,
    reference_mean: np.ndarray,
    reference_covariance: np.ndarray,
) -> float:
    """Return KL[N(mean,covariance) || N(reference_mean,reference_covariance)]."""
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    reference_mean = np.asarray(reference_mean, dtype=float)
    reference_covariance = np.asarray(reference_covariance, dtype=float)

    dimension = mean.size
    delta = reference_mean - mean
    sign_q, logdet_q = np.linalg.slogdet(covariance)
    sign_p, logdet_p = np.linalg.slogdet(reference_covariance)
    if sign_q <= 0 or sign_p <= 0:
        return float("inf")
    solve_covariance = np.linalg.solve(reference_covariance, covariance)
    solve_delta = np.linalg.solve(reference_covariance, delta)
    return float(
        0.5
        * (
            np.trace(solve_covariance)
            + delta @ solve_delta
            - dimension
            + logdet_p
            - logdet_q
        )
    )


def collect_history(work_dir: Path, problem: AnalyticSineProblem) -> dict[str, np.ndarray]:
    """Read the accepted VI states and prepend the common initial state."""
    with np.load(work_dir / "history.npz") as history:
        means = np.asarray(history["vi_history_variational_mean"], dtype=float)
        covariances = np.asarray(
            history["vi_history_variational_covariance"], dtype=float
        )

    if (
        means.shape[0] == 0
        or not np.allclose(means[0], problem.prior_mean)
        or not np.allclose(covariances[0], problem.prior_covariance)
    ):
        means = np.concatenate([problem.prior_mean[None, :], means], axis=0)
        covariances = np.concatenate(
            [problem.prior_covariance[None, :, :], covariances], axis=0
        )

    kl = np.asarray(
        [
            gaussian_kl(
                mean,
                covariance,
                problem.posterior_mean,
                problem.posterior_covariance,
            )
            for mean, covariance in zip(means, covariances)
        ],
        dtype=float,
    )
    return {"mean": means, "covariance": covariances, "kl": kl}


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(standard_deviations, standard_deviations)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance, dtype=float),
        where=denominator > 0.0,
    )
    np.fill_diagonal(correlation, 1.0)
    return correlation


def write_convergence_plot(
    output_path: Path,
    vi_history: dict[str, np.ndarray],
    mf_history: dict[str, np.ndarray],
    title: str,
) -> None:
    """Plot exact Gaussian KL convergence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    axis.semilogy(
        np.arange(vi_history["kl"].size),
        np.maximum(vi_history["kl"], 1.0e-14),
        marker="o",
        label="VI",
    )
    axis.semilogy(
        np.arange(mf_history["kl"].size),
        np.maximum(mf_history["kl"], 1.0e-14),
        marker="^",
        label="MF-VI",
    )
    axis.set(
        xlabel="Accepted variational state",
        ylabel=r"$D_{\mathrm{KL}}(q\,\|\,p_{\mathrm{post}})$",
        title=title,
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_mean_plot(
    output_path: Path,
    problem: AnalyticSineProblem,
    vi_history: dict[str, np.ndarray],
    mf_history: dict[str, np.ndarray],
    title: str,
) -> None:
    """Compare final posterior means against the exact posterior and truth."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.arange(1, DIMENSION + 1)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(indices, problem.truth, "k--", marker="x", label="Truth")
    axis.plot(indices, problem.posterior_mean, marker="o", label="Exact posterior mean")
    axis.plot(indices, vi_history["mean"][-1], marker="s", label="VI")
    axis.plot(indices, mf_history["mean"][-1], marker="^", label="MF-VI")
    axis.set(
        xlabel="Sine coefficient index",
        ylabel="Posterior mean",
        title=title,
        xticks=indices,
    )
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _normal_density(grid: np.ndarray, mean: float, variance: float) -> np.ndarray:
    variance = max(float(variance), 1.0e-16)
    return np.exp(-0.5 * (grid - mean) ** 2 / variance) / np.sqrt(
        2.0 * np.pi * variance
    )


def write_diagonal_posterior_plot(
    output_path: Path,
    problem: AnalyticSineProblem,
    vi_history: dict[str, np.ndarray],
    mf_history: dict[str, np.ndarray],
) -> None:
    """Plot one-dimensional marginals for the diagonal-posterior example."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(12.0, 6.2))
    final_states = (
        ("VI", vi_history["mean"][-1], vi_history["covariance"][-1]),
        ("MF-VI", mf_history["mean"][-1], mf_history["covariance"][-1]),
    )
    for index, axis in enumerate(axes.flat):
        if index >= DIMENSION:
            axis.axis("off")
            continue
        exact_mean = problem.posterior_mean[index]
        exact_variance = problem.posterior_covariance[index, index]
        width = 4.0 * np.sqrt(exact_variance)
        grid = np.linspace(exact_mean - width, exact_mean + width, 300)
        axis.plot(
            grid,
            _normal_density(grid, exact_mean, exact_variance),
            linewidth=2.0,
            label="Exact",
        )
        for label, mean, covariance in final_states:
            axis.plot(
                grid,
                _normal_density(grid, mean[index], covariance[index, index]),
                label=label,
            )
        axis.axvline(problem.truth[index], color="black", linestyle="--", linewidth=1.0)
        axis.set_title(problem.parameter_names[index])
        axis.grid(True, alpha=0.2)
    axes.flat[0].legend()
    figure.suptitle("Posterior marginal densities")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _add_covariance_ellipse(
    axis,
    mean: np.ndarray,
    covariance: np.ndarray,
    label: str,
    linestyle: str,
    linewidth: float,
) -> None:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    for scale in (1.0, 2.0):
        ellipse = Ellipse(
            xy=mean,
            width=2.0 * scale * np.sqrt(max(eigenvalues[0], 0.0)),
            height=2.0 * scale * np.sqrt(max(eigenvalues[1], 0.0)),
            angle=angle,
            fill=False,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label if scale == 1.0 else None,
        )
        axis.add_patch(ellipse)


def write_correlated_posterior_plot(
    output_path: Path,
    problem: AnalyticSineProblem,
    vi_history: dict[str, np.ndarray],
    mf_history: dict[str, np.ndarray],
) -> None:
    """Show posterior correlation and a two-dimensional Gaussian marginal."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exact_correlation = covariance_to_correlation(problem.posterior_covariance)
    off_diagonal = np.abs(exact_correlation - np.eye(DIMENSION))
    first, second = np.unravel_index(np.argmax(off_diagonal), off_diagonal.shape)
    pair = np.array([first, second], dtype=int)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    image = axes[0].imshow(exact_correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[0].set(
        title="Exact posterior correlation",
        xlabel="Parameter index",
        ylabel="Parameter index",
        xticks=np.arange(DIMENSION),
        yticks=np.arange(DIMENSION),
        xticklabels=np.arange(1, DIMENSION + 1),
        yticklabels=np.arange(1, DIMENSION + 1),
    )
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    exact_mean = problem.posterior_mean[pair]
    exact_covariance = problem.posterior_covariance[np.ix_(pair, pair)]
    _add_covariance_ellipse(
        axes[1], exact_mean, exact_covariance, "Exact", "-", 2.2
    )
    _add_covariance_ellipse(
        axes[1],
        vi_history["mean"][-1, pair],
        vi_history["covariance"][-1][np.ix_(pair, pair)],
        "VI",
        "--",
        1.8,
    )
    _add_covariance_ellipse(
        axes[1],
        mf_history["mean"][-1, pair],
        mf_history["covariance"][-1][np.ix_(pair, pair)],
        "MF-VI",
        ":",
        2.0,
    )
    axes[1].plot(
        problem.truth[first],
        problem.truth[second],
        "kx",
        markersize=8,
        label="Truth",
    )

    all_means = np.vstack(
        [
            exact_mean,
            vi_history["mean"][-1, pair],
            mf_history["mean"][-1, pair],
        ]
    )
    max_std = max(
        np.sqrt(np.max(np.linalg.eigvalsh(exact_covariance))),
        np.sqrt(
            np.max(
                np.linalg.eigvalsh(
                    vi_history["covariance"][-1][np.ix_(pair, pair)]
                )
            )
        ),
        np.sqrt(
            np.max(
                np.linalg.eigvalsh(
                    mf_history["covariance"][-1][np.ix_(pair, pair)]
                )
            )
        ),
    )
    lower = np.min(all_means, axis=0) - 3.0 * max_std
    upper = np.max(all_means, axis=0) + 3.0 * max_std
    axes[1].set_xlim(lower[0], upper[0])
    axes[1].set_ylim(lower[1], upper[1])
    axes[1].set(
        title=(
            f"Strongest correlated marginal "
            f"(rho={exact_correlation[first, second]:.2f})"
        ),
        xlabel=problem.parameter_names[first],
        ylabel=problem.parameter_names[second],
    )
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    figure.suptitle("Full-covariance posterior structure")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
