"""Compare the existing greedy driver and batch space filling on steady CDR.

Both use the same Galerkin snapshot builder, candidate set, initial two points,
and FOM budget. Accuracy is measured on a shared independent validation set;
internal stopping estimators are not compared. Greedy selects by relative FOM
residual; space filling uses Euclidean distance in user-scaled coordinates.
The two-argument greedy builder is an adapter around the same four-argument
builder used by space filling. No production workflow is modified.

Run from the checkout:
    python examples/space_filling_greedy_comparison.py --output-dir /tmp/cdr-comparison
"""

import argparse
import contextlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.models.steady_cdr_model import SteadyCdrQoiModel
from romtools.workflows.greedy.run_greedy import run_greedy
from romtools.workflows.space_filling import run_space_filling


NAMES = ("bmag", "theta", "nu", "sigma")
LOWER = np.array([0.2, 0.0, 0.005, 0.1])
UPPER = np.array([2.0, np.pi / 2, 0.08, 1.5])


def scaled_distance(a, b):
    return np.linalg.norm((a - b) / (UPPER - LOWER))


class CandidateSpace:
    """Sample from a common finite candidate set for controlled comparisons.

    The greedy driver requests one replacement point after each enrichment.
    Cycling through this same set keeps its candidates in the common pool;
    already-trained points have negligible residual and need no new FOM solve.
    """

    def __init__(self, points):
        self.points = np.asarray(points)
        self.offset = 0

    def get_names(self):
        return NAMES

    def get_dimensionality(self):
        return 4

    def generate_samples(self, count, seed=None):
        indices = (np.arange(count) + self.offset) % len(self.points)
        self.offset = (self.offset + count) % len(self.points)
        return self.points[indices].copy()


def weights(parameters):
    p = np.atleast_2d(parameters)
    return np.c_[p[:, 2], p[:, 0] * np.cos(p[:, 1]),
                 p[:, 0] * np.sin(p[:, 1]), p[:, 3]]


class ProjectionRom:
    def __init__(self, system, basis, count, budget):
        self.count, self.budget = count, budget
        self.basis = basis
        self.operator_basis = np.stack([
            system.A_diffusion @ basis, -system.A_advection_x @ basis,
            -system.A_advection_y @ basis, -basis,
        ])
        self.reduced_operators = np.einsum("ni,cnj->cij", basis, self.operator_basis)
        self.rhs = -system.g
        self.reduced_rhs = basis.T @ self.rhs
        # Match SteadyCdrQoiModel's boundary-derivative sensor definition.
        self.qoi_basis = np.zeros((system.Nx + 2, basis.shape[1]))
        self.qoi_basis[1:-1] = -basis.reshape(system.Nx, system.Ny, -1)[:, -1] / system.dx

    def coefficients(self, samples):
        lhs = np.einsum("nc,cij->nij", weights(samples), self.reduced_operators)
        rhs = np.broadcast_to(self.reduced_rhs, (len(lhs), len(self.reduced_rhs)))
        return np.linalg.solve(lhs, rhs[..., None])[..., 0]

    def predict(self, samples):
        return self.coefficients(samples) @ self.qoi_basis.T

    def populate_run_directory(self, run_directory, parameter_sample):
        pass

    def run_model(self, run_directory, parameter_sample):
        return 0

    def compute_qoi(self, run_directory, parameter_sample):
        return self.predict([[parameter_sample[name] for name in NAMES]])[0]

    def compute_error_estimate(self, run_directory, parameter_sample):
        # The historical greedy driver has no sample-limit option. A zero
        # indicator at the externally imposed budget terminates it normally.
        if self.count >= self.budget:
            return 0.0
        point = np.array([parameter_sample[name] for name in NAMES])
        coefficients = self.coefficients(point[None])[0]
        residual = np.einsum("c,cni,i->n", weights(point[None])[0],
                             self.operator_basis, coefficients) - self.rhs
        return float(np.linalg.norm(residual) / np.linalg.norm(self.rhs))


class ProjectionBuilder:
    """One snapshot-based builder with a directory adapter for greedy."""

    def __init__(self, system, validation_points, validation_qois, budget):
        self.system = system
        self.validation_points, self.validation_qois = validation_points, validation_qois
        self.budget = budget
        self.history = []
        self.first_points = None

    def build_from_training_dirs(self, offline, directories, parameters=None, qois=None):
        states, file_parameters = [], []
        for directory in directories:
            with np.load(Path(directory) / "solution.npz") as data:
                states.append(data["state"])
                file_parameters.append(data["params"])
        file_parameters = np.asarray(file_parameters)
        if parameters is not None:
            np.testing.assert_allclose(parameters, file_parameters)
        if self.first_points is None:
            self.first_points = file_parameters.copy()
        u, singular, _ = np.linalg.svd(np.column_stack(states), full_matrices=False)
        basis = u[:, singular > singular[0] * 1e-12]
        rom = ProjectionRom(self.system, basis, len(directories), self.budget)
        prediction = rom.predict(self.validation_points)
        relative_errors = np.linalg.norm(prediction - self.validation_qois, axis=1) / np.maximum(
            np.linalg.norm(self.validation_qois, axis=1), 1e-12,
        )
        self.history.append(dict(samples=len(directories), rank=basis.shape[1],
                                 maximum=float(relative_errors.max()),
                                 rms=float(np.sqrt(np.mean(relative_errors**2)))))
        np.savez(Path(offline) / "projection_rom.npz", basis=basis)
        return rom


def reference_qois(model, points):
    from examples.models.steady_cdr import solveFom
    result = []
    for p in points:
        velocity = p[0] * np.array([np.cos(p[1]), np.sin(p[1])])
        state = solveFom(model.system, velocity, p[2], p[3])
        grid = np.zeros((model.system.Nx + 2, model.system.Ny + 2))
        grid[1:-1, 1:-1] = state.reshape(model.system.Nx, model.system.Ny)
        result.append(-grid[:, -2] / model.system.dx)
    return np.asarray(result)


def compare(output_dir, seeds=(1, 2, 3), validation_size=500,
            candidate_size=250, budget=32, concurrency=3):
    if budget < 3 or candidate_size < budget or validation_size < 1 or concurrency < 1 or not seeds:
        raise ValueError("Require budget >= 3, candidate_size >= budget, positive validation size/concurrency, and seeds")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = SteadyCdrQoiModel(nx=15, ny=15)
    validation_points = LOWER + np.random.default_rng(991).uniform(size=(validation_size, 4)) * (UPPER - LOWER)
    validation_qois = reference_qois(model, validation_points)
    trials = []
    for seed in seeds:
        candidates = LOWER + np.random.default_rng(seed + 500).uniform(size=(candidate_size, 4)) * (UPPER - LOWER)
        root = output / f"seed_{seed}"
        root.mkdir()
        print(f"Seed {seed}: running batch space filling...", flush=True)
        sf_builder = ProjectionBuilder(model.system, validation_points, validation_qois, budget)
        run_space_filling(
            model, sf_builder, CandidateSpace(candidates), str(root / "space_filling"),
            tolerance=0.0, initial_sample_size=2, batch_size=5,
            candidate_pool_size=candidate_size, distance_metric=scaled_distance,
            max_concurrent_fom_runs=concurrency, random_seed=seed,
            validation_strategy=lambda context: sf_builder.history[-1]["maximum"],
            max_fom_samples=budget, max_iterations=budget,
        )
        # Greedy always starts with the first two samples. Reorder its common
        # candidate pool so the two algorithms have identical initial data.
        initial = sf_builder.first_points
        initial_ids = [int(np.flatnonzero(np.all(candidates == p, axis=1))[0]) for p in initial]
        remainder = [i for i in range(candidate_size) if i not in initial_ids]
        reordered = candidates[initial_ids + remainder]
        print(f"Seed {seed}: running residual greedy...", flush=True)
        greedy_builder = ProjectionBuilder(model.system, validation_points, validation_qois, budget)
        with open(root / "greedy_stdout.log", "w") as log, contextlib.redirect_stdout(log):
            run_greedy(model, greedy_builder, CandidateSpace(reordered), str(root / "greedy"),
                       tolerance=1e-30, testing_sample_size=candidate_size,
                       random_seed=seed, calibrated_error=False)
        np.testing.assert_array_equal(sf_builder.first_points, greedy_builder.first_points)
        assert sf_builder.history[-1]["samples"] == budget
        assert greedy_builder.history[-1]["samples"] == budget
        trials.append(dict(seed=seed, space_filling=sf_builder.history, greedy=greedy_builder.history))
        print(f"Seed {seed}: final maximum errors: greedy={greedy_builder.history[-1]['maximum']:.4g}, "
              f"space filling={sf_builder.history[-1]['maximum']:.4g}", flush=True)
    metadata = dict(
        problem="Steady convection-diffusion-reaction; 15x15 grid; Galerkin snapshot ROM",
        seeds=list(seeds), validation_size=validation_size, validation_seed=991,
        candidate_size=candidate_size, training_budget=budget, initial_samples=2,
        space_filling_batch_size=5, max_concurrent_fom_runs=concurrency,
        parameter_names=NAMES, lower=LOWER.tolist(), upper=UPPER.tolist(),
        distance="Euclidean after explicit user scaling by parameter ranges",
        greedy_indicator="Relative full-order residual, uncalibrated",
        stopping="Fixed training budget; independent holdout used only for reporting",
        validation_fom_cost="Common validation reference solves excluded from training axis",
        trials=trials,
    )
    (output / "comparison.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot(metadata, output / "greedy_vs_space_filling.png")
    return metadata


def plot(data, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    colors = {"greedy": "#C45826", "space_filling": "#197B9B"}
    labels = {"greedy": "Residual greedy · one point per round",
              "space_filling": "Space filling · batches of five"}
    for ax, metric, heading in zip(axes, ("maximum", "rms"),
                                   ("Worst validation-point error", "RMS validation-point error")):
        for method in ("greedy", "space_filling"):
            history = data["trials"][0][method]
            counts = [h["samples"] for h in history]
            values = np.array([[h[metric] for h in trial[method]] for trial in data["trials"]])
            ax.fill_between(counts, values.min(axis=0), values.max(axis=0),
                            color=colors[method], alpha=0.14, linewidth=0)
            ax.semilogy(counts, np.median(values, axis=0), color=colors[method],
                        linewidth=2.2, marker="o" if method == "space_filling" else None,
                        markersize=5, label=labels[method])
        ax.set_title(heading, fontsize=12, fontweight="bold", loc="left", pad=12)
        ax.set_xlabel("Training FOM simulations")
        ax.set_ylabel("Relative L2 QoI error")
        ax.set_xlim(2, data["training_budget"])
        ax.grid(True, which="major", color="#DFE5EA", linewidth=0.8)
        ax.grid(True, which="minor", color="#EDF0F3", linewidth=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower left", fontsize=9, frameon=True)
    fig.suptitle("Greedy vs batch space filling", fontsize=19, fontweight="bold", x=0.045, ha="left")
    fig.supxlabel(
        f"Same CDR model, ROM builder and initial points · {data['validation_size']} independent validation points\n"
        f"Lines: median of {len(data['seeds'])} seeds · Shading: seed range · Sample efficiency, not wall time",
        fontsize=10, color="#4B5563",
    )
    fig.savefig(path, dpi=180, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--candidate-size", type=int, default=250)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    compare(args.output_dir, tuple(args.seeds), args.validation_size,
            args.candidate_size, args.budget, args.concurrency)
