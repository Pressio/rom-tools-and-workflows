"""Run a two-fidelity autoregressive GP benchmark on the steady CDR PDE.

Run from the repository root:
    python examples/multifidelity_cdr.py --smoke
    python examples/multifidelity_cdr.py --output-dir /tmp/cdr-mf

The benchmark uses the same solver at two spatial resolutions and predicts
the scalar integral of the PDE state. A fixed independent HF test set is used
only for final reporting, never for adaptive sampling or cross-validation.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

MODELS = Path(__file__).resolve().parent / "models"
sys.path.insert(0, str(MODELS))
from steady_cdr_model import SteadyCdrQoiModel

from romtools.rom.multifidelity import (
    AutoregressiveGPBuilder, AutoregressiveGPConfig, GPLevelConfig,
)
from romtools.rom.qoi_surrogates import GaussianProcessKernel, GaussianProcessQoiModel
from romtools.workflows.multifidelity import (
    FidelityLevel, MultiFidelityKFoldValidation, run_adaptive_multifidelity,
)
from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.sampling_methods import MonteCarloSampler


class IntegralCdrModel(SteadyCdrQoiModel):
    """QoI: integral of the steady CDR solution over the unit square."""

    def compute_qoi(self, run_directory, parameter_sample):
        with np.load(Path(run_directory) / "solution.npz") as data:
            solution = data["state"]
        return np.array([solution.sum() * self.system.dx * self.system.dy])


def reference_qois(model, parameter_names, samples, directory):
    directory.mkdir(parents=True, exist_ok=False)
    outputs = []
    for i, values in enumerate(samples):
        run_dir = directory / f"run_{i}"
        run_dir.mkdir()
        parameters = dict(zip(parameter_names, values))
        model.populate_run_directory(str(run_dir), parameters)
        if model.run_model(str(run_dir), parameters) != 0:
            raise RuntimeError(f"reference evaluation {i} failed")
        outputs.append(float(model.compute_qoi(str(run_dir), parameters)[0]))
    return np.array(outputs)


def run_benchmark(output_dir, smoke=False):
    names = ["bmag", "theta", "nu", "sigma"]
    parameter_space = UniformParameterSpace(
        names, [0.2, 0.3, 0.002, 0.4],
        [0.8, 1.2, 0.010, 1.5], MonteCarloSampler,
    )
    low = IntegralCdrModel(9 if smoke else 15, 9 if smoke else 15)
    high = IntegralCdrModel(21 if smoke else 39, 21 if smoke else 39)
    config = AutoregressiveGPConfig(
        levels=[
            GPLevelConfig(GaussianProcessKernel(length_scale=0.5),
                          noise_variance=1e-8),
            GPLevelConfig(GaussianProcessKernel(length_scale=0.5),
                          rho=1.0, noise_variance=1e-8),
        ],
        tune_hyperparameters=not smoke,
        optimizer_restarts=2,
    )
    result = run_adaptive_multifidelity(
        fidelities=[
            FidelityLevel(low, "coarse_CDR", cost=1),
            FidelityLevel(high, "fine_CDR", cost=10),
        ],
        surrogate_builder=AutoregressiveGPBuilder(config, parameter_names=names),
        parameter_space=parameter_space,
        absolute_work_dir=str(output_dir / "adaptive"),
        tolerance=0.0,
        initial_samples=[9, 4] if smoke else [16, 6],
        batch_sizes=[3, 1] if smoke else [8, 2],
        adaptive_sampling_strategy="random",
        pair_fidelity_samples=True,
        validation_strategy=MultiFidelityKFoldValidation(num_folds=2 if smoke else 3),
        max_iterations=2 if smoke else 3,
        random_seed=2026,
    )
    # This test set is independent of training and the K-fold partitions.
    test_samples = parameter_space.generate_samples(
        5 if smoke else 20, seed=93145,
    )
    truth = reference_qois(high, names, test_samples, output_dir / "reference")
    low_prediction = reference_qois(
        low, names, test_samples, output_dir / "reference_low"
    )
    mf_prediction = np.asarray(result.rom_model.predict(test_samples)).ravel()
    hf = result.training_data[-1]
    hf_only_model = GaussianProcessQoiModel(
        hf.parameters, hf.qois,
        kernel=GaussianProcessKernel(length_scale=0.5),
        noise_variance=1e-8,
        normalize_parameters=True,
    )
    hf_only_prediction = np.array([
        hf_only_model.compute_qoi("", dict(zip(names, x)))[0]
        for x in test_samples
    ])

    denom = max(np.linalg.norm(truth), 1e-12)
    metrics = {
        "low_fidelity_direct_relative_l2": float(np.linalg.norm(low_prediction - truth) / denom),
        "hf_only_gp_relative_l2": float(np.linalg.norm(hf_only_prediction - truth) / denom),
        "multifidelity_gp_relative_l2": float(np.linalg.norm(mf_prediction - truth) / denom),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "benchmark_results.npz",
        test_parameters=test_samples, reference_qois=truth,
        low_fidelity_qois=low_prediction,
        hf_only_gp_predictions=hf_only_prediction,
        multifidelity_gp_predictions=mf_prediction,
        kfold_errors=result.validation_errors,
        sample_counts=result.sample_counts,
        nominal_cost=result.total_cost,
        **metrics,
    )
    print("CDR multifidelity benchmark")
    print("sample counts (LF, HF):", result.sample_counts.tolist())
    print("highest-fidelity K-fold errors:", result.validation_errors.tolist())
    print("nominal total training cost:", result.total_cost)
    for name, value in metrics.items():
        print(f"{name}: {value:.6e}")
    print("Results:", output_dir / "benchmark_results.npz")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="small, fast comparison")
    parser.add_argument("--output-dir", type=Path, default=Path("cdr_mf_results"))
    args = parser.parse_args()
    run_benchmark(args.output_dir.resolve(), smoke=args.smoke)
