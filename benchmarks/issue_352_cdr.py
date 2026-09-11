"""Benchmark issue #352 ROM-only MF-EKI substeps on the CDR regression problem."""

import json
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import romtools.workflows
from romtools.workflows.parameter_spaces import MonteCarloSampler, UniformParameterSpace
from tests.romtools.workflows.regression.inverse.cdr_regression_fixture import (
    PARAMETER_MAXES,
    PARAMETER_MINS,
    PARAMETER_NAMES,
    TRUTH_PARAMETERS,
    build_mf_eki_kwargs,
)


VISCOSITY_INDEX = PARAMETER_NAMES.index("nu")
BENCHMARK_PARAMETER_MAXES = PARAMETER_MAXES.copy()
BENCHMARK_PARAMETER_MAXES[VISCOSITY_INDEX] = 1.0
ROM_EXTRA_ENSEMBLE_SIZE = 64


class CountingModel:
    def __init__(self, model):
        self.model = model
        self.run_count = 0

    def populate_run_directory(self, run_directory, parameter_sample):
        return self.model.populate_run_directory(run_directory, parameter_sample)

    def run_model(self, run_directory, parameter_sample):
        self.run_count += 1
        return self.model.run_model(run_directory, parameter_sample)

    def compute_qoi(self, run_directory, parameter_sample):
        return self.model.compute_qoi(run_directory, parameter_sample)


def _truth_parameter_vector():
    return np.array([TRUTH_PARAMETERS[name] for name in PARAMETER_NAMES], dtype=float)


def _build_benchmark_parameter_space():
    return UniformParameterSpace(
        parameter_names=PARAMETER_NAMES,
        lower_bounds=PARAMETER_MINS.copy(),
        upper_bounds=BENCHMARK_PARAMETER_MAXES.copy(),
        sampler=MonteCarloSampler,
    )


def _load_parameter_history(case_dir):
    truth = _truth_parameter_vector()
    history = []
    restart_files = sorted(
        case_dir.glob("iteration_*/restart.npz"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    for restart_path in restart_files:
        iteration = int(restart_path.parent.name.split("_")[-1])
        with np.load(restart_path, allow_pickle=True) as restart:
            parameter_samples = np.asarray(restart["parameter_samples_one"], dtype=float)
        parameter_mean = np.mean(parameter_samples, axis=0)
        relative_error = float(np.linalg.norm(parameter_mean - truth) / np.linalg.norm(truth))
        history.append(
            {
                "iteration": iteration,
                "parameter_mean": parameter_mean.tolist(),
                "relative_parameter_error": relative_error,
            }
        )
    return history


def run_case(root, name, num_substeps, start, end, max_iterations=50,
             error_norm_tolerance=0.0):
    case_dir = root / name
    kwargs = build_mf_eki_kwargs(str(case_dir))
    counting_model = CountingModel(kwargs["model"])
    kwargs["model"] = counting_model
    kwargs["parameter_space"] = _build_benchmark_parameter_space()
    kwargs["parameter_maxes"] = BENCHMARK_PARAMETER_MAXES.copy()
    kwargs["rom_extra_ensemble_size"] = ROM_EXTRA_ENSEMBLE_SIZE
    kwargs["max_iterations"] = max_iterations
    kwargs["initial_step_size"] = 0.25
    kwargs["error_norm_tolerance"] = error_norm_tolerance
    kwargs["rom_substep_start_iteration"] = start
    kwargs["rom_substep_end_iteration"] = end
    kwargs["num_rom_substeps"] = num_substeps
    kwargs["rom_type"] = "gp"
    kwargs["rom_args"] = dict(kwargs["rom_args"])
    kwargs["rom_args"]["normalize_parameters"] = True
    kwargs["rom_args"]["normalize_targets"] = True

    t0 = time.perf_counter()
    parameter_samples, qois = romtools.workflows.mf_eki_with_auto_rom(**kwargs)
    wall_time = time.perf_counter() - t0
    observations = kwargs["observations"]
    residual = float(np.mean(np.linalg.norm(observations[:, None] - qois, axis=0)))
    parameter_history = _load_parameter_history(case_dir)
    return {
        "name": name,
        "num_rom_substeps": num_substeps,
        "start": start,
        "end": end,
        "fom_evaluations": counting_model.run_count,
        "final_fom_residual": residual,
        "wall_time_seconds": wall_time,
        "parameter_mean": np.mean(parameter_samples, axis=0).tolist(),
        "parameter_history": parameter_history,
    }


def _plot_parameter_error(results, output):
    by_name = {case["name"]: case for case in results}
    selected = [
        ("baseline", "Baseline"),
        ("s1_w3_30", "1 ROM substep, [3,30)"),
        ("s2_w3_30", "2 ROM substeps, [3,30)"),
        ("s4_w3_30", "4 ROM substeps, [3,30)"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name, label in selected:
        history = by_name[name]["parameter_history"]
        iterations = [entry["iteration"] for entry in history]
        errors = [entry["relative_parameter_error"] for entry in history]
        ax.plot(iterations, errors, marker="o", linewidth=2, label=label)

    ax.set_xlabel("Outer MF-EKI iteration")
    ax.set_ylabel(r"Relative parameter error  $\|\bar{p}_k-p^*\|_2 / \|p^*\|_2$")
    ax.set_title("CDR auto-ROM benchmark: parameter convergence")
    ax.set_xticks(range(0, 51, 5))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    fixed_budget_cases = [
        ("baseline", 0, 0, None),
        ("s1_w3_30", 1, 3, 30),
        ("s2_w3_30", 2, 3, 30),
        ("s4_w3_30", 4, 3, 30),
    ]

    with tempfile.TemporaryDirectory(prefix="issue352-cdr-") as tmp:
        root = Path(tmp)
        fixed_budget = [
            run_case(root / "fixed", *case, max_iterations=50)
            for case in fixed_budget_cases
        ]

        baseline = fixed_budget[0]
        target = baseline["final_fom_residual"] * (1.0 + 1.0e-12)
        time_to_quality = [
            run_case(
                root / "target",
                f"{case[0]}_target",
                case[1],
                case[2],
                case[3],
                max_iterations=50,
                error_norm_tolerance=target,
            )
            for case in fixed_budget_cases
        ]

        results = {
            "benchmark": "romtools CDR MF-EKI auto-ROM",
            "truth_parameters": {
                name: TRUTH_PARAMETERS[name] for name in PARAMETER_NAMES
            },
            "parameter_error_definition": "||mean(p_k)-p_truth||_2 / ||p_truth||_2",
            "initial_parameter_bounds": {
                name: [float(PARAMETER_MINS[i]), float(BENCHMARK_PARAMETER_MAXES[i])]
                for i, name in enumerate(PARAMETER_NAMES)
            },
            "fom_ensemble_size": 8,
            "rom_extra_ensemble_size": ROM_EXTRA_ENSEMBLE_SIZE,
            "initial_step_size": 0.25,
            "gp": {
                "normalize_parameters": True,
                "normalize_targets": True,
            },
            "fixed_budget_max_iterations": 50,
            "baseline_quality_target": target,
            "fixed_budget": fixed_budget,
            "time_to_quality": time_to_quality,
        }

        _plot_parameter_error(
            fixed_budget,
            Path("benchmarks/results/issue_352_parameter_error_vs_iteration.png"),
        )

    output = Path("benchmarks/results/issue_352_cdr_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
