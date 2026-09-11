"""Benchmark issue #352 ROM-only MF-EKI substeps on the CDR regression problem."""

import json
import tempfile
import time
from pathlib import Path

import numpy as np

import romtools.workflows
from tests.romtools.workflows.regression.inverse.cdr_regression_fixture import (
    build_mf_eki_kwargs,
)


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


def run_case(root, name, num_substeps, start, end, max_iterations=10,
             error_norm_tolerance=0.0):
    case_dir = root / name
    kwargs = build_mf_eki_kwargs(str(case_dir))
    counting_model = CountingModel(kwargs["model"])
    kwargs["model"] = counting_model
    kwargs["max_iterations"] = max_iterations
    kwargs["error_norm_tolerance"] = error_norm_tolerance
    kwargs["rom_substep_start_iteration"] = start
    kwargs["rom_substep_end_iteration"] = end
    kwargs["num_rom_substeps"] = num_substeps
    # Explicitly retain the requested auto-ROM GP normalization.
    kwargs["rom_type"] = "gp"
    kwargs["rom_args"] = dict(kwargs["rom_args"])
    kwargs["rom_args"]["normalize_parameters"] = True
    kwargs["rom_args"]["normalize_targets"] = True

    t0 = time.perf_counter()
    parameter_samples, qois = romtools.workflows.mf_eki_with_auto_rom(**kwargs)
    wall_time = time.perf_counter() - t0
    observations = kwargs["observations"]
    residual = float(np.mean(np.linalg.norm(observations[:, None] - qois, axis=0)))
    return {
        "name": name,
        "num_rom_substeps": num_substeps,
        "start": start,
        "end": end,
        "fom_evaluations": counting_model.run_count,
        "final_fom_residual": residual,
        "wall_time_seconds": wall_time,
        "parameter_mean": np.mean(parameter_samples, axis=0).tolist(),
    }


def main():
    fixed_budget_cases = [
        ("baseline", 0, 0, None),
        ("s1_w0_4", 1, 0, 4),
        ("s2_w0_4", 2, 0, 4),
        ("s4_w0_4", 4, 0, 4),
        ("s1_w0_6", 1, 0, 6),
        ("s2_w0_6", 2, 0, 6),
        ("s4_w0_6", 4, 0, 6),
        ("s1_w2_6", 1, 2, 6),
        ("s2_w2_6", 2, 2, 6),
        ("s4_w2_6", 4, 2, 6),
    ]

    with tempfile.TemporaryDirectory(prefix="issue352-cdr-") as tmp:
        root = Path(tmp)
        fixed_budget = [
            run_case(root / "fixed", *case, max_iterations=10)
            for case in fixed_budget_cases
        ]

        baseline = fixed_budget[0]
        # Use the baseline's 10-outer-iteration residual as a common quality target.
        target = baseline["final_fom_residual"] * (1.0 + 1.0e-12)
        time_to_quality = [
            run_case(
                root / "target",
                f"{case[0]}_target",
                case[1],
                case[2],
                case[3],
                max_iterations=20,
                error_norm_tolerance=target,
            )
            for case in fixed_budget_cases
        ]

        results = {
            "benchmark": "romtools CDR MF-EKI auto-ROM",
            "gp": {
                "normalize_parameters": True,
                "normalize_targets": True,
            },
            "fixed_budget_max_iterations": 10,
            "baseline_quality_target": target,
            "fixed_budget": fixed_budget,
            "time_to_quality": time_to_quality,
        }

    output = Path("/tmp/issue_352_cdr_benchmark.json")
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
