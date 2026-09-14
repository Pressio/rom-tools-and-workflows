"""YAML-driven VI benchmark for the romtools H2-air flame model."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import time
from typing import Optional

import numpy as np
import yaml

from benchmark_common import (
    jsonable,
    load_yaml,
    make_model,
    make_parameter_spaces,
    optimizer_and_line_search,
    prepare_observations,
    reuse_config,
    write_json,
)
from benchmark_results import (
    aggregate_runs,
    collect_history,
    truncate_to_budget,
    write_plots,
)
from romtools.workflows.inverse import mf_vi_with_auto_rom, run_vi


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "method_ablation.yaml"
SMOKE_CONFIG = Path(__file__).resolve().parent / "configs" / "smoke.yaml"


def _common_args(
    config,
    model,
    prior,
    initial,
    observations,
    covariance,
    mins,
    maxes,
    optimizer_name,
    entropy_strategy,
    seed,
    run_dir,
):
    optimizer, line_search = optimizer_and_line_search(config, optimizer_name)
    vi = config["vi"]
    args = {
        "model": model,
        "prior_parameter_space": prior,
        "initial_variational_parameter_space": initial,
        "observations": observations,
        "observations_covariance": covariance,
        "parameter_mins": mins,
        "parameter_maxes": maxes,
        "absolute_work_dir": str(run_dir),
        "optimizer_method": optimizer_name,
        "optimizer_config": optimizer,
        "baseline_method": vi.get("baseline_method", "loo"),
        "random_seed": int(seed),
        "sampling_method": vi.get("sampling_method", "mc"),
        "covariance_regularization": float(vi.get("covariance_regularization", 1e-8)),
        "restart_files_to_keep": int(
            config["benchmark"].get("restart_files_to_keep", 10)
        ),
        "bounded_parameter_handling": vi.get(
            "bounded_parameter_handling", "transform"
        ),
        "transform_interior_margin": float(
            vi.get("transform_interior_margin", 1e-8)
        ),
        "transform_map": vi.get("transform_map", "sigmoid"),
        "min_physical_variational_std_fraction": float(
            vi.get("min_physical_variational_std_fraction", 1e-8)
        ),
        "elbo_scaling_factor": vi.get("elbo_scaling_factor", "auto"),
        "score_function_entropy_strategy": entropy_strategy,
        "create_run_directories": True,
    }
    if optimizer_name == "newton":
        args["line_search_method"] = "stochastic_nonmonotone"
        args["line_search_config"] = line_search
    return args


def run_one_method(
    config,
    method_key,
    method,
    seed,
    observations,
    covariance,
    run_dir,
    sample_overrides=None,
):
    sample_overrides = sample_overrides or {}
    _, mins, maxes, truth, prior, initial = make_parameter_spaces(config)
    model = make_model(config)
    optimizer_name = method["optimizer"]
    entropy = method.get("entropy_strategy", "analytic")
    reuse_enabled = bool(method.get("sample_reuse", False))
    common = _common_args(
        config,
        model,
        prior,
        initial,
        observations,
        covariance,
        mins,
        maxes,
        optimizer_name,
        entropy,
        seed,
        run_dir,
    )
    reuse = reuse_config(config, reuse_enabled)
    concurrency = int(config["benchmark"].get("fom_evaluation_concurrency", 1))

    start = time.perf_counter()
    if method["fidelity"] == "single":
        run_vi(
            **common,
            sample_size=int(
                sample_overrides.get("fom_sample_size", config["vi"]["sample_size"])
            ),
            evaluation_concurrency=concurrency,
            sample_reuse_config=reuse,
        )
    elif method["fidelity"] == "mf":
        mf = config["multifidelity"]
        rom = mf.get("rom", {})
        mf_vi_with_auto_rom(
            **common,
            fom_sample_size=int(
                sample_overrides.get("fom_sample_size", mf["fom_sample_size"])
            ),
            rom_extra_sample_size=int(
                sample_overrides.get(
                    "rom_extra_sample_size", mf["rom_extra_sample_size"]
                )
            ),
            fom_evaluation_concurrency=concurrency,
            rom_evaluation_concurrency=int(
                config["benchmark"].get("rom_evaluation_concurrency", 1)
            ),
            rom_tolerance=float(mf.get("rom_tolerance", 0.005)),
            max_rom_training_history=int(mf.get("max_rom_training_history", 5)),
            rom_type="gp",
            rom_args={
                "pod_energy_fraction": float(rom.get("pod_energy_fraction", 0.9999)),
                "max_pod_modes": rom.get("max_pod_modes"),
                "noise_variance_fraction": float(
                    rom.get("noise_variance_fraction", 1e-6)
                ),
                "tune_hyperparameters": bool(
                    rom.get("tune_hyperparameters", False)
                ),
                "normalize_parameters": bool(
                    rom.get("normalize_parameters", True)
                ),
                "normalize_targets": bool(rom.get("normalize_targets", True)),
            },
            sample_reuse_config=reuse,
        )
    else:
        raise ValueError(f"Unsupported fidelity '{method['fidelity']}'.")
    wall_time = time.perf_counter() - start

    history = collect_history(run_dir, truth, reuse_enabled)
    history = truncate_to_budget(history, config["benchmark"].get("fom_budget"))
    final_mean = history["variational_mean"][-1]
    final_covariance = history["variational_covariance"][-1]
    return {
        "method": method_key,
        "label": method.get("label", method_key),
        "seed": int(seed),
        "fidelity": method["fidelity"],
        "optimizer": optimizer_name,
        "sample_reuse_enabled": reuse_enabled,
        "entropy_strategy": entropy,
        "sample_overrides": sample_overrides,
        "wall_time_seconds": float(wall_time),
        "history": history,
        "summary": {
            "iterations": int(len(history["elbo"])),
            "final_cumulative_fom_evaluations": int(
                history["cumulative_fom_evaluations"][-1]
            ),
            "final_parameter_relative_error": float(
                history["parameter_relative_error"][-1]
            ),
            "final_elbo": float(history["elbo"][-1]),
            "final_mean_relative_mse": float(history["mean_relative_mse"][-1]),
            "posterior_mean": final_mean,
            "posterior_covariance": final_covariance,
            "posterior_std": np.sqrt(
                np.maximum(np.diag(final_covariance), 0.0)
            ),
        },
    }


def _write_run(result, run_dir: Path, config: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(result)
    payload["resolved_config"] = config
    write_json(run_dir / "benchmark_result.json", payload)
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(jsonable(config), stream, sort_keys=False)


def _selected_methods(config: dict, requested: Optional[list[str]]):
    methods = list(config["methods"])
    if requested is None:
        return methods
    unknown = sorted(set(requested) - set(methods))
    if unknown:
        raise ValueError(f"Unknown methods requested: {unknown}")
    return [method for method in methods if method in requested]


def run_ablation(
    config,
    work_root,
    output_dir,
    observations,
    covariance,
    requested_methods,
):
    methods = _selected_methods(config, requested_methods)
    results = []
    for method_key in methods:
        method = config["methods"][method_key]
        for seed in config["benchmark"]["algorithm_seeds"]:
            run_dir = work_root / "ablation" / method_key / f"seed_{int(seed)}"
            shutil.rmtree(run_dir, ignore_errors=True)
            print(
                f"\n=== {method.get('label', method_key)} | seed={seed} ===",
                flush=True,
            )
            result = run_one_method(
                config,
                method_key,
                method,
                seed,
                observations,
                covariance,
                run_dir,
            )
            _write_run(result, run_dir, config)
            results.append(result)
    _, _, _, truth, _, _ = make_parameter_spaces(config)
    parameter_names = list(config["parameters"]["names"])
    aggregate = aggregate_runs(results, methods)
    write_plots(
        aggregate,
        output_dir,
        "ablation",
        parameter_names=parameter_names,
        truth=truth,
    )
    payload = {"runs": results, "aggregate": aggregate}
    write_json(output_dir / "h2_air_flame_vi_ablation.json", payload)
    return payload


def run_sweep(
    config,
    work_root,
    output_dir,
    observations,
    covariance,
    requested_methods,
):
    methods = _selected_methods(config, requested_methods)
    fom_sizes = [int(value) for value in config["sweep"]["fom_sample_sizes"]]
    rom_sizes = [
        int(value) for value in config["sweep"]["mf_rom_extra_sample_sizes"]
    ]
    results = []
    for method_key in methods:
        method = config["methods"][method_key]
        cases = (
            [(fom, None) for fom in fom_sizes]
            if method["fidelity"] == "single"
            else [(fom, rom) for fom in fom_sizes for rom in rom_sizes]
        )
        for fom_size, rom_size in cases:
            overrides = {"fom_sample_size": fom_size}
            case = f"fom_{fom_size}"
            if rom_size is not None:
                overrides["rom_extra_sample_size"] = rom_size
                case += f"_rom_{rom_size}"
            for seed in config["benchmark"]["algorithm_seeds"]:
                run_dir = (
                    work_root / "sweep" / method_key / case / f"seed_{int(seed)}"
                )
                shutil.rmtree(run_dir, ignore_errors=True)
                print(
                    f"\n=== sweep {method_key} | {case} | seed={seed} ===",
                    flush=True,
                )
                result = run_one_method(
                    config,
                    method_key,
                    method,
                    seed,
                    observations,
                    covariance,
                    run_dir,
                    overrides,
                )
                result["sweep_case"] = case
                _write_run(result, run_dir, config)
                results.append(result)
    payload = {"runs": results}
    write_json(output_dir / "h2_air_flame_vi_sample_size_sweep.json", payload)
    return payload


def run_benchmark(
    config_path: Path,
    mode=None,
    work_dir=None,
    output_dir=None,
    methods=None,
):
    config = load_yaml(config_path.resolve())
    mode = mode or config.get("benchmark", {}).get("mode", "ablation")
    if mode not in ("ablation", "sweep"):
        raise ValueError("mode must be 'ablation' or 'sweep'.")
    work_root = (
        work_dir.resolve()
        if work_dir
        else Path(__file__).resolve().parent / "work"
    )
    results_dir = (
        output_dir.resolve()
        if output_dir
        else Path(__file__).resolve().parent / "results"
    )
    work_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    names, _, _, truth, _, _ = make_parameter_spaces(config)
    observations, covariance, metadata = prepare_observations(
        config, work_root, names, truth
    )
    with (results_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(jsonable(config), stream, sort_keys=False)
    write_json(results_dir / "observation_metadata.json", metadata)

    if mode == "ablation":
        payload = run_ablation(
            config,
            work_root,
            results_dir,
            observations,
            covariance,
            methods,
        )
    else:
        payload = run_sweep(
            config,
            work_root,
            results_dir,
            observations,
            covariance,
            methods,
        )
    print(
        f"\nWrote H2-air flame VI benchmark results to {results_dir}",
        flush=True,
    )
    return payload


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--mode", choices=("ablation", "sweep"), default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    config_path = SMOKE_CONFIG if args.smoke else (args.config or DEFAULT_CONFIG)
    run_benchmark(
        config_path,
        args.mode,
        args.work_dir,
        args.output_dir,
        args.methods,
    )


if __name__ == "__main__":
    main()
