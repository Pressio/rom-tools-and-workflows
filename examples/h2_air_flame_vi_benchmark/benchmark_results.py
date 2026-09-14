"""Result collection and plotting for the H2-air flame VI benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_restart_value(data, key: str):
    if key not in data:
        return None
    value = np.asarray(data[key])
    return value.item() if value.shape == () else value.tolist()


def count_fom_work(method_dir: Path) -> np.ndarray:
    """Count real flame solves using their FOM-specific solution artifact."""
    cumulative = 0
    counts = []
    iteration = 0
    while True:
        iteration_dir = method_dir / f"iteration_{iteration}"
        if not iteration_dir.exists():
            break
        cumulative += sum(1 for _ in iteration_dir.rglob("solution.npz"))
        counts.append(cumulative)
        iteration += 1
    return np.asarray(counts, dtype=int)


def count_rom_work(method_dir: Path) -> np.ndarray:
    """Count low-fidelity model evaluations from their per-sample run directories."""
    cumulative = 0
    counts = []
    iteration = 0
    while True:
        iteration_dir = method_dir / f"iteration_{iteration}"
        if not iteration_dir.exists():
            break
        cumulative += sum(
            1
            for path in iteration_dir.rglob("run_rom*")
            if path.is_dir()
        )
        counts.append(cumulative)
        iteration += 1
    return np.asarray(counts, dtype=int)


def collect_history(
    method_dir: Path,
    truth: np.ndarray,
    reuse_enabled: bool,
    reuse_batch_size: int = None,
    reuse_history_batches: int = None,
) -> dict:
    history_path = method_dir / "history.npz"
    if not history_path.exists():
        raise RuntimeError(f"VI history not found: {history_path}")
    with np.load(history_path, allow_pickle=True) as history:
        means = np.asarray(history["vi_history_variational_mean"], dtype=float)
        covariance = np.asarray(
            history["vi_history_variational_covariance"], dtype=float
        )

    elbo, relative_mse, reuse = [], [], []
    iteration = 0
    while True:
        restart_path = method_dir / f"iteration_{iteration}" / "restart.npz"
        if not restart_path.exists():
            break
        with np.load(restart_path, allow_pickle=True) as data:
            elbo.append(float(data["elbo"]) if "elbo" in data else np.nan)
            relative_mse.append(
                float(data["mean_relative_mse"])
                if "mean_relative_mse" in data
                else np.nan
            )
            reuse.append(
                {
                    "used": _read_restart_value(data, "sample_reuse_used"),
                    "ess": _read_restart_value(data, "sample_reuse_ess"),
                    "archive_samples": _read_restart_value(
                        data, "sample_reuse_archive_samples"
                    ),
                    "archive_batches": _read_restart_value(
                        data, "sample_reuse_archive_batches"
                    ),
                    "refresh_reason": _read_restart_value(
                        data, "sample_reuse_refresh_reason"
                    ),
                }
            )
        iteration += 1

    fom_work = count_fom_work(method_dir)
    rom_work = count_rom_work(method_dir)
    count = min(len(means), len(elbo), len(fom_work), len(rom_work))
    if count == 0:
        raise RuntimeError(f"No completed VI iterations found in {method_dir}")
    means = means[:count]
    covariance = covariance[:count]
    fom_work = fom_work[:count]
    rom_work = rom_work[:count]
    elbo = np.asarray(elbo[:count], dtype=float)
    relative_mse = np.asarray(relative_mse[:count], dtype=float)
    reuse = reuse[:count]

    if reuse_enabled:
        previous_fom_work = 0
        refresh_count = 0
        for iteration, (diagnostic, cumulative_fom_work) in enumerate(
            zip(reuse, fom_work)
        ):
            new_fom_work = int(cumulative_fom_work) - previous_fom_work
            if diagnostic["used"] is None:
                used = new_fom_work == 0
                diagnostic["used"] = used
                if used:
                    diagnostic["refresh_reason"] = "reuse"
                elif iteration == 0:
                    diagnostic["refresh_reason"] = "empty"
                else:
                    diagnostic["refresh_reason"] = "inferred_refresh_new_fom"
            if not bool(diagnostic["used"]):
                refresh_count += 1
            if diagnostic["archive_batches"] is None and reuse_history_batches is not None:
                diagnostic["archive_batches"] = min(
                    refresh_count, int(reuse_history_batches)
                )
            if (
                diagnostic["archive_samples"] is None
                and diagnostic["archive_batches"] is not None
                and reuse_batch_size is not None
            ):
                diagnostic["archive_samples"] = (
                    int(diagnostic["archive_batches"]) * int(reuse_batch_size)
                )
            diagnostic["cumulative_fresh_fom_evaluations"] = int(cumulative_fom_work)
            diagnostic["ess_available_in_restart"] = diagnostic["ess"] is not None
            previous_fom_work = int(cumulative_fom_work)

    parameter_error = np.sqrt(
        np.mean(((means - truth[None, :]) / truth[None, :]) ** 2, axis=1)
    )
    std = np.sqrt(
        np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0.0)
    )
    return {
        "cumulative_fom_evaluations": fom_work,
        "cumulative_rom_evaluations": rom_work,
        "elbo": elbo,
        "mean_relative_mse": relative_mse,
        "parameter_relative_error": parameter_error,
        "variational_mean": means,
        "variational_covariance": covariance,
        "variational_std": std,
        "sample_reuse": reuse,
    }


def truncate_to_budget(history: dict, budget) -> dict:
    """Apply a matched-work analysis cap without interrupting a VI iteration."""
    if budget is None:
        return history
    work = np.asarray(history["cumulative_fom_evaluations"])
    keep = np.where(work <= int(budget))[0]
    count = int(keep[-1] + 1) if keep.size else 1
    output = {}
    for key, value in history.items():
        if isinstance(value, np.ndarray) and value.shape[0] == work.shape[0]:
            output[key] = value[:count]
        elif isinstance(value, list) and len(value) == work.shape[0]:
            output[key] = value[:count]
        else:
            output[key] = value
    return output


def _interpolate(history: dict, grid: np.ndarray, key: str):
    x = np.asarray(history["cumulative_fom_evaluations"], dtype=float)
    y = np.asarray(history[key], dtype=float)
    if x.size == 1:
        return np.full(grid.size, y[0])
    x, indices = np.unique(x, return_index=True)
    return np.interp(grid, x, y[indices], left=np.nan, right=y[indices][-1])


def _interpolate_parameter(history: dict, grid: np.ndarray, parameter_index: int):
    x = np.asarray(history["cumulative_fom_evaluations"], dtype=float)
    y = np.asarray(history["variational_mean"], dtype=float)[:, parameter_index]
    if x.size == 1:
        return np.full(grid.size, y[0])
    x, indices = np.unique(x, return_index=True)
    return np.interp(grid, x, y[indices], left=np.nan, right=y[indices][-1])


def aggregate_runs(results: list[dict], method_order: list[str]) -> dict:
    aggregate = {}
    for method in method_order:
        runs = [result for result in results if result["method"] == method]
        if not runs:
            continue
        min_work = min(
            int(run["history"]["cumulative_fom_evaluations"][0]) for run in runs
        )
        max_work = max(
            int(run["history"]["cumulative_fom_evaluations"][-1]) for run in runs
        )
        grid = np.arange(min_work, max_work + 1, dtype=float)
        entry = {
            "label": runs[0]["label"],
            "fom_grid": grid,
            "num_seeds": len(runs),
        }
        for key in ("parameter_relative_error", "elbo", "mean_relative_mse"):
            values = np.vstack(
                [_interpolate(run["history"], grid, key) for run in runs]
            )
            entry[key] = {
                "median": np.nanmedian(values, axis=0),
                "q25": np.nanquantile(values, 0.25, axis=0),
                "q75": np.nanquantile(values, 0.75, axis=0),
            }

        parameter_dimension = runs[0]["history"]["variational_mean"].shape[1]
        posterior_mean = []
        for parameter_index in range(parameter_dimension):
            values = np.vstack(
                [
                    _interpolate_parameter(
                        run["history"], grid, parameter_index
                    )
                    for run in runs
                ]
            )
            posterior_mean.append(
                {
                    "median": np.nanmedian(values, axis=0),
                    "q25": np.nanquantile(values, 0.25, axis=0),
                    "q75": np.nanquantile(values, 0.75, axis=0),
                }
            )
        entry["posterior_mean"] = posterior_mean
        wall_times = np.asarray(
            [run["wall_time_seconds"] for run in runs], dtype=float
        )
        entry["wall_time_seconds"] = {
            "median": float(np.median(wall_times)),
            "q25": float(np.quantile(wall_times, 0.25)),
            "q75": float(np.quantile(wall_times, 0.75)),
        }
        aggregate[method] = entry
    return aggregate


def write_plots(
    aggregate: dict,
    output_dir: Path,
    suffix: str = "ablation",
    parameter_names=None,
    truth=None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel in (
        ("parameter_relative_error", "RMS relative parameter error"),
        ("elbo", "ELBO"),
        ("mean_relative_mse", "Mean relative observation MSE"),
    ):
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        for entry in aggregate.values():
            x = np.asarray(entry["fom_grid"])
            stats = entry[metric]
            axis.plot(x, stats["median"], label=entry["label"])
            axis.fill_between(x, stats["q25"], stats["q75"], alpha=0.18)
        axis.set_xlabel("Cumulative H2-air FOM evaluations")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        if metric != "elbo":
            axis.set_yscale("log")
        axis.legend(fontsize="small")
        figure.tight_layout()
        figure.savefig(
            output_dir / f"h2_air_flame_vi_{suffix}_{metric}.png", dpi=180
        )
        figure.savefig(output_dir / f"h2_air_flame_vi_{suffix}_{metric}.svg")
        plt.close(figure)

    if parameter_names is not None and truth is not None:
        truth = np.asarray(truth, dtype=float)
        for parameter_index, parameter_name in enumerate(parameter_names):
            figure, axis = plt.subplots(figsize=(7.2, 4.6))
            for entry in aggregate.values():
                x = np.asarray(entry["fom_grid"])
                stats = entry["posterior_mean"][parameter_index]
                axis.plot(x, stats["median"], label=entry["label"])
                axis.fill_between(x, stats["q25"], stats["q75"], alpha=0.18)
            axis.axhline(
                truth[parameter_index],
                linestyle="--",
                linewidth=1.0,
                label="Truth",
            )
            axis.set_xlabel("Cumulative H2-air FOM evaluations")
            axis.set_ylabel(f"Posterior mean: {parameter_name}")
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize="small")
            figure.tight_layout()
            safe_name = parameter_name.replace("/", "_")
            figure.savefig(
                output_dir
                / f"h2_air_flame_vi_{suffix}_posterior_mean_{safe_name}.png",
                dpi=180,
            )
            figure.savefig(
                output_dir
                / f"h2_air_flame_vi_{suffix}_posterior_mean_{safe_name}.svg"
            )
            plt.close(figure)

    analytic_key = "mf_bbvi_newton_analytic"
    joint_key = "mf_bbvi_newton_joint"
    if analytic_key in aggregate and joint_key in aggregate:
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        for key in (analytic_key, joint_key):
            entry = aggregate[key]
            x = np.asarray(entry["fom_grid"])
            stats = entry["parameter_relative_error"]
            axis.plot(x, stats["median"], label=entry["label"])
            axis.fill_between(x, stats["q25"], stats["q75"], alpha=0.18)
        axis.set_xlabel("Cumulative H2-air FOM evaluations")
        axis.set_ylabel("RMS relative parameter error")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            output_dir / f"h2_air_flame_vi_{suffix}_mf_newton_entropy.png",
            dpi=180,
        )
        figure.savefig(
            output_dir / f"h2_air_flame_vi_{suffix}_mf_newton_entropy.svg"
        )
        plt.close(figure)


def write_sweep_plots(results: list[dict], output_dir: Path) -> None:
    """Plot final matched-work metrics against the FOM sample size."""
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for result in results:
        overrides = result.get("sample_overrides", {})
        fom_size = int(overrides["fom_sample_size"])
        rom_size = overrides.get("rom_extra_sample_size")
        key = (result["method"], None if rom_size is None else int(rom_size))
        groups.setdefault(key, {}).setdefault(fom_size, []).append(result)

    for summary_key, ylabel in (
        ("final_parameter_relative_error", "Final RMS relative parameter error"),
        ("final_elbo", "Final ELBO"),
        ("final_mean_relative_mse", "Final mean relative observation MSE"),
    ):
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        for (_, rom_size), by_fom_size in groups.items():
            first_runs = next(iter(by_fom_size.values()))
            label = first_runs[0]["label"]
            if rom_size is not None:
                label += f" (ROM extra={rom_size})"
            x = np.asarray(sorted(by_fom_size), dtype=float)
            medians, q25s, q75s = [], [], []
            for fom_size in x.astype(int):
                values = np.asarray(
                    [
                        run["summary"][summary_key]
                        for run in by_fom_size[int(fom_size)]
                    ],
                    dtype=float,
                )
                medians.append(np.median(values))
                q25s.append(np.quantile(values, 0.25))
                q75s.append(np.quantile(values, 0.75))
            medians = np.asarray(medians)
            q25s = np.asarray(q25s)
            q75s = np.asarray(q75s)
            axis.plot(x, medians, marker="o", label=label)
            axis.fill_between(x, q25s, q75s, alpha=0.18)
        axis.set_xlabel("FOM samples per VI iteration")
        axis.set_ylabel(ylabel)
        if summary_key != "final_elbo":
            axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")
        figure.tight_layout()
        figure.savefig(
            output_dir / f"h2_air_flame_vi_sweep_{summary_key}.png", dpi=180
        )
        figure.savefig(
            output_dir / f"h2_air_flame_vi_sweep_{summary_key}.svg"
        )
        plt.close(figure)
