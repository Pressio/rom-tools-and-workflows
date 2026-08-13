"""Create publication-ready figures from a completed hybrid MFUQ workflow.

The script reads only files written by :func:`run_hybrid_mfuq` and never
reruns a model or retrains a surrogate.  In particular, it uses
``visualization_data.npz`` for the optimized surrogate/validation results and
uses ``pilot_results.npz`` (when available) for the QoI-agreement diagnostic.

Examples
--------
Generate PDF and PNG figures without opening windows::

    python postprocess_hybrid_mfuq.py /path/to/work --no-show

Use a different output directory or only vector figures::

    python postprocess_hybrid_mfuq.py /path/to/work --output-dir figures --formats pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "fom": "#202020",
    "rom": "#0072B2",
    "prediction": "#0072B2",
    "validation": "#D55E00",
    "pilot": "#CC79A7",
}


def _as_1d(data: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    """Return an NPZ entry as a finite one-dimensional float array."""
    if key not in data:
        raise KeyError(f"Required dataset '{key}' is absent from visualization_data.npz.")
    values = np.asarray(data[key], dtype=float).squeeze()
    return np.atleast_1d(values)


def _grid(data: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    """Read legacy two-row grids and newer one-dimensional grids alike."""
    if key not in data:
        raise KeyError(f"Required dataset '{key}' is absent from visualization_data.npz.")
    values = np.asarray(data[key], dtype=float)
    if values.ndim == 1:
        return values
    return values[0]


def _model_names(n_aux: int) -> list[str]:
    return [f"Auxiliary {i + 1}" for i in range(n_aux)]


def _style() -> None:
    """Set a compact, color-blind-friendly style suitable for journal figures."""
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def _add_panel_labels(axes: Iterable[plt.Axes]) -> None:
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", axes):
        ax.text(
            -0.14,
            1.06,
            f"({label})",
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )


def _direct_curve_labels(
    ax: plt.Axes, curves: Sequence[tuple[np.ndarray, np.ndarray, str, object]]
) -> None:
    """Label curves at their right edge, avoiding a plot-obscuring legend."""
    ymin, ymax = ax.get_ylim()
    log_scale = ax.get_yscale() == "log"
    transform = np.log10 if log_scale else lambda values: values
    inverse = lambda values: 10.0 ** values if log_scale else values
    span = transform(ymax) - transform(ymin)
    endpoints = transform(np.array([y[-1] for _, y, _, _ in curves], dtype=float))
    order = np.argsort(endpoints)
    separation = span * min(0.09, 0.65 / max(len(curves), 1))
    positions = endpoints.copy()
    for previous, current in zip(order, order[1:]):
        positions[current] = max(positions[current], positions[previous] + separation)
    positions = np.clip(positions, transform(ymin) + 0.04 * span, transform(ymax) - 0.04 * span)
    for previous, current in zip(order[::-1], order[-2::-1]):
        positions[current] = min(positions[current], positions[previous] - separation)

    for index, (x, y, name, color) in enumerate(curves):
        ax.annotate(
            name,
            xy=(x[-1], y[-1]),
            xytext=(x[-1] - 0.025 * (x[-1] - x[0]), inverse(positions[index])),
            ha="right",
            va="center",
            color=color,
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7},
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )


def _plot_fidelity_and_cost(data: Mapping[str, np.ndarray], n_aux: int, work_directory: Path) -> plt.Figure:
    """Plot fitted agreement and normalized cost against ROM basis size."""
    basis = _grid(data, "ss")
    pilot_basis = _grid(data, "pp")
    selected_basis = int(round(float(_as_1d(data, "s_star")[-1])))

    surrogate_corr = float(np.interp(selected_basis, basis, _as_1d(data, "rho_fom_rom_vals")))
    surrogate_cost = float(np.interp(selected_basis, basis, _as_1d(data, "cost_rom_vals")))
    trained_corr, trained_cost = _trained_rom_statistics(work_directory, surrogate_corr, surrogate_cost)
    surrogate_aux_corrs = [
        float(np.interp(selected_basis, basis, _as_1d(data, f"rho_aux{i}_rom_vals")))
        for i in range(n_aux)
    ]
    trained_aux_corrs = _trained_rom_aux_correlations(work_directory, n_aux, surrogate_aux_corrs)

    fig, (ax_corr, ax_cost) = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    ax_corr.plot(
        basis,
        _as_1d(data, "rho_fom_rom_vals"),
        color=COLORS["rom"],
        linewidth=2.3,
    )
    ax_corr.scatter(
        pilot_basis,
        _as_1d(data, "fom_rom_corrs_pilot"),
        color=COLORS["rom"],
        edgecolor="white",
        linewidth=0.7,
        s=48,
        zorder=3,
    )
    corr_curves = [
        (basis, _as_1d(data, "rho_fom_rom_vals"), "FOM–ROM", COLORS["rom"])
    ]
    for i, name in enumerate(_model_names(n_aux)):
        color = plt.cm.Dark2(i % 8)
        rho = _as_1d(data, f"rho_aux{i}_rom_vals")
        ax_corr.plot(
            basis,
            rho,
            color=color,
            linewidth=1.8,
        )
        ax_corr.scatter(
            pilot_basis,
            _as_1d(data, f"rho_aux{i}_rom_pilot"),
            color=color,
            edgecolor="white",
            linewidth=0.6,
            s=34,
            zorder=3,
        )
        corr_curves.append((basis, rho, f"{name}–ROM", color))
        ax_corr.scatter(
            [selected_basis], [trained_aux_corrs[i]],
            marker="*", s=200, color=color,
            edgecolor="white", linewidth=0.8, zorder=5,
        )
        color2 = plt.cm.Dark2((i+3) % 8)
        rho = _as_1d(data, f"rho_fom_aux{i}_vals") * np.ones_like(basis)
        ax_corr.plot(
            basis,
            rho,
            color=color2,
            linewidth=1.8,
        )
        ax_corr.scatter(
            pilot_basis,
            _as_1d(data, f"rho_fom_aux{i}_pilot") * np.ones_like(pilot_basis),
            color=color2,
            edgecolor="white",
            linewidth=0.6,
            s=34,
            zorder=3,
        )
        corr_curves.append((basis, rho, f"FOM-{name}", color2))
        ax_corr.scatter(
            [selected_basis], [float(_as_1d(data, f"rho_fom_aux{i}_pilot")[0])],
            marker="*", s=200, color=color2,
            edgecolor="white", linewidth=0.8, zorder=5,
        )
    ax_corr.axvline(selected_basis, color="0.35", linestyle="--", linewidth=1.2)
    ax_corr.annotate(
        f"selected $s={selected_basis:.0f}$",
        (selected_basis, 0.03),
        xytext=(5, 5),
        textcoords="offset points",
        color="0.3",
    )
    ax_corr.scatter(
        [selected_basis], [trained_corr],
        marker="*", s=240, color=COLORS["rom"],
        edgecolor="white", linewidth=0.8, zorder=5,
    )
    ax_corr.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='0.35', markeredgecolor='white', markersize=5, linestyle='None'),
            plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='0.35', markeredgecolor='white', markersize=11, linestyle='None')
        ],
        labels=['pilot', 'trained ROM'],loc="lower right", frameon=True, framealpha=0.85,
        handletextpad=0.5, borderpad=0.4, fontsize=8.5,
    )
    ax_corr.set_title("Model agreement with the ROM")
    ax_corr.set(xlabel="ROM basis size", ylabel="Pearson correlation", ylim=(-0.05, 1.05))
    _direct_curve_labels(ax_corr, corr_curves)

    ax_cost.plot(
        basis,
        _as_1d(data, "cost_rom_vals"),
        color=COLORS["rom"],
        linewidth=2.3,
    )
    ax_cost.scatter(
        pilot_basis,
        _as_1d(data, "normalized_rom_times_pilot"),
        color=COLORS["rom"],
        edgecolor="white",
        linewidth=0.7,
        s=48,
        zorder=3,
    )
    cost_curves = [(basis, _as_1d(data, "cost_rom_vals"), "ROM", COLORS["rom"])]
    for i, name in enumerate(_model_names(n_aux)):
        color = plt.cm.Dark2(i % 8)
        costs = _as_1d(data, f"cost_aux{i}_vals")
        ax_cost.plot(basis, costs, color=color, linewidth=1.8)
        cost_curves.append((basis, costs, name, color))
    ax_cost.axvline(selected_basis, color="0.35", linestyle="--", linewidth=1.2)
    ax_cost.scatter(
        [selected_basis], [trained_cost],
        marker="*", s=240, color=COLORS["rom"],
        edgecolor="white", linewidth=0.8, zorder=5,
    )
    ax_cost.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='0.35', markeredgecolor='white', markersize=5, linestyle='None'),
            plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='0.35', markeredgecolor='white', markersize=11, linestyle='None')
        ],
        labels=['pilot', 'trained ROM'],loc="lower right", frameon=True, framealpha=0.85,
        handletextpad=0.5, borderpad=0.4, fontsize=8.5,
    )
    ax_cost.set_title("Relative evaluation cost")
    ax_cost.set(xlabel="ROM basis size", ylabel="Cost / FOM cost", ylim=(0, None))
    _direct_curve_labels(ax_cost, cost_curves)
    _add_panel_labels((ax_corr, ax_cost))
    return fig


def _plot_estimator_performance(data: Mapping[str, np.ndarray]) -> plt.Figure:
    """Compare surrogate-predicted and validated normalized estimator variance."""
    budget = _as_1d(data, "xx")
    methods = (
        ("ACV-MF", _as_1d(data, "fMFs"), _as_1d(data, "fMFs_ex"), "o"),
        ("ACV-IS", _as_1d(data, "fISs"), _as_1d(data, "fISs_ex"), "s"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True, constrained_layout=True)

    for ax, (name, predicted, validated, marker) in zip(axes, methods):
        ax.loglog(budget, 1.0 / budget, color=COLORS["fom"], linestyle=":", linewidth=1.3)
        ax.loglog(
            budget, predicted, color=COLORS["prediction"], marker=marker,
            linestyle="--", linewidth=1.9,
        )
        ax.loglog(
            budget, validated, color=COLORS["validation"], marker=marker,
            linestyle="-", linewidth=2.1,
        )
        ax.set_title(name)
        ax.set(xlabel="Computational budget (FOM equivalents)")
        _direct_curve_labels(
            ax,
            (
                (budget, predicted, "Surrogate", COLORS["prediction"]),
                (budget, validated, "Trained-ROM validation", COLORS["validation"]),
                (budget, 1.0 / budget, "MC", COLORS["fom"]),
            ),
        )
    axes[0].set_ylabel("Normalized estimator variance")
    _add_panel_labels(axes)
    return fig


def _allocation_array(data: Mapping[str, np.ndarray], key: str, n_budgets: int) -> np.ndarray:
    values = np.asarray(data[key], dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[0] != n_budgets:
        raise ValueError(f"'{key}' has {values.shape[0]} allocations; expected {n_budgets}.")
    return values


def _rom_validation_cost(work_directory: Path, fallback: float) -> float:
    """Read the measured ROM cost saved after training, if it is available."""
    trained_results = sorted(work_directory.glob("trained_*_sample_rom_results.npz"))
    if not trained_results:
        return fallback
    with np.load(trained_results[-1], allow_pickle=False) as trained_data:
        return float(np.asarray(trained_data["normalized_rom_time"]).item())


def _trained_rom_statistics(work_directory: Path, fallback_corr: float, fallback_cost: float) -> tuple[float, float]:
    """Read the measured trained-ROM correlation and cost, with safe surrogate fallbacks."""
    trained_results = sorted(work_directory.glob("trained_*_sample_rom_results.npz"))
    if not trained_results:
        return fallback_corr, fallback_cost
    with np.load(trained_results[-1], allow_pickle=False) as trained_data:
        return (
            float(np.asarray(trained_data["fom_rom_corr"]).item()),
            float(np.asarray(trained_data["normalized_rom_time"]).item()),
        )


def _trained_rom_aux_correlations(
    work_directory: Path, n_aux: int, fallback_aux_corrs: Sequence[float],
) -> list[float]:
    """Read the measured trained-ROM/auxiliary-model correlations, with safe surrogate fallbacks."""
    trained_results = sorted(work_directory.glob("trained_*_sample_rom_results.npz"))
    if not trained_results:
        return list(fallback_aux_corrs)
    with np.load(trained_results[-1], allow_pickle=False) as trained_data:
        return [
            float(np.asarray(trained_data[f"aux{i}_rom_corr"]).item())
            if f"aux{i}_rom_corr" in trained_data else fallback_aux_corrs[i]
            for i in range(n_aux)
        ]


def _load_trained_rom_qois(work_directory: Path, basis_size: int, n_samples: int) -> np.ndarray | None:
    """Recover cached trained-ROM QoIs, preserving their pilot-sample ordering."""
    rom_dir = work_directory / "pilot" / "rom_optimized" / f"basis_size_{basis_size}"
    qoi_paths = [rom_dir / f"run_{sample_index}" / "qoi.txt" for sample_index in range(n_samples)]
    if not all(path.is_file() for path in qoi_paths):
        return None
    return np.asarray([np.loadtxt(path, dtype=float) for path in qoi_paths], dtype=float).reshape(n_samples)


def _allocation_cost_components(
    data: Mapping[str, np.ndarray], allocation: np.ndarray, n_aux: int, *, training: bool,
    validated_rom_cost: float | None = None,
) -> tuple[np.ndarray, list[str], list[object]]:
    """Convert allocation ratios into the cost components in the MFMC constraint."""
    basis = _grid(data, "ss")
    n_fom = allocation[:, 0]
    aux_costs = [float(_as_1d(data, f"cost_aux{i}_vals")[0]) for i in range(n_aux)]
    rom_cost = np.interp(allocation[:, -1], basis, _as_1d(data, "cost_rom_vals"))
    if validated_rom_cost is not None:
        rom_cost = np.full_like(n_fom, validated_rom_cost)

    components = [n_fom]
    components.extend(n_fom * allocation[:, i + 1] * cost for i, cost in enumerate(aux_costs))
    components.append(n_fom * allocation[:, n_aux + 1] * rom_cost)
    labels = ["FOM"] + _model_names(n_aux) + ["ROM sampling"]
    colors: list[object] = [COLORS["fom"]] + [plt.cm.Dark2(i % 8) for i in range(n_aux)] + [COLORS["rom"]]
    if training:
        components.append(n_fom * allocation[:, -1])
        labels.append("ROM training")
        colors.append("#999999")
    return np.column_stack(components), labels, colors


def _stacked_cost_bars(
    ax: plt.Axes, budget: np.ndarray, components: np.ndarray, labels: Sequence[str], colors: Sequence[object],
    title: str,
) -> list[object]:
    """Plot one interpretable cost-allocation bar per budget."""
    bottoms = np.zeros(len(budget))
    handles = []
    bar_width = 0.72 * np.min(np.diff(budget)) if len(budget) > 1 else 0.65 * budget[0]
    for values, label, color in zip(components.T, labels, colors):
        bars = ax.bar(budget, values, width=bar_width, bottom=bottoms, color=color, edgecolor="white", linewidth=0.55)
        handles.append(bars[0])
        bottoms += values
    ax.plot(budget, budget, color="0.25", linestyle=":", linewidth=1.1)
    ax.set_title(title, pad=9)
    ax.set(xlabel="Budget (FOM equivalents)", ylabel="Allocated cost (FOM equivalents)")
    ax.set_xticks(budget)
    ax.set_xticklabels([f"{value:g}" for value in budget])
    ax.set_ylim(0, max(np.max(bottoms), np.max(budget)) * 1.10)
    return handles


def _plot_allocation_figure(
    data: Mapping[str, np.ndarray], n_aux: int, work_directory: Path,
    alloc_key: str, alloc_key_ex: str, method_label: str,
) -> plt.Figure:
    """Show one allocation type's sampling strategy as a cost-partition bar chart."""
    budget = _as_1d(data, "xx")
    predicted = _allocation_array(data, alloc_key, len(budget))
    validated = _allocation_array(data, alloc_key_ex, len(budget))
    if predicted.shape[1] != n_aux + 3 or validated.shape[1] != n_aux + 3:
        raise ValueError("Unexpected allocation length; expected N, one ratio per low-fidelity model, and s.")

    fallback_cost = float(np.interp(validated[0, -1], _grid(data, "ss"), _as_1d(data, "cost_rom_vals")))
    exact_cost = _rom_validation_cost(work_directory, fallback_cost)
    pred_components, labels, colors = _allocation_cost_components(data, predicted, n_aux, training=True)
    exact_components, _, _ = _allocation_cost_components(
        data, validated, n_aux, training=True, validated_rom_cost=exact_cost
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.73, wspace=0.26)
    _stacked_cost_bars(
        axes[0], budget, pred_components, labels, colors, f"Surrogate-optimized allocation ({method_label})"
    )
    exact_handles = _stacked_cost_bars(
        axes[1], budget, exact_components, labels + ["ROM training"], colors + ["#999999"],
        f"Allocation with the trained ROM ({method_label})",
    )
    fig.legend(
        exact_handles, labels + ["ROM training"], loc="upper center", bbox_to_anchor=(0.5, 0.97),
        ncol=min(len(exact_handles), 5), frameon=False, columnspacing=1.4, handlelength=1.2,
    )
    _add_panel_labels(axes)
    return fig


def _plot_is_allocations(data: Mapping[str, np.ndarray], n_aux: int, work_directory: Path) -> plt.Figure:
    """Show the ACV-IS sampling strategy as a cost-partition bar chart."""
    return _plot_allocation_figure(data, n_aux, work_directory, "fISs_alloc", "fISs_alloc_ex", "ACV-IS")


def _plot_mf_allocations(data: Mapping[str, np.ndarray], n_aux: int, work_directory: Path) -> plt.Figure:
    """Show the ACV-MF sampling strategy as a cost-partition bar chart."""
    return _plot_allocation_figure(data, n_aux, work_directory, "fMFs_alloc", "fMFs_alloc_ex", "ACV-MF")


def _plot_control_variate_tradeoff(
    data: Mapping[str, np.ndarray], n_aux: int, work_directory: Path,
) -> plt.Figure:
    """Show the cost-agreement tradeoff that drives useful control variates."""
    selected_basis = int(round(float(_as_1d(data, "s_star")[-1])))
    surrogate_corr = float(np.interp(selected_basis, _grid(data, "ss"), _as_1d(data, "rho_fom_rom_vals")))
    surrogate_cost = float(np.interp(selected_basis, _grid(data, "ss"), _as_1d(data, "cost_rom_vals")))
    rom_corr, rom_cost = _trained_rom_statistics(work_directory, surrogate_corr, surrogate_cost)

    points: list[tuple[str, float, float, object, str]] = [("FOM", 1.0, 1.0, COLORS["fom"], "D")]
    for i, name in enumerate(_model_names(n_aux)):
        points.append((
            name,
            float(_as_1d(data, f"cost_aux{i}_vals")[0]),
            abs(float(_as_1d(data, f"rho_fom_aux{i}_pilot")[0])),
            plt.cm.Dark2(i % 8),
            "o",
        ))
    points.append((f"Trained ROM ($s={selected_basis:.0f}$)", rom_cost, abs(rom_corr), COLORS["rom"], "s"))

    fig, ax = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)
    for name, cost, corr, color, marker in points:
        ax.scatter(cost, corr, color=color, marker=marker, s=74, edgecolor="white", linewidth=0.7, zorder=3)
        ax.annotate(
            name, (cost, corr), xytext=(6, 5), textcoords="offset points", color=color, fontsize=9,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
    ax.set_xscale("log")
    ax.set(xlabel="Relative evaluation cost (log scale)", ylabel=r"$|\rho(\mathrm{FOM},\ \mathrm{control\ variate})|$", ylim=(0, 1.05))
    ax.set_title("Control-variate cost–agreement tradeoff")
    _add_panel_labels((ax,))
    return fig


def _plot_pilot_qoi_agreement(
    pilot_data: Mapping[str, np.ndarray], n_aux: int, trained_rom_qois: np.ndarray | None = None,
) -> plt.Figure:
    """Create paired FOM-QoI comparisons for all cached control variates."""
    fom = np.asarray(pilot_data["fom_qois_master"], dtype=float).ravel()
    comparisons: list[tuple[str, np.ndarray, object]] = [
        (f"Auxiliary {i + 1}", np.asarray(pilot_data[f"aux{i}_qois_master"], dtype=float).ravel(), plt.cm.Dark2(i % 8))
        for i in range(n_aux)
    ]
    if trained_rom_qois is not None:
        comparisons.append(("Trained ROM", trained_rom_qois.ravel(), COLORS["rom"]))

    ncols = min(3, len(comparisons))
    nrows = int(np.ceil(len(comparisons) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.7 * nrows), squeeze=False, constrained_layout=True)
    axes_flat = axes.ravel()
    for i, ax in enumerate(axes_flat):
        if i >= len(comparisons):
            ax.set_visible(False)
            continue
        model_name, model_qoi, color = comparisons[i]
        finite = np.isfinite(fom) & np.isfinite(model_qoi)
        x, y = fom[finite], model_qoi[finite]
        corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
        ax.scatter(x, y, s=26, alpha=0.78, color=color, edgecolor="white", linewidth=0.35)
        if len(x) > 1 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="0.2", linewidth=1.2)
        ax.text(0.04, 0.96, f"$\\rho={corr:.3f}$\n$n={len(x)}$", transform=ax.transAxes, va="top",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9})
        ax.set_title(model_name)
        ax.set(xlabel="FOM QoI", ylabel=f"{model_name} QoI")
    _add_panel_labels(axes_flat[:len(comparisons)])
    return fig


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: Sequence[str]) -> None:
    for extension in formats:
        fig.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight")


def generate_figures(work_directory: Path, output_dir: Path, formats: Sequence[str], show: bool) -> list[Path]:
    """Generate all figures possible from a completed workflow directory."""
    vis_path = work_directory / "visualization_data.npz"
    if not vis_path.is_file():
        raise FileNotFoundError(
            f"Cannot find {vis_path}. Run the workflow through step 5 before post-processing."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    with np.load(vis_path, allow_pickle=False) as data:
        n_aux = int(np.asarray(data["n_aux"]).item())
        selected_basis = int(round(float(_as_1d(data, "s_star")[-1])))
        figures = {
            "fidelity_and_cost": _plot_fidelity_and_cost(data, n_aux, work_directory),
            "estimator_performance": _plot_estimator_performance(data),
            "is_allocation": _plot_is_allocations(data, n_aux, work_directory),
            "control_variate_tradeoff": _plot_control_variate_tradeoff(data, n_aux, work_directory),
        }

        if "fMFs_alloc" in data:
            figures["mf_allocation"] = _plot_mf_allocations(data, n_aux, work_directory)
        else:
            print(f"Skipping ACV-MF allocation figure: fMFs_alloc was not found in {vis_path.name}.")

    pilot_path = work_directory / "pilot_results.npz"
    if pilot_path.is_file():
        with np.load(pilot_path, allow_pickle=False) as pilot_data:
            trained_rom_qois = _load_trained_rom_qois(
                work_directory, selected_basis, len(np.asarray(pilot_data["fom_qois_master"]).ravel())
            )
            if trained_rom_qois is None:
                print("Skipping trained-ROM QoI scatter: cached optimized-ROM QoIs were not found.")
            figures["pilot_qoi_agreement"] = _plot_pilot_qoi_agreement(pilot_data, n_aux, trained_rom_qois)
    else:
        print(f"Skipping pilot QoI agreement: {pilot_path.name} was not found.")

    outputs = [output_dir / f"{stem}.{extension}" for stem in figures for extension in formats]
    for stem, fig in figures.items():
        _save_figure(fig, output_dir, stem, formats)
    if show:
        plt.show()
    plt.close("all")
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_directory", type=Path, help="Directory produced by run_hybrid_mfuq.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Figure destination (default: <work_directory>/publication_figures).",
    )
    parser.add_argument(
        "--formats", nargs="+", default=("pdf", "png"), choices=("pdf", "png", "svg"),
        help="One or more output formats (default: pdf png).",
    )
    parser.add_argument("--no-show", action="store_true", help="Generate figures without opening interactive windows.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    work_directory = args.work_directory.expanduser().resolve()
    output_dir = (args.output_dir or work_directory / "publication_figures").expanduser().resolve()
    outputs = generate_figures(work_directory, output_dir, args.formats, show=not args.no_show)
    print("Created:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
