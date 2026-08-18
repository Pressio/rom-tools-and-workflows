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


def _rom_key(data: Mapping[str, np.ndarray], prefix: str, rom_index: int, suffix: str = "") -> str:
    """Per-ROM key name (e.g. prefix='ss', rom_index=1 -> 'ss1'), falling
    back to the legacy unsuffixed name for ROM 0 so visualization_data.npz
    files written before multi-ROM support still work."""
    suffixed = f"{prefix}{rom_index}{suffix}"
    if suffixed in data or rom_index != 0:
        return suffixed
    legacy = f"{prefix}{suffix}"
    return legacy if legacy in data else suffixed


def _rom_label(rom_index: int, n_active: int) -> str:
    return f"ROM {rom_index + 1}" if n_active > 1 else "ROM"


_ROM_MARKERS = ["*", "P", "X", "D", "^", "v", "<", ">", "H", "8"]


def _palette(n_total: int) -> list:
    """n_total visually distinct colors, shared by every figure so a given
    model (auxiliary or ROM) always gets the same color and no two models
    are ever assigned the same one. Falls back from the 8-color Dark2
    palette to the larger tab20 palette once there are more than 8 models
    to color, since Dark2 would otherwise wrap around and repeat."""
    if n_total <= 8:
        return [plt.cm.Dark2(i) for i in range(max(n_total, 1))]
    return [plt.cm.tab20(i / max(n_total - 1, 1)) for i in range(n_total)]


def _aux_color(aux_index: int, n_aux: int, n_active: int) -> object:
    """One consistent color per auxiliary model, drawn from the same
    n_aux+n_active palette as ROM colors (see _rom_color) so no two
    models -- auxiliary or ROM -- ever share a color."""
    return _palette(n_aux + n_active)[aux_index]


def _rom_color(rom_index: int, n_active: int, n_aux: int) -> object:
    """One consistent color per trainable ROM. The single-ROM case keeps
    the original fixed blue (unchanged figures); with multiple ROMs,
    colors are drawn from the shared n_aux+n_active palette (see
    _aux_color) so ROM and auxiliary-model colors never collide."""
    if n_active == 1:
        return COLORS["rom"]
    return _palette(n_aux + n_active)[n_aux + rom_index]


def _rom_marker(rom_index: int) -> str:
    """Distinct marker shape per trainable ROM, layered on top of its
    color so a ROM's 'trained' points stay identifiable by shape alone --
    useful once curves overlap or a figure is viewed without color."""
    return _ROM_MARKERS[rom_index % len(_ROM_MARKERS)]


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


def _plot_fidelity_and_cost(
    data: Mapping[str, np.ndarray], n_aux: int, work_directory: Path, n_active: int = 1,
) -> plt.Figure:
    """Plot fitted agreement and normalized cost against ROM basis size.

    All trainable ROMs are drawn together, rather than in one figure per
    ROM, so ROMs can be compared directly.

    Layout: FOM-vs-everything correlation on top (one curve per ROM plus
    the fixed FOM-auxiliary references); cost on one tall axes alongside;
    and the auxiliary-ROM cross terms along the bottom, split into one
    small facet per ROM. The cross terms are faceted rather than sharing
    one axes because, within FOM-vs-everything or cost, every curve
    already has a distinct color (one per model, see _aux_color /
    _rom_color) so color alone identifies it -- but a single ROM
    contributes n_aux cross-term curves, and giving all of them that one
    ROM's color would make them visually identical within that axes.
    Faceting by ROM instead means each facet only ever needs n_aux colors,
    which are the auxiliary models' own dedicated colors and therefore
    already unique within that facet; the facet's title (colored to match
    that ROM's legend entry) supplies the ROM identity that would
    otherwise have come from color.

    Every correlation curve is additionally labeled at its own endpoint
    with the actual pair it represents (e.g. "FOM-ROM 2") since
    correlation is a property of a pair of models, not of one model alone.
    Curves naturally end at different basis sizes when ROMs have
    different tunable ranges; the end label sits exactly at that
    endpoint, so the cutoff is attributed rather than left unexplained.
    A ROM's "trained" points use that ROM's own marker shape (see
    _rom_marker) with the color of whichever curve they sit on, so the
    marker shape -- not color -- is what ties a point back to its ROM,
    consistent with the legend.
    """
    s_star = _as_1d(data, "s_star")
    all_pilot_basis = np.unique(np.concatenate(
        [_grid(data, _rom_key(data, "pp", t)) for t in range(n_active)]
    ))
    # Flat (basis-size-independent) auxiliary reference lines are labeled
    # at the right-hand edge of the widest ROM's range, so their labels
    # cluster at the true right edge of the plot rather than at whichever
    # ROM happens to be narrowest.
    x_edge = max(_grid(data, _rom_key(data, "ss", t)).max() for t in range(n_active))
    x_ref = np.array([1.0, x_edge])

    fig = plt.figure(figsize=(13.2, 8.6) if n_aux > 0 else (12.2, 4.8))
    if n_aux > 0:
        gs = fig.add_gridspec(
            2, 2, height_ratios=[1, 1], width_ratios=[1.1, 1],
            hspace=0.42, wspace=0.26, left=0.065, right=0.99, top=0.88, bottom=0.08,
        )
        ax_corr_top = fig.add_subplot(gs[0, 0])
        bottom_gs = gs[1, 0].subgridspec(1, n_active, wspace=0.1)
        cross_axes = [fig.add_subplot(bottom_gs[0, 0])]
        for t in range(1, n_active):
            cross_axes.append(fig.add_subplot(bottom_gs[0, t], sharey=cross_axes[0]))
        ax_cost = fig.add_subplot(gs[:, 1])
    else:
        # No auxiliary models means no aux-ROM cross terms to plot, so the
        # cross-term facets would just be empty -- skip them.
        gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.28, left=0.07, right=0.99, top=0.85, bottom=0.13)
        ax_corr_top = fig.add_subplot(gs[0, 0])
        cross_axes = []
        ax_cost = fig.add_subplot(gs[0, 1])

    top_curves = []
    cross_curves = [[] for _ in range(n_active)]
    cost_curves = []
    marker_handles = [
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='0.4',
                   markeredgecolor='white', markersize=6, linestyle='None'),
    ]
    marker_labels = ["pilot data"]

    # Auxiliary models: one FOM-aux flat reference line per model (their
    # correlation/cost don't depend on any ROM's basis size), each in that
    # model's own dedicated color -- also the color used for that model
    # everywhere it appears in the cross-term facets below.
    for i, name in enumerate(_model_names(n_aux)):
        color_aux = _aux_color(i, n_aux, n_active)

        rho_fom_aux = float(_as_1d(data, f"rho_fom_aux{i}_vals")[0])
        ax_corr_top.axhline(rho_fom_aux, color=color_aux, linewidth=1.6, zorder=2)
        ax_corr_top.scatter(
            all_pilot_basis, np.full_like(all_pilot_basis, rho_fom_aux),
            color=color_aux, edgecolor="white", linewidth=0.5, s=26, zorder=3,
        )
        top_curves.append((x_ref, np.full_like(x_ref, rho_fom_aux), f"FOM\u2013{name}", color_aux))

        cost_aux = float(_as_1d(data, f"cost_aux{i}_vals")[0])
        ax_cost.axhline(cost_aux, color=color_aux, linewidth=1.6, zorder=2)
        cost_curves.append((x_ref, np.full_like(x_ref, cost_aux), name, color_aux))

        marker_handles.append(plt.Line2D([0], [0], color=color_aux, linewidth=2.2))
        marker_labels.append(name)

    # Trainable ROMs: FOM-ROM_t (top correlation axes, ROM's own color)
    # and aux_i-ROM_t (that ROM's cross-term facet, auxiliary model's
    # color); each curve is individually named via its end label.
    for t in range(n_active):
        basis = _grid(data, _rom_key(data, "ss", t))
        pilot_basis = _grid(data, _rom_key(data, "pp", t))
        selected_basis = int(round(float(s_star[-(n_active - t)])))
        label = _rom_label(t, n_active)
        color = _rom_color(t, n_active, n_aux)
        marker = _rom_marker(t)

        rho_fom_rom = _as_1d(data, _rom_key(data, "rho_fom_rom", t, "_vals"))
        cost_rom = _as_1d(data, _rom_key(data, "cost_rom", t, "_vals"))
        surrogate_corr = float(np.interp(selected_basis, basis, rho_fom_rom))
        surrogate_cost = float(np.interp(selected_basis, basis, cost_rom))
        trained_corr, trained_cost = _trained_rom_statistics(work_directory, t, surrogate_corr, surrogate_cost)
        surrogate_aux_corrs = [
            float(np.interp(selected_basis, basis, _as_1d(data, _rom_key(data, f"rho_aux{i}_rom", t, "_vals"))))
            for i in range(n_aux)
        ]
        trained_aux_corrs = _trained_rom_aux_correlations(work_directory, t, n_aux, surrogate_aux_corrs)

        # FOM-ROM_t (top axes).
        ax_corr_top.plot(basis, rho_fom_rom, color=color, linewidth=2.3, zorder=4)
        ax_corr_top.scatter(
            pilot_basis, _as_1d(data, _rom_key(data, "fom_rom_corrs_pilot", t)),
            color=color, edgecolor="white", linewidth=0.7, s=42, zorder=5,
        )
        ax_corr_top.scatter(
            [selected_basis], [trained_corr], marker=marker, s=230, color=color,
            edgecolor="white", linewidth=0.9, zorder=6,
        )
        ax_corr_top.axvline(selected_basis, color=color, linestyle=":", linewidth=1.1, alpha=0.5, zorder=1)
        top_curves.append((basis, rho_fom_rom, f"FOM\u2013{label}", color))

        # aux_i-ROM_t: this ROM's own cross-term facet, one curve per aux
        # model in that model's dedicated color (never ROM_t's color, so
        # it never collides with anything else drawn in this facet).
        if cross_axes:
            ax_cross = cross_axes[t]
            for i, name in enumerate(_model_names(n_aux)):
                color_aux = _aux_color(i, n_aux, n_active)
                rho = _as_1d(data, _rom_key(data, f"rho_aux{i}_rom", t, "_vals"))
                ax_cross.plot(basis, rho, color=color_aux, linewidth=1.8, zorder=4)
                ax_cross.scatter(
                    pilot_basis, _as_1d(data, _rom_key(data, f"rho_aux{i}_rom", t, "_pilot")),
                    color=color_aux, edgecolor="white", linewidth=0.6, s=32, zorder=5,
                )
                ax_cross.scatter(
                    [selected_basis], [trained_aux_corrs[i]], marker=marker, s=140, color=color_aux,
                    edgecolor="white", linewidth=0.7, zorder=6,
                )
                cross_curves[t].append((basis, rho, name, color_aux))
            ax_cross.axvline(selected_basis, color=color, linestyle=":", linewidth=1.1, alpha=0.5, zorder=1)
            ax_cross.set_title(f"{label}\nvs. auxiliaries", color=color, fontsize=9, linespacing=1.3)

        # Cost (a property of ROM_t alone, so just one curve; same color
        # as ROM_t's correlation curve above).
        ax_cost.plot(basis, cost_rom, color=color, linewidth=2.3, zorder=4)
        ax_cost.scatter(
            pilot_basis, _as_1d(data, _rom_key(data, "normalized_rom_times_pilot", t)),
            color=color, edgecolor="white", linewidth=0.7, s=42, zorder=5,
        )
        ax_cost.scatter(
            [selected_basis], [trained_cost], marker=marker, s=230, color=color,
            edgecolor="white", linewidth=0.9, zorder=6,
        )
        ax_cost.axvline(selected_basis, color=color, linestyle=":", linewidth=1.1, alpha=0.5, zorder=1)
        cost_curves.append((basis, cost_rom, label, color))

        marker_handles.append(
            plt.Line2D([0], [0], color=color, marker=marker, markersize=9,
                       markeredgecolor="white", linewidth=2.2)
        )
        marker_labels.append(f"{label} trained, $s={selected_basis:d}$")

    fig.legend(
        marker_handles, marker_labels, loc="upper center", bbox_to_anchor=(0.5, 0.985),
        ncol=min(len(marker_labels), 5), frameon=False, fontsize=8.3, handletextpad=0.5, columnspacing=1.3,
    )

    ax_corr_top.set_title("Model agreement vs. FOM")
    ax_corr_top.set(xlabel="ROM basis size", ylabel="Pearson correlation", ylim=(-0.05, 1.05))
    _direct_curve_labels(ax_corr_top, top_curves)

    axes_for_labels = [ax_corr_top]
    if cross_axes:
        for t, ax_cross in enumerate(cross_axes):
            ax_cross.set(xlabel="ROM basis size", ylim=(-0.05, 1.05))
            if t == 0:
                ax_cross.set_ylabel("Pearson correlation")
            else:
                ax_cross.tick_params(labelleft=False)
            _direct_curve_labels(ax_cross, cross_curves[t])
        axes_for_labels.extend(cross_axes)

    ax_cost.set_title("Relative evaluation cost")
    ax_cost.set(xlabel="ROM basis size", ylabel="Cost / FOM cost", ylim=(0, None))
    _direct_curve_labels(ax_cost, cost_curves)
    axes_for_labels.append(ax_cost)

    _add_panel_labels(axes_for_labels)
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


def _rom_validation_cost(work_directory: Path, rom_index: int, fallback: float) -> float:
    """Read the measured ROM_t cost saved after training, if it is available."""
    trained_results = sorted(work_directory.glob("trained_*_sample_rom_results.npz"))
    if not trained_results:
        return fallback
    with np.load(trained_results[-1], allow_pickle=False) as trained_data:
        key = f"rom{rom_index}_normalized_time"
        if key not in trained_data and rom_index == 0:
            key = "normalized_rom_time"
        return float(np.asarray(trained_data[key]).item())


def _trained_rom_statistics(
    work_directory: Path, rom_index: int, fallback_corr: float, fallback_cost: float,
) -> tuple[float, float]:
    """Read the measured trained-ROM_t correlation and cost, with safe surrogate fallbacks."""
    trained_results = sorted(work_directory.glob("trained_*_sample_rom_results.npz"))
    if not trained_results:
        return fallback_corr, fallback_cost
    with np.load(trained_results[-1], allow_pickle=False) as trained_data:
        corr_key = f"rom{rom_index}_fom_corr"
        time_key = f"rom{rom_index}_normalized_time"
        if rom_index == 0:
            corr_key = corr_key if corr_key in trained_data else "fom_rom_corr"
            time_key = time_key if time_key in trained_data else "normalized_rom_time"
        return (
            float(np.asarray(trained_data[corr_key]).item()),
            float(np.asarray(trained_data[time_key]).item()),
        )


def _trained_rom_aux_correlations(
    work_directory: Path, rom_index: int, n_aux: int, fallback_aux_corrs: Sequence[float],
) -> list[float]:
    """Read the measured trained-ROM_t/auxiliary-model correlations, with safe surrogate fallbacks."""
    trained_results = sorted(work_directory.glob("trained_*_sample_rom_results.npz"))
    if not trained_results:
        return list(fallback_aux_corrs)
    with np.load(trained_results[-1], allow_pickle=False) as trained_data:
        out = []
        for i in range(n_aux):
            key = f"aux{i}_rom{rom_index}_corr"
            if key not in trained_data and rom_index == 0:
                key = f"aux{i}_rom_corr"
            out.append(float(np.asarray(trained_data[key]).item()) if key in trained_data else fallback_aux_corrs[i])
        return out


def _load_trained_rom_qois(
    work_directory: Path, rom_index: int, basis_size: int, n_samples: int,
) -> np.ndarray | None:
    """Recover cached trained-ROM_t QoIs, preserving their pilot-sample ordering."""
    subdir = f"basis_size_{basis_size}" if rom_index == 0 else f"rom{rom_index}_basis_size_{basis_size}"
    rom_dir = work_directory / "pilot" / "rom_optimized" / subdir
    qoi_paths = [rom_dir / f"run_{sample_index}" / "qoi.txt" for sample_index in range(n_samples)]
    if not all(path.is_file() for path in qoi_paths):
        return None
    return np.asarray([np.loadtxt(path, dtype=float) for path in qoi_paths], dtype=float).reshape(n_samples)


def _allocation_cost_components(
    data: Mapping[str, np.ndarray], allocation: np.ndarray, n_aux: int, n_active: int, *, training: bool,
    validated_rom_costs: Sequence[float] | None = None,
) -> tuple[np.ndarray, list[str], list[object]]:
    """Convert allocation ratios into the cost components in the MFMC constraint.

    Allocation columns are [N, r_1..r_{n_aux+n_active}, s_1..s_n_active]
    (see MFMC.set_objective_and_constraint): one oversampling ratio and,
    for trainable ROMs, one basis size per low-fidelity model.
    """
    n_fom = allocation[:, 0]
    aux_costs = [float(_as_1d(data, f"cost_aux{i}_vals")[0]) for i in range(n_aux)]

    rom_costs = []
    for t in range(n_active):
        if validated_rom_costs is not None:
            rom_costs.append(np.full_like(n_fom, validated_rom_costs[t]))
        else:
            basis_col = allocation[:, n_aux + n_active + 1 + t]
            basis_grid = _grid(data, _rom_key(data, "ss", t))
            cost_curve = _as_1d(data, _rom_key(data, "cost_rom", t, "_vals"))
            rom_costs.append(np.interp(basis_col, basis_grid, cost_curve))

    components = [n_fom]
    components.extend(n_fom * allocation[:, i + 1] * cost for i, cost in enumerate(aux_costs))
    components.extend(n_fom * allocation[:, n_aux + 1 + t] * rom_costs[t] for t in range(n_active))
    labels = ["FOM"] + _model_names(n_aux) + [f"{_rom_label(t, n_active)} sampling" for t in range(n_active)]
    colors: list[object] = (
        [COLORS["fom"]]
        + [_aux_color(i, n_aux, n_active) for i in range(n_aux)]
        + [_rom_color(t, n_active, n_aux) for t in range(n_active)]
    )
    if training:
        components.extend(allocation[:, n_aux + n_active + 1 + t] for t in range(n_active))
        labels.extend(f"{_rom_label(t, n_active)} training" for t in range(n_active))
        colors.extend("#999999" for _ in range(n_active))
    return np.column_stack(components), labels, colors


def _stacked_cost_bars(
    ax: plt.Axes, budget: np.ndarray, components: np.ndarray, labels: Sequence[str], colors: Sequence[object],
    title: str, highlight_index: int | None = None, highlight_label: str = "",
) -> list[object]:
    """Plot one interpretable cost-allocation bar per budget.

    highlight_index, if given, draws a translucent gray outline around
    that budget's stacked bar. Used on the surrogate-optimized panel to
    mark the one budget whose ROM basis size(s) were actually trained: the
    trained-ROM validation panel reuses that single trained ROM for every
    budget it shows, so only the highlighted budget's surrogate solution
    was directly carried into that validation -- the other bars each show
    their own, independently re-optimized basis size and are not what the
    validation panel actually validated.
    """
    bottoms = np.zeros(len(budget))
    handles = []
    diffs = np.diff(budget)
    positive_diffs = diffs[diffs > 0]
    min_spacing = positive_diffs.min() if positive_diffs.size else budget[0]
    bar_width = 0.72 * min_spacing if len(budget) > 1 else 0.65 * budget[0]
    for values, label, color in zip(components.T, labels, colors):
        bars = ax.bar(budget, values, width=bar_width, bottom=bottoms, color=color, edgecolor="white", linewidth=0.55)
        handles.append(bars[0])
        bottoms += values

    top_lim = max(np.max(bottoms), np.max(budget)) * 1.10
    if highlight_index is not None:
        top = bottoms[highlight_index]
        ax.add_patch(plt.Rectangle(
            (budget[highlight_index] - bar_width * 0.56, 0), bar_width * 1.12, top,
            fill=True, facecolor="0.5", alpha=0.16,
            edgecolor="0.35", linewidth=1.6, linestyle="--", zorder=0.5,
        ))
        ax.annotate(
            highlight_label or "trained here",
            (budget[highlight_index], top), xytext=(0, 8), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color="0.3",
        )
        top_lim = max(top_lim, top * 1.22)

    ax.set_title(title, pad=9)
    ax.set(xlabel="Budget (FOM equivalents)", ylabel="Allocated cost (FOM equivalents)")
    ax.set_xticks(budget)
    ax.set_xticklabels([f"{value:g}" for value in budget])
    ax.set_ylim(0, top_lim)
    return handles


def _validation_budget_index(
    data: Mapping[str, np.ndarray], predicted: np.ndarray, n_aux: int, n_active: int,
) -> int:
    """Identify which budget's surrogate-optimized ROM basis size(s) were
    actually trained and carried into the validation panel (see
    run_hybrid_mfuq Step 3's validation_budget_idx).

    Prefers the explicit 'validation_budget_idx' field written by newer
    runs; falls back to matching the recorded s_star against each
    budget's surrogate-optimized basis size(s), for older
    visualization_data.npz files that predate that field.
    """
    if "validation_budget_idx" in data:
        return int(np.asarray(data["validation_budget_idx"]).item())

    if n_active == 0:
        return len(predicted) - 1

    s_star_tail = np.round(_as_1d(data, "s_star")[-n_active:])
    basis_cols = predicted[:, n_aux + n_active + 1: n_aux + n_active + 1 + n_active]
    matches = np.all(np.round(basis_cols) == s_star_tail[None, :], axis=1)
    return int(np.argmax(matches)) if matches.any() else len(predicted) - 1


def _plot_allocation_figure(
    data: Mapping[str, np.ndarray], n_aux: int, work_directory: Path,
    alloc_key: str, alloc_key_ex: str, method_label: str, n_active: int = 1,
) -> plt.Figure:
    """Show one allocation type's sampling strategy as a cost-partition bar chart."""
    budget = _as_1d(data, "xx")
    predicted = _allocation_array(data, alloc_key, len(budget))
    validated = _allocation_array(data, alloc_key_ex, len(budget))
    expected_cols = n_aux + 2 * n_active + 1
    if predicted.shape[1] != expected_cols or validated.shape[1] != expected_cols:
        raise ValueError(
            f"Unexpected allocation length; expected N, one ratio per low-fidelity "
            f"model, and one basis size per trainable ROM ({expected_cols} columns)."
        )
    validation_idx = _validation_budget_index(data, predicted, n_aux, n_active)

    exact_costs = [
        _rom_validation_cost(
            work_directory, t,
            float(np.interp(
                validated[0, n_aux + n_active + 1 + t],
                _grid(data, _rom_key(data, "ss", t)),
                _as_1d(data, _rom_key(data, "cost_rom", t, "_vals")),
            )),
        )
        for t in range(n_active)
    ]
    pred_components, labels, colors = _allocation_cost_components(data, predicted, n_aux, n_active, training=True)
    exact_components, _, _ = _allocation_cost_components(
        data, validated, n_aux, n_active, training=True, validated_rom_costs=exact_costs
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.73, wspace=0.26)
    _stacked_cost_bars(
        axes[0], budget, pred_components, labels, colors, f"Surrogate-optimized allocation ({method_label})",
        highlight_index=validation_idx, highlight_label="ROM(s) trained\nat this budget",
    )
    exact_handles = _stacked_cost_bars(
        axes[1], budget, exact_components, labels, colors,
        f"Allocation with the trained ROM(s) ({method_label})",
    )
    fig.legend(
        exact_handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97),
        ncol=min(len(exact_handles), 5), frameon=False, columnspacing=1.4, handlelength=1.2,
    )
    _add_panel_labels(axes)
    return fig


def _plot_is_allocations(
    data: Mapping[str, np.ndarray], n_aux: int, work_directory: Path, n_active: int = 1,
) -> plt.Figure:
    """Show the ACV-IS sampling strategy as a cost-partition bar chart."""
    return _plot_allocation_figure(data, n_aux, work_directory, "fISs_alloc", "fISs_alloc_ex", "ACV-IS", n_active)


def _plot_mf_allocations(
    data: Mapping[str, np.ndarray], n_aux: int, work_directory: Path, n_active: int = 1,
) -> plt.Figure:
    """Show the ACV-MF sampling strategy as a cost-partition bar chart."""
    return _plot_allocation_figure(data, n_aux, work_directory, "fMFs_alloc", "fMFs_alloc_ex", "ACV-MF", n_active)


def _plot_control_variate_tradeoff(
    data: Mapping[str, np.ndarray], n_aux: int, work_directory: Path, n_active: int = 1,
) -> plt.Figure:
    """Show the cost-agreement tradeoff that drives useful control variates."""
    s_star = _as_1d(data, "s_star")

    points: list[tuple[str, float, float, object, str]] = [("FOM", 1.0, 1.0, COLORS["fom"], "D")]
    for i, name in enumerate(_model_names(n_aux)):
        points.append((
            name,
            float(_as_1d(data, f"cost_aux{i}_vals")[0]),
            abs(float(_as_1d(data, f"rho_fom_aux{i}_pilot")[0])),
            _aux_color(i, n_aux, n_active),
            "o",
        ))
    for t in range(n_active):
        selected_basis = int(round(float(s_star[-(n_active - t)])))
        basis_grid = _grid(data, _rom_key(data, "ss", t))
        surrogate_corr = float(np.interp(selected_basis, basis_grid, _as_1d(data, _rom_key(data, "rho_fom_rom", t, "_vals"))))
        surrogate_cost = float(np.interp(selected_basis, basis_grid, _as_1d(data, _rom_key(data, "cost_rom", t, "_vals"))))
        rom_corr, rom_cost = _trained_rom_statistics(work_directory, t, surrogate_corr, surrogate_cost)
        label = _rom_label(t, n_active)
        points.append((
            f"Trained {label} ($s={selected_basis:.0f}$)", rom_cost, abs(rom_corr),
            _rom_color(t, n_active, n_aux), _rom_marker(t),
        ))

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
    pilot_data: Mapping[str, np.ndarray], n_aux: int,
    trained_rom_qois: Sequence[tuple[int, np.ndarray]] = (), n_active: int = 1,
) -> plt.Figure:
    """Create paired FOM-QoI comparisons for all cached control variates.

    trained_rom_qois: (rom_index, qois) pairs, one per trainable ROM whose
    optimized-basis QoIs were found on disk.
    """
    fom = np.asarray(pilot_data["fom_qois_master"], dtype=float).ravel()
    comparisons: list[tuple[str, np.ndarray, object]] = [
        (f"Auxiliary {i + 1}", np.asarray(pilot_data[f"aux{i}_qois_master"], dtype=float).ravel(), _aux_color(i, n_aux, n_active))
        for i in range(n_aux)
    ]
    for rom_index, qois in trained_rom_qois:
        comparisons.append((
            f"Trained {_rom_label(rom_index, n_active)}", qois.ravel(), _rom_color(rom_index, n_active, n_aux),
        ))

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
        n_active = int(np.asarray(data["n_active"]).item()) if "n_active" in data else 1
        s_star = _as_1d(data, "s_star")
        selected_bases = [int(round(float(s_star[-(n_active - t)]))) for t in range(n_active)]

        figures = {
            "fidelity_and_cost": _plot_fidelity_and_cost(data, n_aux, work_directory, n_active),
        }

        figures["estimator_performance"] = _plot_estimator_performance(data)
        figures["is_allocation"] = _plot_is_allocations(data, n_aux, work_directory, n_active)
        figures["control_variate_tradeoff"] = _plot_control_variate_tradeoff(data, n_aux, work_directory, n_active)

        if "fMFs_alloc" in data:
            figures["mf_allocation"] = _plot_mf_allocations(data, n_aux, work_directory, n_active)
        else:
            print(f"Skipping ACV-MF allocation figure: fMFs_alloc was not found in {vis_path.name}.")

    pilot_path = work_directory / "pilot_results.npz"
    if pilot_path.is_file():
        with np.load(pilot_path, allow_pickle=False) as pilot_data:
            n_samples = len(np.asarray(pilot_data["fom_qois_master"]).ravel())
            trained_rom_qois = []
            for t in range(n_active):
                qois = _load_trained_rom_qois(work_directory, t, selected_bases[t], n_samples)
                if qois is None:
                    print(f"Skipping trained-{_rom_label(t, n_active)} QoI scatter: cached optimized-ROM QoIs were not found.")
                else:
                    trained_rom_qois.append((t, qois))
            figures["pilot_qoi_agreement"] = _plot_pilot_qoi_agreement(pilot_data, n_aux, trained_rom_qois, n_active)
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
