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

import matplotlib.colors as mcolors
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


def _rom_training_color(rom_index: int, n_active: int, n_aux: int) -> object:
    """One color per trainable ROM's training cost, derived from that
    ROM's own sampling color (see _rom_color) by muting it toward gray.
    Training and sampling bars for the same ROM therefore share a hue and
    read as a pair, while different ROMs' training bars stay distinct --
    previously every ROM's training cost was drawn in the same flat gray,
    which made the training bars indistinguishable once n_active > 1. The
    single-ROM case keeps the original gray so those figures are
    unchanged."""
    if n_active == 1:
        return "#999999"
    base = np.asarray(mcolors.to_rgb(_rom_color(rom_index, n_active, n_aux)))
    gray = np.asarray(mcolors.to_rgb("#BFBFBF"))
    return tuple(0.42 * base + 0.58 * gray)


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
    data: Mapping[str, np.ndarray], n_aux: int, trained: TrainedRomRecord, n_active: int = 1,
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
        # The index comes from the record, which resolved it once and checked
        # it against what the workflow declared -- never re-derived here.
        selected_basis = trained.basis_size(t)
        label = _rom_label(t, n_active)
        color = _rom_color(t, n_active, n_aux)
        marker = _rom_marker(t)

        rho_fom_rom = _as_1d(data, _rom_key(data, "rho_fom_rom", t, "_vals"))
        cost_rom = _as_1d(data, _rom_key(data, "cost_rom", t, "_vals"))
        surrogate_corr = float(np.interp(selected_basis, basis, rho_fom_rom))
        surrogate_cost = float(np.interp(selected_basis, basis, cost_rom))
        trained_corr = trained.fom_correlation(t, surrogate_corr)
        trained_cost = trained.normalized_cost(t, surrogate_cost)
        surrogate_aux_corrs = [
            float(np.interp(selected_basis, basis, _as_1d(data, _rom_key(data, f"rho_aux{i}_rom", t, "_vals"))))
            for i in range(n_aux)
        ]
        trained_aux_corrs = trained.aux_correlations(t, surrogate_aux_corrs)

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
        marker_labels.append(f"{label} trained, $s={selected_basis:d}${trained.label_suffix(t)}")

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


def _trained_results_filename(selected_bases: Sequence[int]) -> str:
    """Reproduce the trained-results filename that run_hybrid_mfuq writes.

    Step 3 of run_hybrid_mfuq saves to
    ``trained_{basis_tag}_sample_rom_results.npz``, with
    ``basis_tag = "-".join(str(b) for b in rom_basis_nums)``.  The tag is the
    only thing distinguishing one trained-ROM result from another, so it has
    to be reconstructed exactly.  Earlier revisions took
    ``sorted(glob("trained_*_sample_rom_results.npz"))[-1]`` instead, which
    picks the lexicographically last basis tag present in the directory --
    not the tag belonging to the basis size the figure is labeling.  A work
    directory accumulates one such file per basis size it has ever been run
    at (Step 3 reuses a cached file whenever its tag already exists), so that
    glob quietly reported another ROM's statistics.  Single-vector ROMs were
    hit hardest: ``trained_1_...`` loses the sort to every tag whose first
    digit is 2-9, so an s=1 point was almost never its own.
    """
    tag = "-".join(str(int(basis)) for basis in selected_bases)
    return f"trained_{tag}_sample_rom_results.npz"


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation over jointly finite entries; NaN when undefined.

    This reproduces what the workflow itself computes at validation time:
    Pilot.estimate_pairwise_correlations reduces, for the single replicate
    used there, to a plain Pearson coefficient over all pilot samples.  It is
    also exactly what the QoI-agreement scatter reports, so both figures are
    guaranteed to agree by construction rather than by convention.

    Returns NaN rather than raising when a series is constant, which is a
    real possibility for a ROM built from a single basis vector.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        return float("nan")
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return float("nan")
    x, y = x[finite], y[finite]
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _legacy_rom_qoi_subdir(rom_index: int, basis_size: int) -> str:
    """The pilot/rom_optimized subdirectory convention of run_hybrid_mfuq.

    Only used for runs whose trained record predates `rom{t}_qoi_subdir`;
    newer runs declare the directory they wrote, so this convention does not
    have to be kept in sync across the two scripts.
    """
    return f"basis_size_{basis_size}" if rom_index == 0 else f"rom{rom_index}_basis_size_{basis_size}"


def _load_trained_rom_qois(
    work_directory: Path, subdir: str, n_samples: int,
) -> np.ndarray | None:
    """Recover cached trained-ROM QoIs, preserving their pilot-sample ordering."""
    rom_dir = work_directory / "pilot" / "rom_optimized" / subdir
    qoi_paths = [rom_dir / f"run_{sample_index}" / "qoi.txt" for sample_index in range(n_samples)]
    if not all(path.is_file() for path in qoi_paths):
        return None
    return np.asarray([np.loadtxt(path, dtype=float) for path in qoi_paths], dtype=float).reshape(n_samples)


class TrainedRomInconsistencyError(RuntimeError):
    """Raised when sources disagree about which ROM the trained record describes."""


class TrainedRomRecord:
    """The trained ROMs' identity and measured statistics, resolved once.

    Why this exists
    ---------------
    A trained-ROM result is meaningless without its index: which trainable
    ROM, at which basis size, over which pilot set. Older runs carried that
    index only in a filename (`trained_{tag}_sample_rom_results.npz`) and left
    each consumer to reconstruct it, so the basis size was re-derived from
    `s_star` independently in run_hybrid_mfuq Step 3, in the fidelity figure,
    in the tradeoff figure, and in the QoI scatter's directory lookup. Four
    derivations of one index, none of them checked against the others; a file
    picked by glob order could then supply a different ROM's numbers to one
    figure while another figure read the QoIs of the right one.

    What this class does instead
    ----------------------------
    The index is resolved *once*, preferring the value the producer declared,
    and every figure asks this object for both the basis size it labels and
    the statistics it draws. Nothing downstream re-derives either.

    Statistics resolution, in order:

    1. `visualization_data.npz`, which newer runs write with the validated
       trained-ROM statistics alongside the surrogate curves those statistics
       are plotted against. These are the correlations Step 4 actually fed
       into the validation optimization, so the trained markers are consistent
       with the validated variance curves in the estimator-performance figure.
    2. `trained_{tag}_sample_rom_results.npz` for the exact basis tag, for
       runs that predate (1). If that file declares its own `rom_basis_nums`,
       the declaration is verified rather than trusted to the filename.
    3. The surrogate value at the selected basis size, supplied by the caller
       and labeled as such in the figure.

    Independently of which source supplied them, the correlations are audited
    against the cached per-sample QoIs -- the same arrays the QoI-agreement
    scatter plots. Agreement is then a checked property of the output, not a
    convention. Disagreement means the record and the cached QoIs describe
    different ROMs, which raises by default (`on_inconsistency='raise'`).
    """

    CORRELATION_TOLERANCE = 5e-3

    def __init__(
        self,
        data: Mapping[str, np.ndarray],
        work_directory: Path,
        n_aux: int,
        n_active: int,
        on_inconsistency: str = "raise",
    ) -> None:
        self.work_directory = Path(work_directory)
        self.n_aux = int(n_aux)
        self.n_active = int(n_active)
        self.on_inconsistency = on_inconsistency

        self.basis_sizes = self._resolve_basis_sizes(data)
        self.results_path = self.work_directory / _trained_results_filename(self.basis_sizes)

        self._subdirs = self._resolve_qoi_subdirs(data)
        self._stored = self._read_stats(data)
        self._fom_qois, self._aux_qois = self._read_pilot_qois()

        self.qois: dict[int, np.ndarray] = {}
        self._fom_corr: dict[int, float] = {}
        self._aux_corr: dict[int, list[float | None]] = {}
        self._cost: dict[int, float] = {}
        self._sources: dict[int, dict[str, str]] = {}

        for rom_index in range(self.n_active):
            self._resolve_rom(rom_index)

    # -- identity ----------------------------------------------------------

    def _fail(self, message: str) -> None:
        """Raise or warn, depending on `on_inconsistency`."""
        if self.on_inconsistency == "raise":
            raise TrainedRomInconsistencyError(message)
        print(f"Warning: {message}")

    def _resolve_basis_sizes(self, data: Mapping[str, np.ndarray]) -> list[int]:
        """The one derivation of the trained basis sizes used by every figure.

        `trained_rom_basis_sizes` is what Step 3 actually built, recorded by
        the producer. Rounding `s_star` reproduces Step 3's own arithmetic and
        is the only option for older files, but it is a *re-derivation*: it
        agrees only as long as both scripts round identically. When both are
        available they are cross-checked, which is what turns the rounding
        convention from an unstated assumption into a verified one.
        """
        derived = [
            int(round(float(_as_1d(data, "s_star")[-(self.n_active - t)])))
            for t in range(self.n_active)
        ]
        if "trained_rom_basis_sizes" not in data:
            return derived

        declared = [int(round(v)) for v in np.asarray(data["trained_rom_basis_sizes"], dtype=float).ravel()]
        if len(declared) != self.n_active:
            self._fail(
                f"visualization_data.npz records {len(declared)} trained basis size(s) "
                f"but n_active={self.n_active}."
            )
            return derived
        if declared != derived:
            self._fail(
                f"The trained ROM basis sizes recorded by the workflow ({declared}) disagree "
                f"with the sizes implied by rounding s_star ({derived}). The figures would "
                "otherwise label one ROM and plot another; rerun the workflow, or delete "
                "visualization_data.npz and regenerate it."
            )
        return declared

    def _resolve_qoi_subdirs(self, data: Mapping[str, np.ndarray]) -> list[str]:
        """Per-ROM QoI directories, declared by the producer when available."""
        if "trained_rom_qoi_subdirs" in data:
            declared = [str(name) for name in np.asarray(data["trained_rom_qoi_subdirs"]).ravel()]
            if len(declared) == self.n_active:
                return declared
        return [
            _legacy_rom_qoi_subdir(t, self.basis_sizes[t]) for t in range(self.n_active)
        ]

    # -- statistics --------------------------------------------------------

    def _read_stats(self, data: Mapping[str, np.ndarray]) -> dict[str, float] | None:
        """Validated trained-ROM statistics from the best available source."""
        if "trained_rom_fom_corrs" in data:
            self.stats_source = "visualization_data.npz"
            stored = {}
            corrs = np.asarray(data["trained_rom_fom_corrs"], dtype=float).ravel()
            times = np.asarray(data["trained_rom_normalized_times"], dtype=float).ravel()
            for t in range(min(self.n_active, corrs.size)):
                stored[f"rom{t}_fom_corr"] = float(corrs[t])
                stored[f"rom{t}_normalized_time"] = float(times[t])
            for i in range(self.n_aux):
                key = f"trained_aux{i}_rom_corrs"
                if key in data:
                    aux = np.asarray(data[key], dtype=float).ravel()
                    for t in range(min(self.n_active, aux.size)):
                        stored[f"aux{i}_rom{t}_corr"] = float(aux[t])
            return stored

        self.stats_source = self.results_path.name
        if not self.results_path.is_file():
            others = sorted(
                path.name for path in self.work_directory.glob("trained_*_sample_rom_results.npz")
            )
            print(f"Trained-ROM statistics: {self.results_path.name} was not found.")
            if others:
                print(
                    f"  Ignoring trained results for other basis sizes ({', '.join(others)}); "
                    "they describe different ROMs."
                )
            return None

        with np.load(self.results_path, allow_pickle=False) as trained_data:
            declared = (
                [int(round(v)) for v in np.asarray(trained_data["rom_basis_nums"], dtype=float).ravel()]
                if "rom_basis_nums" in trained_data
                else None
            )
            stored = {}
            for key in trained_data.files:
                values = np.asarray(trained_data[key], dtype=float).reshape(-1) if trained_data[key].dtype.kind in "fiub" else np.empty(0)
                if values.size == 1:
                    stored[key] = float(values[0])

        if declared is not None and declared != self.basis_sizes:
            self._fail(
                f"{self.results_path.name} records basis sizes {declared} but the figures are "
                f"drawing {self.basis_sizes}. The filename tag and the file's contents disagree."
            )
            return None
        return stored

    def _read_pilot_qois(self) -> tuple[np.ndarray | None, list[np.ndarray]]:
        """The master pilot QoIs the audit correlates against."""
        pilot_path = self.work_directory / "pilot_results.npz"
        if not pilot_path.is_file():
            print(
                f"Trained-ROM statistics: {pilot_path.name} was not found, so the recorded "
                "correlations cannot be audited against the cached QoIs."
            )
            return None, []
        with np.load(pilot_path, allow_pickle=False) as pilot_data:
            fom_qois = np.asarray(pilot_data["fom_qois_master"], dtype=float).ravel()
            aux_qois = [
                np.asarray(pilot_data[f"aux{i}_qois_master"], dtype=float).ravel()
                for i in range(self.n_aux)
                if f"aux{i}_qois_master" in pilot_data
            ]
        return fom_qois, aux_qois

    def _stored_scalar(self, *keys: str) -> float | None:
        if self._stored is None:
            return None
        for key in keys:
            if key in self._stored:
                return self._stored[key]
        return None

    def _resolve_rom(self, rom_index: int) -> None:
        label = _rom_label(rom_index, self.n_active)
        basis = self.basis_sizes[rom_index]
        legacy = rom_index == 0
        sources: dict[str, str] = {}

        qois = None
        if self._fom_qois is not None:
            qois = _load_trained_rom_qois(
                self.work_directory, self._subdirs[rom_index], self._fom_qois.size
            )
            if qois is None:
                print(
                    f"Trained {label}: cached QoIs for s={basis} "
                    f"({self._subdirs[rom_index]}) are missing or incomplete; the recorded "
                    "statistics cannot be audited."
                )
        if qois is not None:
            self.qois[rom_index] = qois

        # FOM-ROM correlation: recorded value drawn, QoI value audits it.
        recorded = self._stored_scalar(
            f"rom{rom_index}_fom_corr", *(("fom_rom_corr",) if legacy else ())
        )
        audited = _pearson(self._fom_qois, qois) if qois is not None else float("nan")
        value = self._reconcile(label, basis, "FOM-ROM correlation", recorded, audited)
        if value is not None:
            self._fom_corr[rom_index] = value
            sources["fom_corr"] = "recorded" if recorded is not None else "qois"

        # Auxiliary-ROM correlations.
        aux_values: list[float | None] = []
        for i in range(self.n_aux):
            recorded_aux = self._stored_scalar(
                f"aux{i}_rom{rom_index}_corr", *((f"aux{i}_rom_corr",) if legacy else ())
            )
            audited_aux = (
                _pearson(self._aux_qois[i], qois)
                if qois is not None and i < len(self._aux_qois)
                else float("nan")
            )
            aux_values.append(
                self._reconcile(
                    label, basis, f"auxiliary {i + 1}-ROM correlation", recorded_aux, audited_aux
                )
            )
        self._aux_corr[rom_index] = aux_values
        if self.n_aux:
            sources["aux_corr"] = "surrogate" if any(v is None for v in aux_values) else "recorded"

        # Normalized cost: a timing measurement, not recoverable from QoIs.
        cost = self._stored_scalar(
            f"rom{rom_index}_normalized_time", *(("normalized_rom_time",) if legacy else ())
        )
        if cost is not None:
            self._cost[rom_index] = cost
            sources["cost"] = "recorded"

        for kind, description in (("fom_corr", "FOM-ROM correlation"), ("cost", "normalized cost")):
            if kind not in sources:
                sources[kind] = "surrogate"
                print(
                    f"Trained {label}: no measured {description} for s={basis}; the figure "
                    "shows the surrogate value, marked as such."
                )

        self._sources[rom_index] = sources

    def _reconcile(
        self, label: str, basis: int, description: str,
        recorded: float | None, audited: float,
    ) -> float | None:
        """Return the value to draw, after checking the two sources agree.

        The recorded value is drawn when present: it is what the validation
        optimization consumed, so drawing it keeps the trained markers
        consistent with the validated variance curves. The QoI-derived value
        is the audit -- in a coherent run the two are the same number computed
        twice, so any gap is a provenance failure, not a modeling choice.
        """
        if not np.isfinite(audited):
            return recorded
        if recorded is None:
            return audited
        if abs(recorded - audited) > self.CORRELATION_TOLERANCE:
            self._fail(
                f"Trained {label} {description} at s={basis} is {audited:.4f} recomputed from "
                f"the cached QoIs but {recorded:.4f} in {self.stats_source}. The two describe "
                "different ROMs; retrain at this basis size, or delete the stale artifact."
            )
            return audited
        return recorded

    # -- accessors ---------------------------------------------------------

    def basis_size(self, rom_index: int) -> int:
        """The basis size every figure labels for ROM `rom_index`."""
        return self.basis_sizes[rom_index]

    def fom_correlation(self, rom_index: int, surrogate_fallback: float) -> float:
        return self._fom_corr.get(rom_index, surrogate_fallback)

    def aux_correlations(
        self, rom_index: int, surrogate_fallbacks: Sequence[float],
    ) -> list[float]:
        values = self._aux_corr.get(rom_index) or [None] * len(surrogate_fallbacks)
        return [
            fallback if value is None else value
            for value, fallback in zip(values, surrogate_fallbacks)
        ]

    def normalized_cost(self, rom_index: int, surrogate_fallback: float) -> float:
        return self._cost.get(rom_index, surrogate_fallback)

    def label_suffix(self, rom_index: int) -> str:
        """', surrogate' when a drawn value is not a measurement, else ''."""
        sources = self._sources.get(rom_index, {})
        return ", surrogate" if any(s == "surrogate" for s in sources.values()) else ""

    def qoi_pairs(self) -> list[tuple[int, np.ndarray]]:
        """(rom_index, qois) for every ROM whose cached QoIs were found."""
        return [(rom_index, self.qois[rom_index]) for rom_index in sorted(self.qois)]


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
        colors.extend(_rom_training_color(t, n_active, n_aux) for t in range(n_active))
    return np.column_stack(components), labels, colors


def _stacked_cost_bars(
    ax: plt.Axes, budget: np.ndarray, components: np.ndarray, labels: Sequence[str], colors: Sequence[object],
    title: str, highlight_index: int | None = None, highlight_label: str = "",
) -> list[object]:
    """Plot one interpretable cost-allocation bar per budget.

    Bars sit on a categorical axis: the plotting area is divided into one
    equally sized slot per budget, and bar width and spacing are fixed
    fractions of a slot. Budget values therefore set only the tick
    labels, not the geometry, so a widely spread sweep (a log-spaced one,
    say) still gives every budget the same visual weight instead of
    crowding the small budgets together at the left edge.

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
    positions = np.arange(len(budget), dtype=float)
    bar_width = 0.72
    for values, label, color in zip(components.T, labels, colors):
        bars = ax.bar(positions, values, width=bar_width, bottom=bottoms, color=color, edgecolor="white", linewidth=0.55)
        handles.append(bars[0])
        bottoms += values

    top_lim = max(np.max(bottoms), np.max(budget)) * 1.10
    if highlight_index is not None:
        top = bottoms[highlight_index]
        ax.add_patch(plt.Rectangle(
            (positions[highlight_index] - bar_width * 0.56, 0), bar_width * 1.12, top,
            fill=True, facecolor="0.5", alpha=0.16,
            edgecolor="0.35", linewidth=1.6, linestyle="--", zorder=0.5,
        ))
        ax.annotate(
            highlight_label or "trained here",
            (positions[highlight_index], top), xytext=(0, 8), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color="0.3",
        )
        top_lim = max(top_lim, top * 1.22)

    ax.set_title(title, pad=9)
    ax.set(xlabel="Budget (FOM equivalents)", ylabel="Allocated cost (FOM equivalents)")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{value:g}" for value in budget])
    ax.set_xlim(-0.5, len(budget) - 0.5)
    ax.set_ylim(0, top_lim)
    return handles


def _validation_budget_index(
    data: Mapping[str, np.ndarray], predicted: np.ndarray, n_aux: int,
    basis_sizes: Sequence[int],
) -> int:
    """Identify which budget's surrogate-optimized ROM basis size(s) were
    actually trained and carried into the validation panel (see
    run_hybrid_mfuq Step 3's validation_budget_idx).

    Prefers the explicit 'validation_budget_idx' field written by newer
    runs; falls back to matching the trained basis size(s) against each
    budget's surrogate-optimized basis size(s), for older
    visualization_data.npz files that predate that field. `basis_sizes` is
    the record's resolved index, so this fallback matches on the same
    numbers every other figure draws.
    """
    if "validation_budget_idx" in data:
        return int(np.asarray(data["validation_budget_idx"]).item())

    n_active = len(basis_sizes)
    if n_active == 0:
        return len(predicted) - 1

    target = np.asarray(basis_sizes, dtype=float)
    basis_cols = predicted[:, n_aux + n_active + 1: n_aux + n_active + 1 + n_active]
    matches = np.all(np.round(basis_cols) == target[None, :], axis=1)
    return int(np.argmax(matches)) if matches.any() else len(predicted) - 1


def _plot_allocation_figure(
    data: Mapping[str, np.ndarray], n_aux: int, trained: TrainedRomRecord,
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
    validation_idx = _validation_budget_index(data, predicted, n_aux, trained.basis_sizes)

    exact_costs = [
        trained.normalized_cost(
            t,
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
    data: Mapping[str, np.ndarray], n_aux: int, trained: TrainedRomRecord, n_active: int = 1,
) -> plt.Figure:
    """Show the ACV-IS sampling strategy as a cost-partition bar chart."""
    return _plot_allocation_figure(data, n_aux, trained, "fISs_alloc", "fISs_alloc_ex", "ACV-IS", n_active)


def _plot_mf_allocations(
    data: Mapping[str, np.ndarray], n_aux: int, trained: TrainedRomRecord, n_active: int = 1,
) -> plt.Figure:
    """Show the ACV-MF sampling strategy as a cost-partition bar chart."""
    return _plot_allocation_figure(data, n_aux, trained, "fMFs_alloc", "fMFs_alloc_ex", "ACV-MF", n_active)


def _plot_control_variate_tradeoff(
    data: Mapping[str, np.ndarray], n_aux: int, trained: TrainedRomRecord, n_active: int = 1,
) -> plt.Figure:
    """Show the cost-agreement tradeoff that drives useful control variates.

    Trained-ROM correlations come from `trained`, i.e. from the same cached
    QoIs the agreement scatter plots, so the |rho| drawn here for a ROM is the
    magnitude of the rho annotated on that ROM's scatter panel. Surrogate
    fallbacks are labeled as such rather than presented as measurements.
    """
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
        selected_basis = trained.basis_size(t)
        basis_grid = _grid(data, _rom_key(data, "ss", t))
        surrogate_corr = float(np.interp(selected_basis, basis_grid, _as_1d(data, _rom_key(data, "rho_fom_rom", t, "_vals"))))
        surrogate_cost = float(np.interp(selected_basis, basis_grid, _as_1d(data, _rom_key(data, "cost_rom", t, "_vals"))))
        rom_corr = trained.fom_correlation(t, surrogate_corr)
        rom_cost = trained.normalized_cost(t, surrogate_cost)
        label = _rom_label(t, n_active)
        points.append((
            f"Trained {label} ($s={selected_basis:.0f}${trained.label_suffix(t)})",
            rom_cost, abs(rom_corr),
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
    optimized-basis QoIs were found on disk. Supplied by TrainedRomRecord so
    these panels and the cost-agreement tradeoff correlate the same arrays.
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


def generate_figures(
    work_directory: Path, output_dir: Path, formats: Sequence[str], show: bool,
    on_inconsistency: str = "raise",
) -> list[Path]:
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
        # The trained ROMs' identity and statistics, resolved once and shared
        # by every figure below. The basis index is resolved here and nowhere
        # else, so no two figures can label or draw different ROMs.
        trained = TrainedRomRecord(data, work_directory, n_aux, n_active, on_inconsistency)

        figures = {
            "fidelity_and_cost": _plot_fidelity_and_cost(data, n_aux, trained, n_active),
        }

        figures["estimator_performance"] = _plot_estimator_performance(data)
        figures["is_allocation"] = _plot_is_allocations(data, n_aux, trained, n_active)
        figures["control_variate_tradeoff"] = _plot_control_variate_tradeoff(data, n_aux, trained, n_active)

        if "fMFs_alloc" in data:
            figures["mf_allocation"] = _plot_mf_allocations(data, n_aux, trained, n_active)
        else:
            print(f"Skipping ACV-MF allocation figure: fMFs_alloc was not found in {vis_path.name}.")

    pilot_path = work_directory / "pilot_results.npz"
    if pilot_path.is_file():
        with np.load(pilot_path, allow_pickle=False) as pilot_data:
            trained_rom_qois = trained.qoi_pairs()
            for t in range(n_active):
                if t not in trained.qois:
                    print(
                        f"Skipping trained-{_rom_label(t, n_active)} QoI scatter: "
                        "cached optimized-ROM QoIs were not found."
                    )
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
    parser.add_argument(
        "--on-inconsistency", choices=("raise", "warn"), default="raise",
        help=(
            "What to do when the trained-ROM record, the basis size implied by s_star, and the "
            "cached QoIs disagree about which ROM is being drawn (default: raise)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    work_directory = args.work_directory.expanduser().resolve()
    output_dir = (args.output_dir or work_directory / "publication_figures").expanduser().resolve()
    outputs = generate_figures(
        work_directory, output_dir, args.formats, show=not args.no_show,
        on_inconsistency=args.on_inconsistency,
    )
    print("Created:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
