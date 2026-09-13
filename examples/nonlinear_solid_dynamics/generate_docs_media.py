"""Generate solid-dynamics documentation figures and animation from the examples."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MODELS = REPO_ROOT / "examples" / "models"
for path in (HERE, MODELS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import beam  # noqa: E402
import clamped_gaussian  # noqa: E402
import linear_wave  # noqa: E402
from solid_dynamics import two_way_gaussian_solution  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "docs" / "source" / "_static" / "solid_dynamics"


def _save(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_support(ax, x, y0, y1, side):
    ax.plot([x, x], [y0, y1], "k", lw=2)
    direction = -1.0 if side == "left" else 1.0
    for y in np.linspace(y0, y1, 8):
        ax.plot([x, x + 0.12 * direction], [y, y - 0.12], "k", lw=1)


def cantilever_setup_figure():
    fig, ax = plt.subplots(figsize=(8, 3))
    length, height = 4.0, 1.0
    ax.add_patch(Rectangle((0, 0), length, height, facecolor="0.9", edgecolor="k"))
    _draw_support(ax, 0.0, -0.15, 1.15, "left")
    for y in np.linspace(0.15, 0.85, 5):
        ax.annotate("", xy=(length, y - 0.13), xytext=(length, y + 0.13),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.text(length - 0.05, 1.12, r"transient $F(t)$", ha="right")
    ax.text(-0.25, 0.5, "clamped", ha="right", va="center")
    ax.annotate("", xy=(length, -0.28), xytext=(0.0, -0.28),
                arrowprops=dict(arrowstyle="<->"))
    ax.text(length / 2, -0.38, r"$L$", ha="center")
    ax.annotate("", xy=(length + 0.25, height), xytext=(length + 0.25, 0.0),
                arrowprops=dict(arrowstyle="<->"))
    ax.text(length + 0.34, height / 2, r"$H$", va="center")
    ax.set_xlim(-0.55, length + 0.6)
    ax.set_ylim(-0.55, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Cantilever beam with transient transverse loading")
    _save(fig, "cantilever_setup.png")


def gaussian_setup_figure():
    fig, ax = plt.subplots(figsize=(8, 3))
    length, height = 4.0, 1.0
    ax.add_patch(Rectangle((0, 0), length, height, facecolor="0.9", edgecolor="k"))
    _draw_support(ax, 0.0, -0.15, 1.15, "left")
    _draw_support(ax, length, -0.15, 1.15, "right")
    x = np.linspace(0.0, length, 400)
    g = 0.5 + 0.65 * np.exp(-0.5 * ((x - length / 2) / 0.55) ** 2)
    ax.plot(x, g, lw=2)
    ax.axhline(0.5, ls="--", lw=1, color="0.5")
    ax.text(length / 2, 1.28,
            r"$u_y(X,0)=A\exp[-(X-X_c)^2/(2\sigma^2)]$",
            ha="center")
    ax.text(-0.18, 0.5, "clamped", ha="right", va="center")
    ax.text(length + 0.18, 0.5, "clamped", ha="left", va="center")
    ax.set_xlim(-0.65, length + 0.65)
    ax.set_ylim(-0.35, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Doubly clamped Gaussian perturbation")
    _save(fig, "clamped_gaussian_setup.png")


def wave_setup_figure():
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    length = 4.0
    x = np.linspace(0.0, length, 400)
    for ax in axes:
        ax.add_patch(Rectangle((0, 0), length, 0.25, facecolor="0.9", edgecolor="k"))
        _draw_support(ax, 0.0, -0.08, 0.33, "left")
        _draw_support(ax, length, -0.08, 0.33, "right")
        ax.text(-0.12, 0.125, r"$u=0$", ha="right", va="center")
        ax.text(length + 0.12, 0.125, r"$u=0$", ha="left", va="center")
        ax.set_ylim(-0.12, 0.9)
        ax.axis("off")
    g0 = 0.18 + 0.55 * np.exp(-0.5 * ((x - 2.0) / 0.28) ** 2)
    axes[0].plot(x, g0, lw=2)
    axes[0].text(2.0, 0.78, r"$g(x)$", ha="center")
    axes[0].set_title("Initial condition")
    gl = 0.18 + 0.28 * np.exp(-0.5 * ((x - 1.2) / 0.28) ** 2)
    gr = 0.18 + 0.28 * np.exp(-0.5 * ((x - 2.8) / 0.28) ** 2)
    axes[1].plot(x, gl, lw=2)
    axes[1].plot(x, gr, lw=2)
    axes[1].annotate(r"$c_p$", xy=(0.55, 0.50), xytext=(0.95, 0.50),
                     arrowprops=dict(arrowstyle="-|>"), va="center")
    axes[1].annotate(r"$c_p$", xy=(3.45, 0.50), xytext=(3.05, 0.50),
                     arrowprops=dict(arrowstyle="-|>"), va="center")
    axes[1].set_title("Two-way propagation")
    fig.suptitle("Longitudinal two-way wave")
    fig.tight_layout()
    _save(fig, "linear_wave_setup.png")


def cantilever_solution_and_animation():
    model, times, displacements, _ = beam.run()
    tip = model.mesh.node_nearest(model.mesh.length, 0.5 * model.mesh.height)
    tip_y = displacements[:, 2 * tip + 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, tip_y)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Tip vertical displacement [m]")
    ax.set_title("Cantilever response")
    fig.tight_layout()
    _save(fig, "cantilever_response.png")

    coords = model.mesh.coordinates
    elements = model.mesh.elements
    u_hist = displacements.reshape(displacements.shape[0], model.mesh.num_nodes, 2)
    max_disp = float(np.max(np.linalg.norm(u_hist, axis=2)))
    if max_disp > 0.0:
        scale = min(20.0, max(1.0, 0.5 * model.mesh.height / max_disp))
    else:
        scale = 1.0

    def segments(points):
        return [points[np.r_[elem, elem[0]]] for elem in elements]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.add_collection(LineCollection(segments(coords), colors="0.75", linewidths=0.8))
    deformed = LineCollection(segments(coords), linewidths=1.5)
    ax.add_collection(deformed)
    time_text = ax.text(0.02, 0.94, "", transform=ax.transAxes, va="top")
    ax.text(0.98, 0.94, f"deformation x{scale:.1f}", transform=ax.transAxes,
            ha="right", va="top")
    margin = 0.35 * model.mesh.height
    ax.set_xlim(-margin, model.mesh.length + margin)
    ax.set_ylim(-margin - 0.5 * model.mesh.height, 1.5 * model.mesh.height + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Cantilever deformation")

    frame_ids = np.arange(0, len(times), 2)
    if frame_ids[-1] != len(times) - 1:
        frame_ids = np.append(frame_ids, len(times) - 1)

    def update(frame_id):
        points = coords + scale * u_hist[frame_id]
        deformed.set_segments(segments(points))
        time_text.set_text(f"t = {times[frame_id]:.3f} s")
        return deformed, time_text

    movie = animation.FuncAnimation(
        fig, update, frames=frame_ids, interval=80, blit=False
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    movie.save(
        OUTPUT_DIR / "cantilever_deformation.gif",
        writer=animation.PillowWriter(fps=12),
        dpi=120,
    )
    plt.close(fig)


def gaussian_solution_figure():
    model, times, displacements, _ = clamped_gaussian.run()
    center = model.mesh.node_nearest(0.5 * model.mesh.length, 0.5 * model.mesh.height)
    center_y = displacements[:, 2 * center + 1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, center_y)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Center vertical displacement [m]")
    ax.set_title("Doubly clamped Gaussian response")
    fig.tight_layout()
    _save(fig, "clamped_gaussian_response.png")


def wave_solution_and_convergence_figures():
    model, wave_speed, times, displacements, _ = linear_wave.solve_wave(
        nx=128,
        length=4.0,
        amplitude=0.02,
        width=0.40,
        t_end=0.005,
        cfl=0.10,
        snapshot_stride=10**9,
    )
    x = model.mesh.coordinates[: model.mesh.nx + 1, 0]
    numerical = displacements[-1, 0 : 2 * (model.mesh.nx + 1) : 2]
    exact = two_way_gaussian_solution(
        x,
        times[-1],
        amplitude=0.02,
        width=0.40,
        center=0.5 * model.mesh.length,
        wave_speed=wave_speed,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, numerical, label="FE solution")
    ax.plot(x, exact, "--", label="analytic solution")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Axial displacement [m]")
    ax.set_title(f"Two-way wave at t={times[-1]:.3f} s")
    ax.legend()
    fig.tight_layout()
    _save(fig, "linear_wave_solution.png")

    spacings, errors, rates = linear_wave.convergence_study()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.loglog(spacings, errors, "o-")
    for i, rate in enumerate(rates, start=1):
        ax.annotate(f"p={rate:.2f}", (spacings[i], errors[i]),
                    xytext=(8, 6), textcoords="offset points")
    ax.set_xlabel(r"$\Delta x$ [m]")
    ax.set_ylabel(r"Relative discrete $L^2$ error")
    ax.set_title("Linear-wave convergence")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    _save(fig, "linear_wave_convergence.png")


def main():
    cantilever_setup_figure()
    gaussian_setup_figure()
    wave_setup_figure()
    cantilever_solution_and_animation()
    gaussian_solution_figure()
    wave_solution_and_convergence_figures()
    print(f"Wrote solid-dynamics documentation media to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
