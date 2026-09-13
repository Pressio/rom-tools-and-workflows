"""Reproduce the Stickle et al. nonlinear cantilever benchmark.

Reference:
M. M. Stickle et al., Computational Mechanics 69, 639--660 (2022),
https://doi.org/10.1007/s00466-021-02107-0
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

MODELS = Path(__file__).resolve().parents[1] / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from nonlinear_solid_dynamics import cantilever_model  # noqa: E402


def run(dt=1.0e-3, t_end=3.0, snapshot_stride=10):
    model = cantilever_model(
        length=4.0,
        height=1.0,
        nx=8,
        ny=2,
        young_modulus=1.0e6,
        poisson_ratio=0.3,
        density=1050.0,
        mass_type="consistent",
    )
    gravity = lambda _time: model.body_force(np.array([0.0, -10.0]))
    state = model.initial_state(external_force=gravity)
    times, displacements, velocities = model.solve_implicit(
        state,
        dt=dt,
        num_steps=int(round(t_end / dt)),
        external_force=gravity,
        beta=0.25,
        gamma=0.5,
        newton_tolerance=1.0e-8,
        snapshot_stride=snapshot_stride,
    )
    return model, times, displacements, velocities


def main():
    model, times, displacements, _velocities = run()
    node_a = model.mesh.node_nearest(4.0, 0.0)
    response = displacements[:, 2 * node_a + 1]
    first_window = times <= 1.2
    i_min = np.argmin(response[first_window])
    print(
        "First minimum: "
        f"u_y={response[first_window][i_min]:.4f} m at "
        f"t={times[first_window][i_min]:.4f} s"
    )
    plt.plot(times, response, label="romtools Q4")
    plt.xlabel("Time [s]")
    plt.ylabel("Vertical displacement at node A [m]")
    plt.title("Stickle et al. nonlinear cantilever benchmark")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
