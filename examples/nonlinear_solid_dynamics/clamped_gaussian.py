"""Doubly clamped solid released from a Gaussian transverse perturbation."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

MODELS = Path(__file__).resolve().parents[1] / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from solid_dynamics import doubly_clamped_model, gaussian_displacement  # noqa: E402


def run(material_model="neo_hookean"):
    model = doubly_clamped_model(
        nx=12,
        ny=2,
        mass_type="consistent",
        material_model=material_model,
    )
    initial_displacement = gaussian_displacement(
        model,
        amplitude=0.15,
        width=0.45,
        center=0.5 * model.mesh.length,
        direction="transverse",
    )
    zero_force = lambda _time: np.zeros(model.ndof)
    state = model.initial_state(
        displacement=initial_displacement,
        external_force=zero_force,
    )
    times, displacements, velocities = model.solve_implicit(
        state,
        dt=0.005,
        num_steps=200,
        external_force=zero_force,
        snapshot_stride=2,
    )
    return model, times, displacements, velocities


def main():
    model, times, displacements, _velocities = run()
    center = model.mesh.node_nearest(0.5 * model.mesh.length, 0.5 * model.mesh.height)
    center_y = displacements[:, 2 * center + 1]
    plt.plot(times, center_y)
    plt.xlabel("Time [s]")
    plt.ylabel("Center vertical displacement [m]")
    plt.title("Doubly clamped Gaussian perturbation")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
