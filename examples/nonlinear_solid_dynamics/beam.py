"""Transiently loaded cantilever beam example."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

MODELS = Path(__file__).resolve().parents[1] / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from solid_dynamics import cantilever_model  # noqa: E402


def run(material_model="neo_hookean"):
    model = cantilever_model(
        nx=8,
        ny=2,
        mass_type="consistent",
        material_model=material_model,
    )
    load_amplitude = 1000.0
    pulse_duration = 0.15
    edge_shape = model.boundary_force_x(model.mesh.length, np.array([0.0, -1.0]))

    def external_force(time):
        if 0.0 <= time <= pulse_duration:
            return load_amplitude * np.sin(np.pi * time / pulse_duration) * edge_shape
        return np.zeros(model.ndof)

    state = model.initial_state(external_force=external_force)
    times, displacements, velocities = model.solve_implicit(
        state,
        dt=0.01,
        num_steps=100,
        external_force=external_force,
        snapshot_stride=1,
    )
    return model, times, displacements, velocities


def main():
    model, times, displacements, _velocities = run()
    tip = model.mesh.node_nearest(model.mesh.length, 0.5 * model.mesh.height)
    tip_y = displacements[:, 2 * tip + 1]
    plt.plot(times, tip_y)
    plt.xlabel("Time [s]")
    plt.ylabel("Tip vertical displacement [m]")
    plt.title("Cantilever response")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
