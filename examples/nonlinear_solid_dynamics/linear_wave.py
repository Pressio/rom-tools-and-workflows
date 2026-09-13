"""Linear longitudinal Gaussian pulse in a doubly clamped bar."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

MODELS = Path(__file__).resolve().parents[1] / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from solid_dynamics import (  # noqa: E402
    gaussian_displacement,
    longitudinal_wave_model,
    longitudinal_wave_speed,
)


def run():
    model = longitudinal_wave_model(nx=80, ny=1, mass_type="lumped")
    initial_displacement = gaussian_displacement(
        model,
        amplitude=0.02,
        width=0.20,
        center=0.5 * model.mesh.length,
        direction="axial",
    )
    zero_force = lambda _time: np.zeros(model.ndof)
    state = model.initial_state(
        displacement=initial_displacement,
        external_force=zero_force,
    )

    wave_speed = longitudinal_wave_speed(model.material)
    dx = model.mesh.length / model.mesh.nx
    dt = 0.20 * dx / wave_speed
    t_end = 0.04
    num_steps = int(np.ceil(t_end / dt))
    dt = t_end / num_steps
    times, displacements, velocities = model.solve_explicit(
        state,
        dt=dt,
        num_steps=num_steps,
        external_force=zero_force,
        snapshot_stride=max(1, num_steps // 20),
    )
    return model, wave_speed, times, displacements, velocities


def main():
    model, wave_speed, times, displacements, _velocities = run()
    x = model.mesh.coordinates[: model.mesh.nx + 1, 0]
    for index in (0, len(times) // 2, len(times) - 1):
        ux = displacements[index, 0 : 2 * (model.mesh.nx + 1) : 2]
        plt.plot(x, ux, label=f"t={times[index]:.3f} s")
    plt.xlabel("x [m]")
    plt.ylabel("Axial displacement [m]")
    plt.title(f"Linear two-way wave, c={wave_speed:.2f} m/s")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
