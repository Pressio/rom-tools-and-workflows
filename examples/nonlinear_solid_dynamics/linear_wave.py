"""Linear longitudinal Gaussian pulse and convergence study."""

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
    two_way_gaussian_solution,
)


def solve_wave(
    nx=80,
    length=4.0,
    height=1.0,
    amplitude=0.02,
    width=0.20,
    center=None,
    t_end=0.04,
    cfl=0.20,
    snapshot_stride=None,
):
    model = longitudinal_wave_model(
        length=length,
        height=height,
        nx=nx,
        ny=1,
        mass_type="lumped",
    )
    if center is None:
        center = 0.5 * model.mesh.length
    initial_displacement = gaussian_displacement(
        model,
        amplitude=amplitude,
        width=width,
        center=center,
        direction="axial",
    )
    zero_force = lambda _time: np.zeros(model.ndof)
    state = model.initial_state(
        displacement=initial_displacement,
        external_force=zero_force,
    )

    wave_speed = longitudinal_wave_speed(model.material)
    dx = model.mesh.length / model.mesh.nx
    dt_target = cfl * dx / wave_speed
    num_steps = int(np.ceil(t_end / dt_target))
    dt = t_end / num_steps
    if snapshot_stride is None:
        snapshot_stride = max(1, num_steps // 20)
    times, displacements, velocities = model.solve_explicit(
        state,
        dt=dt,
        num_steps=num_steps,
        external_force=zero_force,
        snapshot_stride=snapshot_stride,
    )
    return model, wave_speed, times, displacements, velocities


def run():
    return solve_wave()


def convergence_study(
    nx_values=(32, 64, 128),
    length=4.0,
    amplitude=0.02,
    width=0.40,
    t_end=0.005,
    cfl=0.10,
):
    """Compare the FE solution with the analytic two-way Gaussian solution."""
    errors = []
    spacings = []
    for nx in nx_values:
        model, wave_speed, times, displacements, _velocities = solve_wave(
            nx=nx,
            length=length,
            amplitude=amplitude,
            width=width,
            center=0.5 * length,
            t_end=t_end,
            cfl=cfl,
            snapshot_stride=10**9,
        )
        x = model.mesh.coordinates[: model.mesh.nx + 1, 0]
        numerical = displacements[-1, 0 : 2 * (model.mesh.nx + 1) : 2]
        exact = two_way_gaussian_solution(
            x,
            times[-1],
            amplitude=amplitude,
            width=width,
            center=0.5 * length,
            wave_speed=wave_speed,
        )
        dx = length / nx
        error = np.sqrt(dx * np.sum((numerical - exact) ** 2))
        exact_norm = np.sqrt(dx * np.sum(exact**2))
        errors.append(error / exact_norm)
        spacings.append(dx)

    rates = [
        np.log(errors[i - 1] / errors[i])
        / np.log(spacings[i - 1] / spacings[i])
        for i in range(1, len(errors))
    ]
    return np.asarray(spacings), np.asarray(errors), np.asarray(rates)


def main():
    model, wave_speed, times, displacements, _velocities = run()
    x = model.mesh.coordinates[: model.mesh.nx + 1, 0]
    for index in (0, len(times) // 2, len(times) - 1):
        ux = displacements[index, 0 : 2 * (model.mesh.nx + 1) : 2]
        plt.plot(x, ux, label=f"FE, t={times[index]:.3f} s")
    exact = two_way_gaussian_solution(
        x,
        times[-1],
        amplitude=0.02,
        width=0.20,
        center=0.5 * model.mesh.length,
        wave_speed=wave_speed,
    )
    plt.plot(x, exact, "--", label=f"analytic, t={times[-1]:.3f} s")
    plt.xlabel("x [m]")
    plt.ylabel("Axial displacement [m]")
    plt.title(f"Linear two-way wave, c={wave_speed:.2f} m/s")
    plt.legend()
    plt.tight_layout()
    plt.show()

    spacings, errors, rates = convergence_study()
    print("\nTwo-way Gaussian convergence study")
    print("    dx        relative L2 error       observed rate")
    for i, (dx, error) in enumerate(zip(spacings, errors)):
        rate = "-" if i == 0 else f"{rates[i - 1]:.3f}"
        print(f"{dx:10.4e}   {error:16.8e}   {rate:>12}")


if __name__ == "__main__":
    main()
