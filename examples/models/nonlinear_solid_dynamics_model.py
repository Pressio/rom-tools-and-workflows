"""romtools QoI-model wrapper for the solid-dynamics example."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

try:
    import solid_dynamics
except ImportError:  # pragma: no cover
    from . import solid_dynamics


class NonlinearSolidBeamQoiModel:
    """Parameterised cantilever model with tip displacement as the QoI.

    The parameter sample accepts ``young_modulus``, ``load_amplitude``, and
    ``pulse_duration``.  The full displacement and velocity snapshot histories
    are retained so the same evaluations can later be reused for ROM training.
    ``material_model`` may be set to ``"neo_hookean"`` (default) or ``"linear"``
    when constructing the wrapper.
    """

    def __init__(
        self,
        nx: int = 8,
        ny: int = 2,
        length: float = 4.0,
        height: float = 1.0,
        poisson_ratio: float = 0.3,
        density: float = 1050.0,
        dt: float = 1.0e-2,
        t_end: float = 0.6,
        snapshot_stride: int = 2,
        material_model: str = "neo_hookean",
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.length = float(length)
        self.height = float(height)
        self.poisson_ratio = float(poisson_ratio)
        self.density = float(density)
        self.dt = float(dt)
        self.t_end = float(t_end)
        self.snapshot_stride = int(snapshot_stride)
        self.material_model = str(material_model)

    def populate_run_directory(self, run_directory: str, parameter_sample: Dict) -> None:
        os.makedirs(run_directory, exist_ok=True)
        with open(os.path.join(run_directory, "params.txt"), "w", encoding="utf-8") as handle:
            for key, value in parameter_sample.items():
                handle.write(f"{key}: {value}\n")

    def run_model(self, run_directory: str, parameter_sample: Dict) -> int:
        model = solid_dynamics.cantilever_model(
            length=self.length,
            height=self.height,
            nx=self.nx,
            ny=self.ny,
            young_modulus=float(parameter_sample["young_modulus"]),
            poisson_ratio=self.poisson_ratio,
            density=self.density,
            mass_type="consistent",
            material_model=self.material_model,
        )
        load_amplitude = float(parameter_sample["load_amplitude"])
        pulse_duration = float(parameter_sample["pulse_duration"])
        edge_shape = model.boundary_force_x(model.mesh.length, np.array([0.0, -1.0]))

        def external_force(time: float) -> np.ndarray:
            if 0.0 <= time <= pulse_duration:
                scale = load_amplitude * np.sin(np.pi * time / pulse_duration)
            else:
                scale = 0.0
            return scale * edge_shape

        state = model.initial_state(external_force=external_force)
        num_steps = int(round(self.t_end / self.dt))
        times, displacements, velocities = model.solve_implicit(
            state,
            dt=self.dt,
            num_steps=num_steps,
            external_force=external_force,
            snapshot_stride=self.snapshot_stride,
        )
        tip_node = model.mesh.node_nearest(self.length, 0.5 * self.height)
        qoi = displacements[:, 2 * tip_node + 1]
        np.savez(
            os.path.join(run_directory, "solution.npz"),
            times=times,
            displacements=displacements,
            velocities=velocities,
            qoi=qoi,
        )
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: Dict) -> np.ndarray:
        del parameter_sample
        return np.load(os.path.join(run_directory, "solution.npz"))["qoi"]


def _main() -> None:
    run_directory = os.path.join(os.path.dirname(__file__), "nonlinear_solid_output")
    model = NonlinearSolidBeamQoiModel()
    sample = {
        "young_modulus": 1.0e6,
        "load_amplitude": 1000.0,
        "pulse_duration": 0.15,
    }
    model.populate_run_directory(run_directory, sample)
    model.run_model(run_directory, sample)
    qoi = model.compute_qoi(run_directory, sample)
    print(f"Saved {os.path.join(run_directory, 'solution.npz')}")
    print(f"QoI snapshots: {qoi.size}; minimum tip displacement: {qoi.min():.6e} m")


if __name__ == "__main__":
    _main()
