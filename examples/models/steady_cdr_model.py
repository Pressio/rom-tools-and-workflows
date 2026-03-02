"""ROM-tools model wrapper for the steady convection-diffusion-reaction equation."""

import os
import numpy as np

try:
    import steady_cdr as cdr
except ImportError:  # pragma: no cover
    from . import steady_cdr as cdr


class SteadyCdrQoiModel:
    """QoiModel-compatible wrapper for the steady CDR equation."""

    def __init__(self, nx: int = 31, ny: int = 31) -> None:
        self.system = cdr.AdvectionDiffusionSystem(nx, ny)

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        with open(os.path.join(run_directory, "params.txt"), "w", encoding="utf-8") as handle:
            for key, value in parameter_sample.items():
                handle.write(f"{key}: {value}\n")

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        b_magnitude = float(parameter_sample["bmag"])
        theta = float(parameter_sample["theta"])
        nu = float(parameter_sample["nu"])
        sigma = float(parameter_sample["sigma"])

        b_vec = np.zeros(2)
        b_vec[0] = b_magnitude * np.cos(theta)
        b_vec[1] = b_magnitude * np.sin(theta)

        state = cdr.solveFom(self.system, b_vec, nu, sigma)

        state_grid = np.zeros((self.system.Nx + 2, self.system.Ny + 2))
        state_grid[1:-1, 1:-1] = state.reshape(self.system.Nx, self.system.Ny)

        # QoI: one-sided estimate of du/dx along the right boundary.
        qoi = -state_grid[:, -2] / self.system.dx
        params = np.array([b_magnitude, theta, nu, sigma], dtype=float)

        np.savez(
            os.path.join(run_directory, "solution.npz"),
            state=state,
            qoi=qoi,
            params=params,
        )
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        solution = np.load(os.path.join(run_directory, "solution.npz"))
        return solution["qoi"]


CdrQoiModel = SteadyCdrQoiModel


def _main():
    run_directory = os.path.join(os.path.dirname(__file__), "steady_cdr_output")
    os.makedirs(run_directory, exist_ok=True)

    model = SteadyCdrQoiModel(nx=21, ny=21)
    sample = {"bmag": 0.5, "theta": np.pi / 3.0, "nu": 1e-3, "sigma": 1.0}
    model.populate_run_directory(run_directory, sample)
    model.run_model(run_directory, sample)
    qoi = model.compute_qoi(run_directory, sample)
    print(f"Saved {os.path.join(run_directory, 'solution.npz')}")
    print(f"QoI shape: {qoi.shape}, QoI L2 norm: {np.linalg.norm(qoi):.6e}")


if __name__ == "__main__":
    _main()
