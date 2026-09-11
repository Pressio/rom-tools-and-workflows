"""Pure-Python two-dimensional premixed H2-air flame model.

The implementation is adapted from the four-parameter Zhar CDR benchmark used
in the MFVI paper. It solves a transient advection-diffusion-reaction system
for H2, O2, H2O mass fractions and a nondimensional temperature using
Crank-Nicolson time integration and Newton iterations.
"""

from typing import Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


class H2AirFlame:
    """Two-dimensional premixed H2-air advection-diffusion-reaction model.

    The state ordering is ``[Y_H2, Y_O2, Y_H2O, theta]``, where
    ``theta = T / temperature_scale``. The four model parameters are the
    diffusivity ``kappa``, scaled activation energy ``E / 1000``, and the two
    advection-velocity components ``beta_x`` and ``beta_y``.

    Parameters
    ----------
    nx, ny:
        Number of nodal points in the x and y directions, including boundary
        points.
    dt:
        Time step in seconds.
    t_end:
        Final time in seconds. ``t_end / dt`` must be an integer.
    snapshot_stride:
        Number of time steps between stored snapshots. The initial and final
        states are always stored.
    newton_relative_tolerance, newton_absolute_tolerance:
        Newton convergence tolerances for each implicit time step.
    newton_max_iterations:
        Maximum Newton iterations per time step.
    """

    field_names = ("Y_H2", "Y_O2", "Y_H2O", "theta")
    parameter_names = ("kappa", "scaled_activation_energy", "beta_x", "beta_y")

    def __init__(
        self,
        nx: int = 25,
        ny: int = 13,
        dt: float = 1.0e-4,
        t_end: float = 5.0e-3,
        snapshot_stride: int = 10,
        newton_relative_tolerance: float = 1.0e-6,
        newton_absolute_tolerance: float = 1.0e-10,
        newton_max_iterations: int = 80,
    ) -> None:
        if nx < 4:
            raise ValueError("nx must be at least 4 for the x-upwind stencil")
        if ny < 3:
            raise ValueError("ny must be at least 3")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if t_end <= 0.0:
            raise ValueError("t_end must be positive")
        if snapshot_stride < 1:
            raise ValueError("snapshot_stride must be at least 1")
        if newton_relative_tolerance <= 0.0 or newton_absolute_tolerance < 0.0:
            raise ValueError(
                "Newton tolerances must be nonnegative and rtol must be positive"
            )
        if newton_max_iterations < 1:
            raise ValueError("newton_max_iterations must be at least 1")

        n_steps_float = t_end / dt
        n_steps = int(round(n_steps_float))
        if not np.isclose(n_steps_float, n_steps, rtol=0.0, atol=1.0e-12):
            raise ValueError("t_end must be an integer multiple of dt")

        self.nx = int(nx)
        self.ny = int(ny)
        self.dt = float(dt)
        self.t_end = float(t_end)
        self.snapshot_stride = int(snapshot_stride)
        self.newton_relative_tolerance = float(newton_relative_tolerance)
        self.newton_absolute_tolerance = float(newton_absolute_tolerance)
        self.newton_max_iterations = int(newton_max_iterations)
        self.num_steps = n_steps

        self.length_x = 1.8  # cm
        self.length_y = 0.9  # cm
        self.x = np.linspace(0.0, self.length_x, self.nx)
        self.y = np.linspace(0.0, self.length_y, self.ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.temperature_scale = 300.0  # K
        self.reaction_prefactor = 4.464956898694298e12
        self.molecular_weights = np.array([2.016, 31.9, 18.0])  # g/mol
        self.density = 1.39e-3  # g/cm^3
        self.heat_of_reaction_scaled = 9800.0 / self.temperature_scale
        self.gas_constant_times_temperature_scale = 8.314472 * self.temperature_scale
        self.stoichiometric_coefficients = np.array([2.0, 1.0, -2.0])

        self.ambient_temperature = 300.0
        self.inlet_temperature = 950.0
        self.inlet_h2_mass_fraction = 0.0282
        self.inlet_o2_mass_fraction = 0.2259
        self.inlet_h2o_mass_fraction = 0.0

        self._gamma_1 = self.y >= 0.6
        self._gamma_2 = np.logical_and(self.y > 0.3, self.y < 0.6)
        self._gamma_3 = self.y <= 0.3
        self._reaction_mask = np.ones((self.nx, self.ny))
        self._reaction_mask[0, :] = 0.0
        self._reaction_mask[-1, :] = 0.0
        self._reaction_mask[:, 0] = 0.0
        self._reaction_mask[:, -1] = 0.0

    def initial_state(self) -> np.ndarray:
        """Return the initial state with boundary conditions applied."""
        state = np.zeros((4, self.nx, self.ny))
        state[3, :, :] = self.ambient_temperature / self.temperature_scale
        return self.apply_boundary_conditions(state)

    def apply_boundary_conditions(self, state: np.ndarray) -> np.ndarray:
        """Apply Dirichlet inlet and homogeneous-Neumann outlet/wall conditions."""
        if state.shape != (4, self.nx, self.ny):
            raise ValueError(
                "state must have shape (4, nx, ny); got {}".format(state.shape)
            )

        state[:, :, 0] = state[:, :, 1]
        state[:, :, -1] = state[:, :, -2]
        state[:, -1, :] = state[:, -2, :]

        state[0, 0, self._gamma_1] = 0.0
        state[1, 0, self._gamma_1] = 0.0
        state[2, 0, self._gamma_1] = 0.0
        state[3, 0, self._gamma_1] = self.ambient_temperature / self.temperature_scale

        state[0, 0, self._gamma_2] = self.inlet_h2_mass_fraction
        state[1, 0, self._gamma_2] = self.inlet_o2_mass_fraction
        state[2, 0, self._gamma_2] = self.inlet_h2o_mass_fraction
        state[3, 0, self._gamma_2] = self.inlet_temperature / self.temperature_scale

        state[0, 0, self._gamma_3] = 0.0
        state[1, 0, self._gamma_3] = 0.0
        state[2, 0, self._gamma_3] = 0.0
        state[3, 0, self._gamma_3] = self.ambient_temperature / self.temperature_scale
        return state

    def _validate_parameters(
        self,
        kappa: float,
        scaled_activation_energy: float,
        beta_x: float,
        beta_y: float,
    ) -> Tuple[float, float, float, float]:
        values = np.asarray(
            [kappa, scaled_activation_energy, beta_x, beta_y], dtype=float
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("model parameters must be finite")
        if kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if scaled_activation_energy <= 0.0:
            raise ValueError("scaled_activation_energy must be positive")
        if beta_x < 0.0:
            raise ValueError(
                "beta_x must be nonnegative because the inlet and x-upwind stencil "
                "assume left-to-right flow"
            )
        return tuple(float(value) for value in values)

    @staticmethod
    def _laplacian(state: np.ndarray, dx: float, dy: float) -> np.ndarray:
        result = np.zeros_like(state)
        result[:, 1:-1, :] += (
            state[:, 2:, :] - 2.0 * state[:, 1:-1, :] + state[:, :-2, :]
        ) / dx**2
        result[:, :, 1:-1] += (
            state[:, :, 2:] - 2.0 * state[:, :, 1:-1] + state[:, :, :-2]
        ) / dy**2
        return result

    @staticmethod
    def _x_upwind_derivative(state: np.ndarray, dx: float) -> np.ndarray:
        result = np.zeros_like(state)
        result[:, 1:, :] = (state[:, 1:, :] - state[:, :-1, :]) / dx
        result[:, 2:, :] = (
            1.5 * state[:, 2:, :]
            - 2.0 * state[:, 1:-1, :]
            + 0.5 * state[:, :-2, :]
        ) / dx
        return result

    @staticmethod
    def _y_upwind_derivative(
        state: np.ndarray, dy: float, beta_y: float
    ) -> np.ndarray:
        result = np.zeros_like(state)
        if beta_y >= 0.0:
            result[:, :, 1:] = (state[:, :, 1:] - state[:, :, :-1]) / dy
        else:
            result[:, :, :-1] = (state[:, :, 1:] - state[:, :, :-1]) / dy
        return result

    def _linear_rhs(
        self, state: np.ndarray, kappa: float, beta_x: float, beta_y: float
    ) -> np.ndarray:
        return (
            kappa * self._laplacian(state, self.dx, self.dy)
            - beta_x * self._x_upwind_derivative(state, self.dx)
            - beta_y * self._y_upwind_derivative(state, self.dy, beta_y)
        )

    def _reaction_rhs(
        self, state: np.ndarray, scaled_activation_energy: float
    ) -> np.ndarray:
        theta = state[3]
        if np.any(theta <= 0.0):
            raise FloatingPointError("temperature became nonpositive during the solve")

        rho = self.density
        weights = self.molecular_weights
        activation_energy = scaled_activation_energy * 1.0e3
        exponential = np.exp(
            -activation_energy / (self.gas_constant_times_temperature_scale * theta)
        )
        h2_concentration = rho * state[0] / weights[0]
        o2_concentration = rho * state[1] / weights[1]
        rate = (
            h2_concentration**2
            * o2_concentration
            * self.reaction_prefactor
            * exponential
        )

        reaction = np.zeros_like(state)
        for species in range(3):
            reaction[species] = (
                -self.stoichiometric_coefficients[species]
                * (weights[species] / rho)
                * rate
            )
        reaction[3] = reaction[2] * self.heat_of_reaction_scaled
        return reaction

    def rhs(
        self,
        state: np.ndarray,
        kappa: float,
        scaled_activation_energy: float,
        beta_x: float,
        beta_y: float,
    ) -> np.ndarray:
        """Evaluate the semi-discrete right-hand side."""
        kappa, scaled_activation_energy, beta_x, beta_y = self._validate_parameters(
            kappa, scaled_activation_energy, beta_x, beta_y
        )
        return self._linear_rhs(state, kappa, beta_x, beta_y) + self._reaction_rhs(
            state, scaled_activation_energy
        )

    def _apply_boundary_conditions_to_residual(
        self, state: np.ndarray, residual: np.ndarray
    ) -> np.ndarray:
        state_view = state.reshape((4, self.nx, self.ny))
        residual_view = residual.reshape((4, self.nx, self.ny))

        residual_view[:, :, 0] = state_view[:, :, 1] - state_view[:, :, 0]
        residual_view[:, :, -1] = state_view[:, :, -1] - state_view[:, :, -2]
        residual_view[:, -1, :] = state_view[:, -1, :] - state_view[:, -2, :]

        residual_view[0, 0, self._gamma_1] = state_view[0, 0, self._gamma_1]
        residual_view[1, 0, self._gamma_1] = state_view[1, 0, self._gamma_1]
        residual_view[2, 0, self._gamma_1] = state_view[2, 0, self._gamma_1]
        residual_view[3, 0, self._gamma_1] = (
            state_view[3, 0, self._gamma_1]
            - self.ambient_temperature / self.temperature_scale
        )

        residual_view[0, 0, self._gamma_2] = (
            state_view[0, 0, self._gamma_2] - self.inlet_h2_mass_fraction
        )
        residual_view[1, 0, self._gamma_2] = (
            state_view[1, 0, self._gamma_2] - self.inlet_o2_mass_fraction
        )
        residual_view[2, 0, self._gamma_2] = (
            state_view[2, 0, self._gamma_2] - self.inlet_h2o_mass_fraction
        )
        residual_view[3, 0, self._gamma_2] = (
            state_view[3, 0, self._gamma_2]
            - self.inlet_temperature / self.temperature_scale
        )

        residual_view[0, 0, self._gamma_3] = state_view[0, 0, self._gamma_3]
        residual_view[1, 0, self._gamma_3] = state_view[1, 0, self._gamma_3]
        residual_view[2, 0, self._gamma_3] = state_view[2, 0, self._gamma_3]
        residual_view[3, 0, self._gamma_3] = (
            state_view[3, 0, self._gamma_3]
            - self.ambient_temperature / self.temperature_scale
        )
        return residual_view.reshape(-1)

    def _residual(
        self,
        next_state_flat: np.ndarray,
        previous_state: np.ndarray,
        previous_rhs: np.ndarray,
        kappa: float,
        scaled_activation_energy: float,
        beta_x: float,
        beta_y: float,
    ) -> np.ndarray:
        next_state = next_state_flat.reshape((4, self.nx, self.ny))
        next_rhs = self._linear_rhs(
            next_state, kappa, beta_x, beta_y
        ) + self._reaction_rhs(next_state, scaled_activation_energy)
        residual = (
            next_state.reshape(-1)
            - previous_state.reshape(-1)
            - 0.5
            * self.dt
            * (next_rhs.reshape(-1) + previous_rhs.reshape(-1))
        )
        return self._apply_boundary_conditions_to_residual(next_state_flat, residual)

    def _build_linear_residual_jacobian(
        self, kappa: float, beta_x: float, beta_y: float
    ) -> scipy.sparse.csr_matrix:
        block_size = self.nx * self.ny
        block = scipy.sparse.lil_matrix((block_size, block_size), dtype=float)
        dx2_inv = 1.0 / self.dx**2
        dy2_inv = 1.0 / self.dy**2
        dx_inv = 1.0 / self.dx
        dy_inv = 1.0 / self.dy

        def index(i: int, j: int) -> int:
            return i * self.ny + j

        for i in range(self.nx):
            for j in range(self.ny):
                row = index(i, j)
                if i == 0:
                    block[row, row] = 1.0
                    continue
                if i == self.nx - 1:
                    block[row, row] = 1.0
                    block[row, index(i - 1, j)] = -1.0
                    continue
                if j == 0:
                    block[row, row] = -1.0
                    block[row, index(i, j + 1)] = 1.0
                    continue
                if j == self.ny - 1:
                    block[row, row] = 1.0
                    block[row, index(i, j - 1)] = -1.0
                    continue

                diagonal = 1.0 + self.dt * kappa * (dx2_inv + dy2_inv)
                x_im1 = -0.5 * self.dt * kappa * dx2_inv
                x_im2 = 0.0
                if i == 1:
                    diagonal += 0.5 * self.dt * beta_x * dx_inv
                    x_im1 -= 0.5 * self.dt * beta_x * dx_inv
                else:
                    diagonal += 0.75 * self.dt * beta_x * dx_inv
                    x_im1 -= self.dt * beta_x * dx_inv
                    x_im2 = 0.25 * self.dt * beta_x * dx_inv

                if beta_y >= 0.0:
                    diagonal += 0.5 * self.dt * beta_y * dy_inv
                    y_jm1 = (
                        -0.5 * self.dt * kappa * dy2_inv
                        - 0.5 * self.dt * beta_y * dy_inv
                    )
                    y_jp1 = -0.5 * self.dt * kappa * dy2_inv
                else:
                    diagonal -= 0.5 * self.dt * beta_y * dy_inv
                    y_jm1 = -0.5 * self.dt * kappa * dy2_inv
                    y_jp1 = (
                        -0.5 * self.dt * kappa * dy2_inv
                        + 0.5 * self.dt * beta_y * dy_inv
                    )

                block[row, row] = diagonal
                block[row, index(i + 1, j)] = -0.5 * self.dt * kappa * dx2_inv
                block[row, index(i - 1, j)] = x_im1
                block[row, index(i, j - 1)] = y_jm1
                block[row, index(i, j + 1)] = y_jp1
                if i >= 2:
                    block[row, index(i - 2, j)] = x_im2

        return scipy.sparse.block_diag(
            [block.tocsr() for _ in range(4)], format="csr"
        )

    def _reaction_jacobian(
        self, state_flat: np.ndarray, scaled_activation_energy: float
    ) -> scipy.sparse.csr_matrix:
        state = state_flat.reshape((4, self.nx, self.ny))
        theta = state[3]
        if np.any(theta <= 0.0):
            raise FloatingPointError("temperature became nonpositive during the solve")

        rho = self.density
        weights = self.molecular_weights
        activation_energy = scaled_activation_energy * 1.0e3
        exponential = np.exp(
            -activation_energy / (self.gas_constant_times_temperature_scale * theta)
        )
        h2_concentration = rho * state[0] / weights[0]
        o2_concentration = rho * state[1] / weights[1]

        prefactor = self.reaction_prefactor * exponential
        rate = h2_concentration**2 * o2_concentration * prefactor
        d_rate_d_h2 = (
            2.0
            * (rho / weights[0])
            * h2_concentration
            * o2_concentration
            * prefactor
        )
        d_rate_d_o2 = (rho / weights[1]) * h2_concentration**2 * prefactor
        d_rate_d_theta = rate * activation_energy / (
            self.gas_constant_times_temperature_scale * theta**2
        )

        d_rate_d_h2 *= self._reaction_mask
        d_rate_d_o2 *= self._reaction_mask
        d_rate_d_theta *= self._reaction_mask

        block_size = self.nx * self.ny
        local = np.arange(block_size, dtype=int)
        rows = []
        cols = []
        data = []

        for species in range(4):
            if species < 3:
                coefficient = (
                    -self.stoichiometric_coefficients[species]
                    * weights[species]
                    / rho
                )
            else:
                coefficient = (
                    -self.stoichiometric_coefficients[2]
                    * weights[2]
                    / rho
                    * self.heat_of_reaction_scaled
                )

            row = species * block_size + local
            rows.extend([row, row, row])
            cols.extend(
                [
                    0 * block_size + local,
                    1 * block_size + local,
                    3 * block_size + local,
                ]
            )
            data.extend(
                [
                    coefficient * d_rate_d_h2.reshape(-1),
                    coefficient * d_rate_d_o2.reshape(-1),
                    coefficient * d_rate_d_theta.reshape(-1),
                ]
            )

        return scipy.sparse.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(4 * block_size, 4 * block_size),
        )

    def _newton_solve(
        self,
        previous_state: np.ndarray,
        previous_rhs: np.ndarray,
        linear_jacobian: scipy.sparse.csr_matrix,
        kappa: float,
        scaled_activation_energy: float,
        beta_x: float,
        beta_y: float,
    ) -> np.ndarray:
        iterate = previous_state.reshape(-1).copy()
        residual = self._residual(
            iterate,
            previous_state,
            previous_rhs,
            kappa,
            scaled_activation_energy,
            beta_x,
            beta_y,
        )
        initial_norm = np.linalg.norm(residual)
        target = (
            self.newton_absolute_tolerance
            + self.newton_relative_tolerance * initial_norm
        )
        if initial_norm <= target:
            return iterate.reshape(previous_state.shape)

        for iteration in range(1, self.newton_max_iterations + 1):
            reaction_jacobian = self._reaction_jacobian(
                iterate, scaled_activation_energy
            )
            # Crank-Nicolson contributes one half of the new-state RHS Jacobian.
            jacobian = linear_jacobian - 0.5 * self.dt * reaction_jacobian
            update = scipy.sparse.linalg.spsolve(jacobian, -residual)
            if not np.all(np.isfinite(update)):
                raise RuntimeError(
                    "Newton linear solve produced non-finite values at iteration {}".format(
                        iteration
                    )
                )
            iterate += update
            residual = self._residual(
                iterate,
                previous_state,
                previous_rhs,
                kappa,
                scaled_activation_energy,
                beta_x,
                beta_y,
            )
            residual_norm = np.linalg.norm(residual)
            if not np.isfinite(residual_norm):
                raise RuntimeError(
                    "Newton residual became non-finite at iteration {}".format(iteration)
                )
            if residual_norm <= target:
                return iterate.reshape(previous_state.shape)

        final_relative = np.linalg.norm(residual) / max(
            initial_norm, np.finfo(float).tiny
        )
        raise RuntimeError(
            "Newton solver failed to converge in {} iterations; "
            "final relative residual={:.3e}".format(
                self.newton_max_iterations, final_relative
            )
        )

    def solve(
        self,
        kappa: float,
        scaled_activation_energy: float,
        beta_x: float,
        beta_y: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate the flame model and return ``(states, times)``.

        ``states`` has shape ``(num_snapshots, 4, nx, ny)``. The temperature
        component is nondimensional; multiply ``states[:, 3]`` by
        :attr:`temperature_scale` to obtain kelvin.
        """
        kappa, scaled_activation_energy, beta_x, beta_y = self._validate_parameters(
            kappa, scaled_activation_energy, beta_x, beta_y
        )
        linear_jacobian = self._build_linear_residual_jacobian(
            kappa, beta_x, beta_y
        )

        state = self.initial_state()
        states = [state.copy()]
        times = [0.0]

        for step in range(1, self.num_steps + 1):
            previous_state = state.copy()
            previous_rhs = self._linear_rhs(
                previous_state, kappa, beta_x, beta_y
            ) + self._reaction_rhs(previous_state, scaled_activation_energy)
            state = self._newton_solve(
                previous_state,
                previous_rhs,
                linear_jacobian,
                kappa,
                scaled_activation_energy,
                beta_x,
                beta_y,
            )
            if step % self.snapshot_stride == 0 or step == self.num_steps:
                states.append(state.copy())
                times.append(step * self.dt)

        return np.stack(states, axis=0), np.asarray(times)


def extract_temperature_sensors(
    states: np.ndarray,
    spatial_stride: int = 4,
    temporal_stride: int = 1,
    temperature_scale: float = 1.0,
) -> np.ndarray:
    """Extract a flattened temperature-sensor QoI from a state history.

    Parameters
    ----------
    states:
        State history with shape ``(num_snapshots, 4, nx, ny)``.
    spatial_stride:
        Sample every ``spatial_stride`` grid points in each spatial direction.
    temporal_stride:
        Sample every ``temporal_stride`` stored snapshots.
    temperature_scale:
        Optional multiplicative scale. Use ``300.0`` to convert the default
        nondimensional temperature to kelvin.
    """
    states = np.asarray(states)
    if states.ndim != 4 or states.shape[1] != 4:
        raise ValueError("states must have shape (num_snapshots, 4, nx, ny)")
    if spatial_stride < 1 or temporal_stride < 1:
        raise ValueError("sensor strides must be at least 1")
    if not np.isfinite(temperature_scale) or temperature_scale <= 0.0:
        raise ValueError("temperature_scale must be finite and positive")

    temperature = states[
        ::temporal_stride, 3, ::spatial_stride, ::spatial_stride
    ]
    return (temperature_scale * temperature).reshape(-1)
