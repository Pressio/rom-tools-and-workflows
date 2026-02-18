"""Steady convection-diffusion-reaction (CDR) operators and solve utilities."""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


class _SparseMatrixBuilder:
    """Incremental sparse matrix builder."""

    def __init__(self, n_rows: int, n_cols: int):
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._row_indices = []
        self._col_indices = []
        self._values = []

    def add_entry(self, row_index: int, col_index: int, value: float) -> None:
        self._row_indices.append(row_index)
        self._col_indices.append(col_index)
        self._values.append(value)

    def assemble(self) -> scipy.sparse.csr_matrix:
        return scipy.sparse.csr_matrix(
            (self._values, (self._row_indices, self._col_indices)),
            shape=(self._n_rows, self._n_cols),
        )


def build_advection_matrices(nx: int, ny: int, dx: float, dy: float):
    """Build first-order advection operators using 2nd-order upwind stencils."""
    ax_builder = _SparseMatrixBuilder(nx * ny, nx * ny)
    ay_builder = _SparseMatrixBuilder(nx * ny, nx * ny)

    for j in range(1, ny):
        for i in range(1, nx):
            index = i + j * nx
            index_im1 = index - 1
            index_im2 = index - 2
            index_jm1 = index - nx
            index_jm2 = index - 2 * nx

            ax_builder.add_entry(index, index, 1.5 / dx)
            ax_builder.add_entry(index, index_im1, -2.0 / dx)
            if i > 1:
                ax_builder.add_entry(index, index_im2, 0.5 / dx)

            ay_builder.add_entry(index, index, 1.5 / dy)
            ay_builder.add_entry(index, index_jm1, -2.0 / dy)
            if j > 1:
                ay_builder.add_entry(index, index_jm2, 0.5 / dy)

    for i in range(1, nx):
        index = i
        index_im1 = index - 1
        index_im2 = index - 2

        ax_builder.add_entry(index, index, 1.5 / dx)
        ax_builder.add_entry(index, index_im1, -2.0 / dx)
        if i > 1:
            ax_builder.add_entry(index, index_im2, 0.5 / dx)
        ay_builder.add_entry(index, index, 1.0 / dy)

    for j in range(1, ny):
        index = j * nx
        index_jm1 = index - nx
        index_jm2 = index - 2 * nx

        ax_builder.add_entry(index, index, 1.0 / dx)
        ay_builder.add_entry(index, index, 1.5 / dy)
        ay_builder.add_entry(index, index_jm1, -2.0 / dy)
        if j > 1:
            ay_builder.add_entry(index, index_jm2, 0.5 / dy)

    ax_builder.add_entry(0, 0, 1.0 / dx)
    ay_builder.add_entry(0, 0, 1.0 / dy)

    return ax_builder.assemble(), ay_builder.assemble()


def build_diffusion_matrix(nx: int, ny: int, dx: float, dy: float):
    """Build Laplacian operator using 2nd-order central stencils."""
    matrix_builder = _SparseMatrixBuilder(nx * ny, nx * ny)
    for j in range(ny):
        for i in range(nx):
            index = i + j * nx
            index_im1 = index - 1
            index_ip1 = index + 1
            index_jm1 = index - nx
            index_jp1 = index + nx

            matrix_builder.add_entry(index, index, -2.0 / dx**2 - 2.0 / dy**2)
            if i > 0:
                matrix_builder.add_entry(index, index_im1, 1.0 / dx**2)
            if i < nx - 1:
                matrix_builder.add_entry(index, index_ip1, 1.0 / dx**2)
            if j > 0:
                matrix_builder.add_entry(index, index_jm1, 1.0 / dy**2)
            if j < ny - 1:
                matrix_builder.add_entry(index, index_jp1, 1.0 / dy**2)
    return matrix_builder.assemble()


def build_qoi_vector(nx: int, ny: int, dx: float, dy: float):
    """Build QoI vector for integral of du/dx along the right boundary."""
    qoi_vector = np.zeros(nx * ny)
    for j in range(ny):
        index = nx - 1 + j * nx
        index_im1 = index - 1
        qoi_vector[index] += -2.0 / dx * dy
        qoi_vector[index_im1] += 0.5 / dx * dy
    return qoi_vector


class AdvectionDiffusionSystem:
    """Container for steady CDR operators on a uniform grid."""

    def __init__(self, nx: int, ny: int):
        self.Lx = 1.0
        self.Ly = 1.0
        self.Nx = nx
        self.Ny = ny
        self.N = nx * ny
        self.dx = self.Lx / (self.Nx + 1)
        self.dy = self.Ly / (self.Ny + 1)
        self.g = np.ones(self.N)
        self.x = np.linspace(self.dx, self.Lx - self.dx, self.Nx)
        self.y = np.linspace(self.dy, self.Ly - self.dy, self.Ny)

        self.A_diffusion = build_diffusion_matrix(self.Nx, self.Ny, self.dx, self.dy)
        self.A_advection_x, self.A_advection_y = build_advection_matrices(
            self.Nx, self.Ny, self.dx, self.dy
        )
        self.I = scipy.sparse.identity(self.N, format="csr")
        self.C = build_qoi_vector(self.Nx, self.Ny, self.dx, self.dy)


def solveFom(system: AdvectionDiffusionSystem, b: np.ndarray, nu: float, sigma: float):
    """Solve the steady CDR system for a given parameter sample."""
    lhs = (
        system.A_diffusion * nu
        - b[0] * system.A_advection_x
        - b[1] * system.A_advection_y
        - sigma * system.I
    )
    rhs = -system.g
    return scipy.sparse.linalg.spsolve(lhs, rhs)

