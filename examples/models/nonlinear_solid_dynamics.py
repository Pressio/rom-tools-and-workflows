"""Lightweight 2-D nonlinear solid dynamics reference model.

This module implements a structured Q4 total-Lagrangian finite-element model
with a compressible Neo-Hookean material.  It is intentionally compact and is
meant for romtools examples, not production structural analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.linalg

Array = np.ndarray


@dataclass(frozen=True)
class Material:
    """Compressible Neo-Hookean material in plane strain."""

    young_modulus: float
    poisson_ratio: float
    density: float

    @property
    def lame_lambda(self) -> float:
        e = float(self.young_modulus)
        nu = float(self.poisson_ratio)
        return e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @property
    def lame_mu(self) -> float:
        e = float(self.young_modulus)
        nu = float(self.poisson_ratio)
        return e / (2.0 * (1.0 + nu))


@dataclass
class SolidState:
    displacement: Array
    velocity: Array
    acceleration: Array
    time: float = 0.0


class StructuredQ4Mesh:
    """Uniform rectangular Q4 mesh."""

    def __init__(self, length: float, height: float, nx: int, ny: int):
        if nx < 1 or ny < 1:
            raise ValueError("nx and ny must both be positive")
        self.length = float(length)
        self.height = float(height)
        self.nx = int(nx)
        self.ny = int(ny)

        xs = np.linspace(0.0, self.length, self.nx + 1)
        ys = np.linspace(0.0, self.height, self.ny + 1)
        self.coordinates = np.array([[x, y] for y in ys for x in xs], dtype=float)

        elements = []
        stride = self.nx + 1
        for j in range(self.ny):
            for i in range(self.nx):
                n0 = j * stride + i
                n1 = n0 + 1
                n3 = n0 + stride
                n2 = n3 + 1
                elements.append([n0, n1, n2, n3])
        self.elements = np.asarray(elements, dtype=int)

    @property
    def num_nodes(self) -> int:
        return self.coordinates.shape[0]

    @property
    def num_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def ndof(self) -> int:
        return 2 * self.num_nodes

    def nodes_on_x(self, x: float, atol: float = 1.0e-12) -> Array:
        return np.flatnonzero(np.isclose(self.coordinates[:, 0], x, atol=atol, rtol=0.0))

    def node_nearest(self, x: float, y: float) -> int:
        point = np.array([x, y], dtype=float)
        return int(np.argmin(np.linalg.norm(self.coordinates - point[None, :], axis=1)))


def _shape_q4(xi: float, eta: float) -> tuple[Array, Array]:
    n = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )
    dndxi = 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [+(1.0 - eta), -(1.0 + xi)],
            [+(1.0 + eta), +(1.0 + xi)],
            [-(1.0 + eta), +(1.0 - xi)],
        ]
    )
    return n, dndxi


_GAUSS = 1.0 / np.sqrt(3.0)
_GAUSS_POINTS = ((-_GAUSS, -_GAUSS), (_GAUSS, -_GAUSS), (_GAUSS, _GAUSS), (-_GAUSS, _GAUSS))


class NonlinearSolid2D:
    """Total-Lagrangian Q4 finite-element solid with plane-strain kinematics."""

    def __init__(
        self,
        mesh: StructuredQ4Mesh,
        material: Material,
        constrained_dofs: Optional[Array] = None,
        mass_type: str = "consistent",
        rayleigh_alpha: float = 0.0,
        rayleigh_beta: float = 0.0,
    ):
        self.mesh = mesh
        self.material = material
        self.constrained_dofs = np.unique(
            np.asarray([] if constrained_dofs is None else constrained_dofs, dtype=int)
        )
        all_dofs = np.arange(mesh.ndof, dtype=int)
        self.free_dofs = np.setdiff1d(all_dofs, self.constrained_dofs, assume_unique=True)
        if mass_type not in ("consistent", "lumped"):
            raise ValueError("mass_type must be 'consistent' or 'lumped'")
        self.mass_type = mass_type
        self._quadrature_cache = [self._compute_element_quadrature(e) for e in range(mesh.num_elements)]
        self._mass = self._assemble_mass(lumped=(mass_type == "lumped"))
        self.rayleigh_alpha = float(rayleigh_alpha)
        self.rayleigh_beta = float(rayleigh_beta)
        self._initial_tangent = None

    @staticmethod
    def clamp_dofs(mesh: StructuredQ4Mesh, left: bool = True, right: bool = False) -> Array:
        nodes = []
        if left:
            nodes.extend(mesh.nodes_on_x(0.0).tolist())
        if right:
            nodes.extend(mesh.nodes_on_x(mesh.length).tolist())
        nodes = np.unique(np.asarray(nodes, dtype=int))
        dofs = np.empty(2 * nodes.size, dtype=int)
        dofs[0::2] = 2 * nodes
        dofs[1::2] = 2 * nodes + 1
        return np.sort(dofs)

    @property
    def ndof(self) -> int:
        return self.mesh.ndof

    def mass_matrix(self) -> Array:
        return self._mass.copy()

    def damping_matrix(self) -> Array:
        c = self.rayleigh_alpha * self._mass
        if self.rayleigh_beta != 0.0:
            if self._initial_tangent is None:
                self._initial_tangent = self.tangent_stiffness(np.zeros(self.ndof))
            c = c + self.rayleigh_beta * self._initial_tangent
        return c

    def _element_dofs(self, nodes: Array) -> Array:
        dofs = np.empty(2 * len(nodes), dtype=int)
        dofs[0::2] = 2 * nodes
        dofs[1::2] = 2 * nodes + 1
        return dofs

    def _compute_element_quadrature(self, element_id: int):
        nodes = self.mesh.elements[element_id]
        x = self.mesh.coordinates[nodes]
        data = []
        for xi, eta in _GAUSS_POINTS:
            n, dndxi = _shape_q4(xi, eta)
            jac = x.T @ dndxi
            det_jac = np.linalg.det(jac)
            if det_jac <= 0.0:
                raise ValueError("reference element has non-positive Jacobian")
            dndx = dndxi @ np.linalg.inv(jac)
            data.append((n, dndx, det_jac))
        return tuple(data)

    def _element_quadrature(self, element_id: int):
        return self._quadrature_cache[element_id]

    def _assemble_mass(self, lumped: bool) -> Array:
        m = np.zeros((self.ndof, self.ndof), dtype=float)
        rho = self.material.density
        for e, nodes in enumerate(self.mesh.elements):
            me_scalar = np.zeros((4, 4), dtype=float)
            for n, _dndx, dv in self._element_quadrature(e):
                me_scalar += rho * np.outer(n, n) * dv
            me = np.kron(me_scalar, np.eye(2))
            dofs = self._element_dofs(nodes)
            m[np.ix_(dofs, dofs)] += me
        if lumped:
            m = np.diag(np.sum(m, axis=1))
        return m

    def _first_piola(self, f2: Array) -> tuple[Array, float, Array, float]:
        f = np.eye(3)
        f[:2, :2] = f2
        j = np.linalg.det(f)
        if j <= 0.0:
            raise ValueError("non-positive deformation Jacobian")
        mu = self.material.lame_mu
        lam = self.material.lame_lambda
        finvt = np.linalg.inv(f).T
        a = 0.5 * lam * (j * j - 1.0) - mu
        p = mu * f + a * finvt
        return p[:2, :2], j, finvt, a

    def _first_piola_and_tangent(self, f2: Array) -> tuple[Array, Array]:
        """Return in-plane P and dP/dF for the paper's compressible NH model."""
        p, j, finvt, a = self._first_piola(f2)
        mu = self.material.lame_mu
        lam = self.material.lame_lambda
        eye = np.eye(3)
        amat = (
            mu * np.einsum("ik,JL->iJkL", eye, eye)
            + lam * j * j * np.einsum("iJ,kL->iJkL", finvt, finvt)
            - a * np.einsum("kJ,iL->iJkL", finvt, finvt)
        )
        return p, amat[:2, :2, :2, :2]

    def _element_force_and_tangent(self, element_id: int, u_local: Array) -> tuple[Array, Array]:
        u_local = np.asarray(u_local, dtype=float).reshape(4, 2)
        fe = np.zeros((4, 2), dtype=float)
        ke4 = np.zeros((4, 2, 4, 2), dtype=float)
        for _n, grad_n, dv in self._element_quadrature(element_id):
            grad_u = u_local.T @ grad_n
            f2 = np.eye(2) + grad_u
            p, amat = self._first_piola_and_tangent(f2)
            fe += (grad_n @ p.T) * dv
            ke4 += np.einsum("aJ,iJkL,bL->aibk", grad_n, amat, grad_n) * dv
        return fe.reshape(-1), ke4.reshape(8, 8)

    def element_internal_force(self, element_id: int, u_local: Array) -> Array:
        u_local = np.asarray(u_local, dtype=float).reshape(4, 2)
        fe = np.zeros((4, 2), dtype=float)
        for _n, grad_n, dv in self._element_quadrature(element_id):
            grad_u = u_local.T @ grad_n
            p, _j, _finvt, _a = self._first_piola(np.eye(2) + grad_u)
            fe += (grad_n @ p.T) * dv
        return fe.reshape(-1)

    def element_tangent_stiffness(self, element_id: int, u_local: Array) -> Array:
        return self._element_force_and_tangent(element_id, u_local)[1]

    def internal_force_and_tangent(self, displacement: Array) -> tuple[Array, Array]:
        u = np.asarray(displacement, dtype=float)
        fint = np.zeros(self.ndof, dtype=float)
        k = np.zeros((self.ndof, self.ndof), dtype=float)
        for e, nodes in enumerate(self.mesh.elements):
            dofs = self._element_dofs(nodes)
            fe, ke = self._element_force_and_tangent(e, u[dofs])
            fint[dofs] += fe
            k[np.ix_(dofs, dofs)] += ke
        return fint, k

    def internal_force(self, displacement: Array) -> Array:
        u = np.asarray(displacement, dtype=float)
        fint = np.zeros(self.ndof, dtype=float)
        for e, nodes in enumerate(self.mesh.elements):
            dofs = self._element_dofs(nodes)
            fint[dofs] += self.element_internal_force(e, u[dofs])
        return fint

    def tangent_stiffness(self, displacement: Array) -> Array:
        return self.internal_force_and_tangent(displacement)[1]

    def body_force(self, acceleration: Array) -> Array:
        acc = np.asarray(acceleration, dtype=float)
        if acc.shape != (2,):
            raise ValueError("body acceleration must have shape (2,)")
        nodal_acc = np.tile(acc, self.mesh.num_nodes)
        return self._mass @ nodal_acc

    def nodal_force(self, node: int, force: Array) -> Array:
        out = np.zeros(self.ndof, dtype=float)
        out[2*node:2*node+2] = np.asarray(force, dtype=float)
        return out

    def boundary_force_x(self, x: float, total_force: Array) -> Array:
        """Distribute a resultant force consistently along a vertical boundary."""
        nodes = self.mesh.nodes_on_x(x)
        nodes = nodes[np.argsort(self.mesh.coordinates[nodes, 1])]
        weights = np.zeros(nodes.size)
        y = self.mesh.coordinates[nodes, 1]
        for i in range(nodes.size - 1):
            segment = y[i + 1] - y[i]
            weights[i] += 0.5 * segment
            weights[i + 1] += 0.5 * segment
        weights /= np.sum(weights)
        out = np.zeros(self.ndof)
        force = np.asarray(total_force, dtype=float)
        for node, weight in zip(nodes, weights):
            out[2*node:2*node+2] += weight * force
        return out

    def acceleration(
        self,
        displacement: Array,
        velocity: Array,
        time: float,
        external_force: Callable[[float], Array],
    ) -> Array:
        c = self.damping_matrix()
        rhs = external_force(time) - self.internal_force(displacement) - c @ velocity
        a = np.zeros(self.ndof, dtype=float)
        ff = np.ix_(self.free_dofs, self.free_dofs)
        a[self.free_dofs] = scipy.linalg.solve(self._mass[ff], rhs[self.free_dofs], assume_a="sym")
        return a

    def initial_state(
        self,
        displacement: Optional[Array] = None,
        velocity: Optional[Array] = None,
        external_force: Optional[Callable[[float], Array]] = None,
        time: float = 0.0,
    ) -> SolidState:
        u = np.zeros(self.ndof) if displacement is None else np.array(displacement, dtype=float, copy=True)
        v = np.zeros(self.ndof) if velocity is None else np.array(velocity, dtype=float, copy=True)
        u[self.constrained_dofs] = 0.0
        v[self.constrained_dofs] = 0.0
        if external_force is None:
            external_force = lambda _t: np.zeros(self.ndof)
        a = self.acceleration(u, v, time, external_force)
        return SolidState(u, v, a, float(time))

    def solve_implicit(
        self,
        state: SolidState,
        dt: float,
        num_steps: int,
        external_force: Callable[[float], Array],
        beta: float = 0.25,
        gamma: float = 0.5,
        newton_tolerance: float = 1.0e-9,
        max_newton_iterations: int = 25,
        line_search: bool = True,
        snapshot_stride: int = 1,
    ) -> tuple[Array, Array, Array]:
        """Integrate with velocity-primary Newmark and Newton iteration."""
        u = np.array(state.displacement, copy=True)
        v = np.array(state.velocity, copy=True)
        a = np.array(state.acceleration, copy=True)
        t = float(state.time)
        cmat = self.damping_matrix()
        ff = np.ix_(self.free_dofs, self.free_dofs)
        mff = self._mass[ff]
        cff = cmat[ff]
        cu = beta * dt / gamma

        times = [t]
        displacements = [u.copy()]
        velocities = [v.copy()]

        for istep in range(1, num_steps + 1):
            tnext = t + dt
            v_old = v.copy()
            u_old = u.copy()
            a_old = a.copy()
            v_trial = v_old + dt * a_old
            v_trial[self.constrained_dofs] = 0.0

            def kinematics(v_candidate):
                anew = (
                    (v_candidate - v_old) / (gamma * dt)
                    - ((1.0 - gamma) / gamma) * a_old
                )
                unew = (
                    u_old
                    + dt * v_old
                    + dt * dt * ((0.5 - beta) * a_old + beta * anew)
                )
                unew[self.constrained_dofs] = 0.0
                anew[self.constrained_dofs] = 0.0
                return unew, anew

            converged = False
            fext = external_force(tnext)
            for _iteration in range(max_newton_iterations):
                u_trial, a_trial = kinematics(v_trial)
                fint, kt = self.internal_force_and_tangent(u_trial)
                inertial = self._mass @ a_trial
                damping = cmat @ v_trial
                residual = inertial + damping + fint - fext
                rfree = residual[self.free_dofs]
                scale = max(
                    np.linalg.norm(fext[self.free_dofs]),
                    np.linalg.norm(inertial[self.free_dofs]),
                    np.linalg.norm(fint[self.free_dofs]),
                    1.0,
                )
                rnorm = np.linalg.norm(rfree)
                if rnorm <= newton_tolerance * scale:
                    converged = True
                    break

                jac = mff / (gamma * dt) + cff + cu * kt[ff]
                dv = scipy.linalg.solve(jac, -rfree, assume_a="sym")

                alpha = 1.0
                if line_search:
                    base = rnorm
                    for _ in range(10):
                        candidate = v_trial.copy()
                        candidate[self.free_dofs] += alpha * dv
                        candidate[self.constrained_dofs] = 0.0
                        uc, ac = kinematics(candidate)
                        rc = (
                            self._mass @ ac
                            + cmat @ candidate
                            + self.internal_force(uc)
                            - fext
                        )[self.free_dofs]
                        if np.linalg.norm(rc) < base:
                            break
                        alpha *= 0.5
                v_trial[self.free_dofs] += alpha * dv
                v_trial[self.constrained_dofs] = 0.0

            if not converged:
                raise RuntimeError(f"Newton failed to converge at step {istep}, t={tnext:.6e}")

            u, a = kinematics(v_trial)
            v = v_trial
            t = tnext
            if istep % snapshot_stride == 0 or istep == num_steps:
                times.append(t)
                displacements.append(u.copy())
                velocities.append(v.copy())

        return np.asarray(times), np.asarray(displacements), np.asarray(velocities)

    def solve_explicit(
        self,
        state: SolidState,
        dt: float,
        num_steps: int,
        external_force: Callable[[float], Array],
        snapshot_stride: int = 1,
    ) -> tuple[Array, Array, Array]:
        """Integrate with velocity Verlet."""
        u = np.array(state.displacement, copy=True)
        v = np.array(state.velocity, copy=True)
        a = np.array(state.acceleration, copy=True)
        t = float(state.time)
        times = [t]
        displacements = [u.copy()]
        velocities = [v.copy()]

        for istep in range(1, num_steps + 1):
            vhalf = v + 0.5 * dt * a
            unew = u + dt * vhalf
            unew[self.constrained_dofs] = 0.0
            anew = self.acceleration(unew, vhalf, t + dt, external_force)
            vnew = vhalf + 0.5 * dt * anew
            vnew[self.constrained_dofs] = 0.0
            u, v, a = unew, vnew, anew
            t += dt
            if istep % snapshot_stride == 0 or istep == num_steps:
                times.append(t)
                displacements.append(u.copy())
                velocities.append(v.copy())
        return np.asarray(times), np.asarray(displacements), np.asarray(velocities)

    def gaussian_transverse_displacement(
        self, amplitude: float, width: float, center: Optional[float] = None
    ) -> Array:
        if center is None:
            center = 0.5 * self.mesh.length
        x = self.mesh.coordinates[:, 0]
        profile = float(amplitude) * np.exp(-0.5 * ((x - center) / float(width)) ** 2)
        u = np.zeros(self.ndof)
        u[1::2] = profile
        u[self.constrained_dofs] = 0.0
        return u

    def gaussian_transverse_velocity(
        self, amplitude: float, width: float, center: Optional[float] = None
    ) -> Array:
        return self.gaussian_transverse_displacement(amplitude, width, center)


def cantilever_model(
    length: float = 4.0,
    height: float = 1.0,
    nx: int = 8,
    ny: int = 2,
    young_modulus: float = 1.0e6,
    poisson_ratio: float = 0.3,
    density: float = 1050.0,
    mass_type: str = "consistent",
) -> NonlinearSolid2D:
    mesh = StructuredQ4Mesh(length, height, nx, ny)
    material = Material(young_modulus, poisson_ratio, density)
    constrained = NonlinearSolid2D.clamp_dofs(mesh, left=True, right=False)
    return NonlinearSolid2D(mesh, material, constrained, mass_type=mass_type)


def doubly_clamped_model(
    length: float = 4.0,
    height: float = 1.0,
    nx: int = 16,
    ny: int = 4,
    young_modulus: float = 1.0e6,
    poisson_ratio: float = 0.3,
    density: float = 1050.0,
    mass_type: str = "consistent",
) -> NonlinearSolid2D:
    mesh = StructuredQ4Mesh(length, height, nx, ny)
    material = Material(young_modulus, poisson_ratio, density)
    constrained = NonlinearSolid2D.clamp_dofs(mesh, left=True, right=True)
    return NonlinearSolid2D(mesh, material, constrained, mass_type=mass_type)
