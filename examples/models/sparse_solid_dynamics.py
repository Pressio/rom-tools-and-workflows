"""Sparse linear-algebra helpers for the solid-dynamics reference model.

These helpers preserve the velocity-primary Newmark formulation and element
operators from :mod:`nonlinear_solid_dynamics`, but assemble the Newton tangent
as a sparse matrix and use sparse linear solves.  They are intended for larger
example meshes where the reference model's dense linear algebra dominates the
runtime.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def _sparse_internal_force_and_tangent(model, displacement):
    """Assemble internal force and tangent without forming a dense matrix."""
    u = np.asarray(displacement, dtype=float)
    fint = np.zeros(model.ndof, dtype=float)
    rows = []
    cols = []
    data = []

    for element_id, nodes in enumerate(model.mesh.elements):
        dofs = model._element_dofs(nodes)
        fe, ke = model._element_force_and_tangent(element_id, u[dofs])
        fint[dofs] += fe
        rows.extend(np.repeat(dofs, dofs.size))
        cols.extend(np.tile(dofs, dofs.size))
        data.extend(ke.reshape(-1))

    tangent = scipy.sparse.coo_matrix(
        (np.asarray(data), (np.asarray(rows), np.asarray(cols))),
        shape=(model.ndof, model.ndof),
    ).tocsr()
    tangent.sum_duplicates()
    return fint, tangent


def acceleration_sparse(self, displacement, velocity, time, external_force):
    """Sparse equivalent of ``NonlinearSolid2D.acceleration``."""
    damping = scipy.sparse.csr_matrix(self.damping_matrix())
    mass = scipy.sparse.csr_matrix(self._mass)
    rhs = (
        external_force(time)
        - self.internal_force(displacement)
        - damping @ velocity
    )
    acceleration = np.zeros(self.ndof, dtype=float)
    free = self.free_dofs
    acceleration[free] = scipy.sparse.linalg.spsolve(
        mass[free][:, free],
        rhs[free],
    )
    return acceleration


def solve_implicit_sparse(
    self,
    state,
    dt,
    num_steps,
    external_force,
    beta=0.25,
    gamma=0.5,
    newton_tolerance=1.0e-9,
    max_newton_iterations=25,
    line_search=True,
    snapshot_stride=1,
):
    """Velocity-primary Newmark using sparse Newton linear algebra."""
    u = np.array(state.displacement, copy=True)
    v = np.array(state.velocity, copy=True)
    a = np.array(state.acceleration, copy=True)
    t = float(state.time)

    mass = scipy.sparse.csr_matrix(self._mass)
    damping = scipy.sparse.csr_matrix(self.damping_matrix())
    free = self.free_dofs
    mff = mass[free][:, free]
    cff = damping[free][:, free]
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
            fint, tangent = _sparse_internal_force_and_tangent(self, u_trial)
            inertial = mass @ a_trial
            damping_force = damping @ v_trial
            residual = inertial + damping_force + fint - fext
            rfree = residual[free]
            scale = max(
                np.linalg.norm(fext[free]),
                np.linalg.norm(inertial[free]),
                np.linalg.norm(fint[free]),
                1.0,
            )
            rnorm = np.linalg.norm(rfree)
            if rnorm <= newton_tolerance * scale:
                converged = True
                break

            kff = tangent[free][:, free]
            jacobian = mff / (gamma * dt) + cff + cu * kff
            dv = scipy.sparse.linalg.spsolve(jacobian.tocsc(), -rfree)

            alpha = 1.0
            if line_search:
                base = rnorm
                for _ in range(10):
                    candidate = v_trial.copy()
                    candidate[free] += alpha * dv
                    candidate[self.constrained_dofs] = 0.0
                    uc, ac = kinematics(candidate)
                    rc = (
                        mass @ ac
                        + damping @ candidate
                        + self.internal_force(uc)
                        - fext
                    )[free]
                    if np.linalg.norm(rc) < base:
                        break
                    alpha *= 0.5

            v_trial[free] += alpha * dv
            v_trial[self.constrained_dofs] = 0.0

        if not converged:
            raise RuntimeError(
                f"Newton failed to converge at step {istep}, t={tnext:.6e}"
            )

        u, a = kinematics(v_trial)
        v = v_trial
        t = tnext
        if istep % snapshot_stride == 0 or istep == num_steps:
            times.append(t)
            displacements.append(u.copy())
            velocities.append(v.copy())

    return np.asarray(times), np.asarray(displacements), np.asarray(velocities)


def enable_sparse_implicit_solver(nonlinear_solid_class):
    """Patch a ``NonlinearSolid2D``-compatible class to use sparse solves."""
    nonlinear_solid_class.acceleration = acceleration_sparse
    nonlinear_solid_class.solve_implicit = solve_implicit_sparse
    return nonlinear_solid_class
