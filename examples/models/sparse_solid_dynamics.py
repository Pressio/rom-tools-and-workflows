"""Sparse/vectorized linear algebra for the solid-dynamics reference model.

The routines here preserve the Q4 total-Lagrangian element equations and the
velocity-primary Newmark scheme from :mod:`nonlinear_solid_dynamics`.  Element
constitutive evaluations are vectorized across the structured mesh and the
Newton tangent is assembled as a sparse matrix, which makes larger example
meshes practical without changing the discrete problem.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def _assembly_data(model):
    """Cache element DOFs and quadrature data in array form."""
    cached = getattr(model, "_vectorized_sparse_assembly_data", None)
    if cached is not None:
        return cached

    elements = np.asarray(model.mesh.elements, dtype=int)
    num_elements = elements.shape[0]
    dofs = np.empty((num_elements, 8), dtype=int)
    dofs[:, 0::2] = 2 * elements
    dofs[:, 1::2] = 2 * elements + 1

    num_quadrature = len(model._element_quadrature(0))
    gradients = []
    volumes = []
    for q in range(num_quadrature):
        gradients.append(
            np.stack(
                [model._element_quadrature(e)[q][1] for e in range(num_elements)],
                axis=0,
            )
        )
        volumes.append(
            np.asarray(
                [model._element_quadrature(e)[q][2] for e in range(num_elements)],
                dtype=float,
            )
        )

    rows = np.repeat(dofs, 8, axis=1).reshape(-1)
    cols = np.tile(dofs, (1, 8)).reshape(-1)
    cached = (
        dofs,
        tuple(gradients),
        tuple(volumes),
        rows,
        cols,
    )
    model._vectorized_sparse_assembly_data = cached
    return cached


def _batch_piola(model, deformation_gradients):
    """Vectorized in-plane first Piola stress for the existing NH law."""
    f = np.asarray(deformation_gradients, dtype=float)
    jacobian = np.linalg.det(f)
    if np.any(jacobian <= 0.0):
        raise ValueError("non-positive deformation Jacobian")

    mu = model.material.lame_mu
    lam = model.material.lame_lambda
    finvt = np.swapaxes(np.linalg.inv(f), 1, 2)
    coefficient = 0.5 * lam * (jacobian * jacobian - 1.0) - mu
    stress = mu * f + coefficient[:, None, None] * finvt
    return stress, jacobian, finvt, coefficient


def _batch_piola_and_tangent(model, deformation_gradients):
    """Vectorized stress and consistent in-plane material tangent."""
    stress, jacobian, finvt, coefficient = _batch_piola(
        model,
        deformation_gradients,
    )
    mu = model.material.lame_mu
    lam = model.material.lame_lambda
    eye = np.eye(2)
    elastic_identity = mu * np.einsum("ik,JL->iJkL", eye, eye)
    tangent = elastic_identity[None, ...].copy()
    tangent = tangent + (
        lam
        * jacobian[:, None, None, None, None] ** 2
        * np.einsum("eiJ,ekL->eiJkL", finvt, finvt)
    )
    tangent = tangent - (
        coefficient[:, None, None, None, None]
        * np.einsum("ekJ,eiL->eiJkL", finvt, finvt)
    )
    return stress, tangent


def _vectorized_internal_force(model, displacement):
    """Assemble the nonlinear internal force with batched element algebra."""
    u = np.asarray(displacement, dtype=float)
    dofs, gradients, volumes, _rows, _cols = _assembly_data(model)
    element_u = u[dofs].reshape((-1, 4, 2))
    element_force = np.zeros_like(element_u)
    identity = np.eye(2)[None, :, :]

    for grad_n, dv in zip(gradients, volumes):
        grad_u = np.einsum("eai,eaJ->eiJ", element_u, grad_n)
        stress, _jacobian, _finvt, _coefficient = _batch_piola(
            model,
            identity + grad_u,
        )
        element_force += (
            np.einsum("eaJ,eiJ->eai", grad_n, stress)
            * dv[:, None, None]
        )

    fint = np.zeros(model.ndof, dtype=float)
    np.add.at(fint, dofs.reshape(-1), element_force.reshape(-1))
    return fint


def _sparse_internal_force_and_tangent(model, displacement):
    """Vectorized force assembly and sparse consistent tangent assembly."""
    u = np.asarray(displacement, dtype=float)
    dofs, gradients, volumes, rows, cols = _assembly_data(model)
    element_u = u[dofs].reshape((-1, 4, 2))
    num_elements = element_u.shape[0]
    element_force = np.zeros_like(element_u)
    element_tangent = np.zeros((num_elements, 4, 2, 4, 2), dtype=float)
    identity = np.eye(2)[None, :, :]

    for grad_n, dv in zip(gradients, volumes):
        grad_u = np.einsum("eai,eaJ->eiJ", element_u, grad_n)
        stress, material_tangent = _batch_piola_and_tangent(
            model,
            identity + grad_u,
        )
        element_force += (
            np.einsum("eaJ,eiJ->eai", grad_n, stress)
            * dv[:, None, None]
        )
        element_tangent += (
            np.einsum(
                "eaJ,eiJkL,ebL->eaibk",
                grad_n,
                material_tangent,
                grad_n,
            )
            * dv[:, None, None, None, None]
        )

    fint = np.zeros(model.ndof, dtype=float)
    np.add.at(fint, dofs.reshape(-1), element_force.reshape(-1))
    tangent = scipy.sparse.coo_matrix(
        (element_tangent.reshape(-1), (rows, cols)),
        shape=(model.ndof, model.ndof),
    ).tocsr()
    tangent.sum_duplicates()
    return fint, tangent


def internal_force_vectorized(self, displacement):
    return _vectorized_internal_force(self, displacement)


def acceleration_sparse(self, displacement, velocity, time, external_force):
    """Sparse equivalent of ``NonlinearSolid2D.acceleration``."""
    damping = scipy.sparse.csr_matrix(self.damping_matrix())
    mass = scipy.sparse.csr_matrix(self._mass)
    rhs = (
        external_force(time)
        - _vectorized_internal_force(self, displacement)
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
    """Velocity-primary Newmark using vectorized sparse Newton algebra."""
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
                        + _vectorized_internal_force(self, uc)
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
    nonlinear_solid_class.internal_force = internal_force_vectorized
    nonlinear_solid_class.acceleration = acceleration_sparse
    nonlinear_solid_class.solve_implicit = solve_implicit_sparse
    return nonlinear_solid_class
