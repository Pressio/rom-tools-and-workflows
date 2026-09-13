"""Material-model facade for the lightweight solid-dynamics examples.

The nonlinear implementation lives in :mod:`nonlinear_solid_dynamics`.  This
module adds a small-strain isotropic linear-elastic material while reusing the
same mesh, mass matrix, boundary-condition handling, and explicit/implicit time
integrators.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

try:
    import nonlinear_solid_dynamics as nonlinear
except ImportError:  # pragma: no cover
    from . import nonlinear_solid_dynamics as nonlinear

Array = np.ndarray
Material = nonlinear.Material
SolidState = nonlinear.SolidState
StructuredQ4Mesh = nonlinear.StructuredQ4Mesh
NonlinearSolid2D = nonlinear.NonlinearSolid2D


class LinearElasticSolid2D(NonlinearSolid2D):
    """Small-strain isotropic linear-elastic Q4 solid in plane strain."""

    def _elasticity_tensor(self) -> Array:
        lam = self.material.lame_lambda
        mu = self.material.lame_mu
        eye = np.eye(2)
        return (
            lam * np.einsum("ij,kl->ijkl", eye, eye)
            + mu * np.einsum("ik,jl->ijkl", eye, eye)
            + mu * np.einsum("il,jk->ijkl", eye, eye)
        )

    def _cauchy_stress(self, grad_u: Array) -> Array:
        strain = 0.5 * (grad_u + grad_u.T)
        return (
            self.material.lame_lambda * np.trace(strain) * np.eye(2)
            + 2.0 * self.material.lame_mu * strain
        )

    def _element_force_and_tangent(
        self, element_id: int, u_local: Array
    ) -> tuple[Array, Array]:
        u_local = np.asarray(u_local, dtype=float).reshape(4, 2)
        fe = np.zeros((4, 2), dtype=float)
        ke4 = np.zeros((4, 2, 4, 2), dtype=float)
        elasticity = self._elasticity_tensor()
        for _n, grad_n, dv in self._element_quadrature(element_id):
            grad_u = u_local.T @ grad_n
            stress = self._cauchy_stress(grad_u)
            fe += (grad_n @ stress.T) * dv
            ke4 += (
                np.einsum("aJ,iJkL,bL->aibk", grad_n, elasticity, grad_n) * dv
            )
        return fe.reshape(-1), ke4.reshape(8, 8)

    def element_internal_force(self, element_id: int, u_local: Array) -> Array:
        u_local = np.asarray(u_local, dtype=float).reshape(4, 2)
        fe = np.zeros((4, 2), dtype=float)
        for _n, grad_n, dv in self._element_quadrature(element_id):
            grad_u = u_local.T @ grad_n
            stress = self._cauchy_stress(grad_u)
            fe += (grad_n @ stress.T) * dv
        return fe.reshape(-1)

    def acceleration(
        self,
        displacement: Array,
        velocity: Array,
        time: float,
        external_force: Callable[[float], Array],
    ) -> Array:
        """Evaluate acceleration, using diagonal inversion for lumped mass."""
        if self.mass_type != "lumped":
            return super().acceleration(displacement, velocity, time, external_force)
        c = self.damping_matrix()
        rhs = (
            external_force(time)
            - self.internal_force(displacement)
            - c @ velocity
        )
        acceleration = np.zeros(self.ndof, dtype=float)
        mass_diagonal = np.diag(self._mass)
        acceleration[self.free_dofs] = (
            rhs[self.free_dofs] / mass_diagonal[self.free_dofs]
        )
        return acceleration


def _solid_class(material_model: str):
    key = material_model.lower().replace("-", "_")
    if key in ("neo_hookean", "neohookean", "nonlinear"):
        return NonlinearSolid2D
    if key in ("linear", "linear_elastic", "linear_elasticity"):
        return LinearElasticSolid2D
    raise ValueError(
        "material_model must be 'neo_hookean' or 'linear' "
        f"(received {material_model!r})"
    )


def cantilever_model(
    length: float = 4.0,
    height: float = 1.0,
    nx: int = 8,
    ny: int = 2,
    young_modulus: float = 1.0e6,
    poisson_ratio: float = 0.3,
    density: float = 1050.0,
    mass_type: str = "consistent",
    material_model: str = "neo_hookean",
):
    """Construct a left-clamped beam with the requested constitutive model."""
    mesh = StructuredQ4Mesh(length, height, nx, ny)
    material = Material(young_modulus, poisson_ratio, density)
    cls = _solid_class(material_model)
    constrained = cls.clamp_dofs(mesh, left=True, right=False)
    return cls(mesh, material, constrained, mass_type=mass_type)


def doubly_clamped_model(
    length: float = 4.0,
    height: float = 1.0,
    nx: int = 16,
    ny: int = 4,
    young_modulus: float = 1.0e6,
    poisson_ratio: float = 0.3,
    density: float = 1050.0,
    mass_type: str = "consistent",
    material_model: str = "neo_hookean",
):
    """Construct a beam with both ends clamped."""
    mesh = StructuredQ4Mesh(length, height, nx, ny)
    material = Material(young_modulus, poisson_ratio, density)
    cls = _solid_class(material_model)
    constrained = cls.clamp_dofs(mesh, left=True, right=True)
    return cls(mesh, material, constrained, mass_type=mass_type)


def gaussian_displacement(
    model: NonlinearSolid2D,
    amplitude: float,
    width: float,
    center: Optional[float] = None,
    direction: str = "transverse",
) -> Array:
    """Create an axial or transverse Gaussian nodal displacement."""
    if center is None:
        center = 0.5 * model.mesh.length
    x = model.mesh.coordinates[:, 0]
    profile = float(amplitude) * np.exp(-0.5 * ((x - center) / float(width)) ** 2)
    displacement = np.zeros(model.ndof)
    key = direction.lower()
    if key in ("axial", "x", "longitudinal"):
        displacement[0::2] = profile
    elif key in ("transverse", "y"):
        displacement[1::2] = profile
    else:
        raise ValueError("direction must be 'axial' or 'transverse'")
    displacement[model.constrained_dofs] = 0.0
    return displacement


def longitudinal_wave_model(
    length: float = 4.0,
    height: float = 1.0,
    nx: int = 80,
    ny: int = 1,
    young_modulus: float = 1.0e6,
    poisson_ratio: float = 0.3,
    density: float = 1050.0,
    mass_type: str = "lumped",
) -> LinearElasticSolid2D:
    """Construct a 1-D-like plane-strain bar for a scalar wave-equation check.

    All transverse DOFs are constrained and the axial DOFs are fixed at both
    ends.  For fields that are uniform through the height, the continuum
    equations reduce to ``u_tt = c_p**2 u_xx`` with
    ``c_p = sqrt((lambda + 2*mu)/rho)``.
    """
    mesh = StructuredQ4Mesh(length, height, nx, ny)
    material = Material(young_modulus, poisson_ratio, density)
    transverse = np.arange(1, mesh.ndof, 2, dtype=int)
    left = mesh.nodes_on_x(0.0)
    right = mesh.nodes_on_x(mesh.length)
    axial_ends = 2 * np.concatenate((left, right))
    constrained = np.unique(np.concatenate((transverse, axial_ends)))
    return LinearElasticSolid2D(mesh, material, constrained, mass_type=mass_type)


def longitudinal_wave_speed(material: Material) -> float:
    """Plane-strain longitudinal wave speed for ``longitudinal_wave_model``."""
    return float(
        np.sqrt((material.lame_lambda + 2.0 * material.lame_mu) / material.density)
    )


def two_way_gaussian_solution(
    x: Array,
    time: float,
    amplitude: float,
    width: float,
    center: float,
    wave_speed: float,
) -> Array:
    """D'Alembert solution for a stationary Gaussian displacement pulse.

    This whole-line solution is the exact solution of ``u_tt = c**2 u_xx`` for
    ``u(x, 0) = g(x)`` and ``u_t(x, 0) = 0``.  It can be compared directly to
    the clamped-bar solution before the two traveling pulses reach the ends,
    provided the initial Gaussian is negligible at the boundaries.
    """
    x = np.asarray(x, dtype=float)
    sigma = float(width)
    c = float(wave_speed)
    t = float(time)
    left = np.exp(-0.5 * ((x + c * t - center) / sigma) ** 2)
    right = np.exp(-0.5 * ((x - c * t - center) / sigma) ** 2)
    return 0.5 * float(amplitude) * (left + right)
