"""Tests for the lightweight solid-dynamics example model."""

from pathlib import Path
import sys

import numpy as np

MODELS = Path(__file__).resolve().parents[2] / "examples" / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from solid_dynamics import (  # noqa: E402
    cantilever_model,
    gaussian_displacement,
    longitudinal_wave_model,
    longitudinal_wave_speed,
    two_way_gaussian_solution,
)


def test_undeformed_internal_force_is_zero():
    model = cantilever_model(nx=2, ny=1)
    force = model.internal_force(np.zeros(model.ndof))
    np.testing.assert_allclose(force, 0.0, atol=1.0e-12)


def test_consistent_tangent_matches_directional_finite_difference():
    model = cantilever_model(nx=2, ny=1)
    rng = np.random.default_rng(4)
    displacement = np.zeros(model.ndof)
    displacement[model.free_dofs] = 1.0e-3 * rng.standard_normal(model.free_dofs.size)
    direction = np.zeros(model.ndof)
    direction[model.free_dofs] = rng.standard_normal(model.free_dofs.size)
    direction /= np.linalg.norm(direction)

    tangent = model.tangent_stiffness(displacement)
    epsilon = 1.0e-7
    finite_difference = (
        model.internal_force(displacement + epsilon * direction)
        - model.internal_force(displacement - epsilon * direction)
    ) / (2.0 * epsilon)
    tangent_action = tangent @ direction
    relative_error = np.linalg.norm(finite_difference - tangent_action) / np.linalg.norm(
        finite_difference
    )
    assert relative_error < 1.0e-6


def test_linear_material_is_linear_and_has_constant_tangent():
    model = cantilever_model(nx=2, ny=1, material_model="linear")
    rng = np.random.default_rng(8)
    u1 = np.zeros(model.ndof)
    u2 = np.zeros(model.ndof)
    u1[model.free_dofs] = 1.0e-3 * rng.standard_normal(model.free_dofs.size)
    u2[model.free_dofs] = 1.0e-3 * rng.standard_normal(model.free_dofs.size)

    np.testing.assert_allclose(
        model.internal_force(u1 + u2),
        model.internal_force(u1) + model.internal_force(u2),
        rtol=1.0e-12,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        model.tangent_stiffness(u1),
        model.tangent_stiffness(u2),
        rtol=1.0e-13,
        atol=1.0e-10,
    )


def test_longitudinal_wave_configuration_is_symmetric():
    model = longitudinal_wave_model(nx=20, ny=1, mass_type="lumped")
    initial_displacement = gaussian_displacement(
        model,
        amplitude=1.0e-2,
        width=0.30,
        direction="axial",
    )
    zero_force = lambda _time: np.zeros(model.ndof)
    state = model.initial_state(
        displacement=initial_displacement,
        external_force=zero_force,
    )
    wave_speed = longitudinal_wave_speed(model.material)
    dx = model.mesh.length / model.mesh.nx
    dt = 0.1 * dx / wave_speed
    _times, displacements, _velocities = model.solve_explicit(
        state,
        dt=dt,
        num_steps=10,
        external_force=zero_force,
    )
    bottom_row_ux = displacements[-1, 0 : 2 * (model.mesh.nx + 1) : 2]
    np.testing.assert_allclose(bottom_row_ux, bottom_row_ux[::-1], atol=1.0e-12)


def test_longitudinal_wave_converges_to_analytic_solution():
    """Check h/dt convergence to the d'Alembert two-way Gaussian solution."""
    length = 4.0
    amplitude = 2.0e-2
    width = 0.40
    center = 0.5 * length
    t_end = 5.0e-3
    cfl = 0.10
    nx_values = (32, 64, 128)
    errors = []
    spacings = []

    for nx in nx_values:
        model = longitudinal_wave_model(
            length=length,
            nx=nx,
            ny=1,
            mass_type="lumped",
        )
        wave_speed = longitudinal_wave_speed(model.material)
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

        dx = length / nx
        dt_target = cfl * dx / wave_speed
        num_steps = int(np.ceil(t_end / dt_target))
        dt = t_end / num_steps
        times, displacements, _velocities = model.solve_explicit(
            state,
            dt=dt,
            num_steps=num_steps,
            external_force=zero_force,
            snapshot_stride=num_steps,
        )

        x = model.mesh.coordinates[: nx + 1, 0]
        numerical = displacements[-1, 0 : 2 * (nx + 1) : 2]
        exact = two_way_gaussian_solution(
            x,
            times[-1],
            amplitude=amplitude,
            width=width,
            center=center,
            wave_speed=wave_speed,
        )
        error = np.sqrt(dx * np.sum((numerical - exact) ** 2))
        exact_norm = np.sqrt(dx * np.sum(exact**2))
        errors.append(error / exact_norm)
        spacings.append(dx)

    errors = np.asarray(errors)
    spacings = np.asarray(spacings)
    rates = np.log(errors[:-1] / errors[1:]) / np.log(
        spacings[:-1] / spacings[1:]
    )

    assert np.all(np.diff(errors) < 0.0)
    assert np.all(rates > 1.25)
    assert errors[-1] < 0.3 * errors[0]


def test_velocity_primary_newmark_smoke():
    model = cantilever_model(nx=2, ny=1)
    gravity = lambda _time: model.body_force(np.array([0.0, -1.0]))
    state = model.initial_state(external_force=gravity)
    times, displacements, velocities = model.solve_implicit(
        state,
        dt=2.0e-3,
        num_steps=5,
        external_force=gravity,
        newton_tolerance=1.0e-9,
    )
    assert times[-1] == 1.0e-2
    assert np.all(np.isfinite(displacements))
    assert np.all(np.isfinite(velocities))
    np.testing.assert_allclose(displacements[-1, model.constrained_dofs], 0.0)
    np.testing.assert_allclose(velocities[-1, model.constrained_dofs], 0.0)


def test_stickle_cantilever_first_minimum_regression():
    """Check the first minimum of the published 4 m x 1 m gravity benchmark.

    Stickle et al. (Computational Mechanics 69, 639--660, 2022) report a
    lower-right displacement near -3.3 m at about 1 s.  This Q4 calculation
    uses a coarser time step than the published benchmark so the tolerance is
    intentionally broad; the full dt=1e-3 reproduction lives in the example.
    """
    model = cantilever_model(
        length=4.0,
        height=1.0,
        nx=8,
        ny=2,
        young_modulus=1.0e6,
        poisson_ratio=0.3,
        density=1050.0,
        mass_type="consistent",
    )
    gravity = lambda _time: model.body_force(np.array([0.0, -10.0]))
    state = model.initial_state(external_force=gravity)
    times, displacements, _velocities = model.solve_implicit(
        state,
        dt=1.0e-2,
        num_steps=120,
        external_force=gravity,
        newton_tolerance=1.0e-8,
    )
    node_a = model.mesh.node_nearest(4.0, 0.0)
    response = displacements[:, 2 * node_a + 1]
    i_min = int(np.argmin(response))
    assert -3.6 < response[i_min] < -2.8
    assert 0.8 < times[i_min] < 1.2
