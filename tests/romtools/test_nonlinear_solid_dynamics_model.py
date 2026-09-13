"""Tests for the lightweight nonlinear solid-dynamics example model."""

from pathlib import Path
import sys

import numpy as np

MODELS = Path(__file__).resolve().parents[2] / "examples" / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from nonlinear_solid_dynamics import cantilever_model  # noqa: E402


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
