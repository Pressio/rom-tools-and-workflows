import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLE_MODELS = Path(__file__).resolve().parents[3] / "examples" / "models"
if str(EXAMPLE_MODELS) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_MODELS))

from h2_air_flame import H2AirFlame, extract_temperature_sensors
from h2_air_flame_model import H2AirFlameQoiModel


def _small_model():
    return H2AirFlame(
        nx=8, ny=6, dt=1.0e-4, t_end=2.0e-4, snapshot_stride=1
    )


def test_small_flame_solve_is_finite_and_repeatable():
    model = _small_model()
    parameters = (2.0, 8.0, 40.0, 7.0)
    states_a, times_a = model.solve(*parameters)
    states_b, times_b = model.solve(*parameters)

    assert states_a.shape == (3, 4, 8, 6)
    assert np.all(np.isfinite(states_a))
    assert np.array_equal(times_a, times_b)
    assert np.allclose(states_a, states_b)
    assert times_a[-1] == pytest.approx(2.0e-4)


def test_boundary_conditions_are_enforced():
    model = _small_model()
    states, _ = model.solve(2.0, 8.0, 40.0, 7.0)
    state = states[-1]

    assert np.allclose(state[:, -1, :], state[:, -2, :])
    assert np.allclose(state[:, :, 0], state[:, :, 1])
    assert np.allclose(state[:, :, -1], state[:, :, -2])

    inlet = model._gamma_2
    assert np.allclose(state[0, 0, inlet], model.inlet_h2_mass_fraction)
    assert np.allclose(state[1, 0, inlet], model.inlet_o2_mass_fraction)
    assert np.allclose(state[2, 0, inlet], model.inlet_h2o_mass_fraction)
    assert np.allclose(
        state[3, 0, inlet], model.inlet_temperature / model.temperature_scale
    )


def test_solution_responds_to_all_four_parameters():
    model = H2AirFlame(
        nx=8, ny=6, dt=1.0e-4, t_end=3.0e-4, snapshot_stride=1
    )
    base = [2.0, 8.0, 40.0, 7.0]
    base_states, _ = model.solve(*base)
    base_qoi = extract_temperature_sensors(base_states, spatial_stride=2)

    perturbations = [2.1, 8.4, 42.0, 8.0]
    for parameter_index, perturbed_value in enumerate(perturbations):
        parameters = list(base)
        parameters[parameter_index] = perturbed_value
        states, _ = model.solve(*parameters)
        qoi = extract_temperature_sensors(states, spatial_stride=2)
        assert np.linalg.norm(qoi - base_qoi) > 1.0e-10


def test_crank_nicolson_jacobian_matches_finite_difference():
    model = H2AirFlame(
        nx=6, ny=5, dt=1.0e-4, t_end=1.0e-4, snapshot_stride=1
    )
    kappa, scaled_e, beta_x, beta_y = 2.0, 8.0, 40.0, 7.0
    rng = np.random.default_rng(2)

    state = model.initial_state()
    state[0, 1:-1, 1:-1] = 0.01 + 0.002 * rng.random((4, 3))
    state[1, 1:-1, 1:-1] = 0.10 + 0.01 * rng.random((4, 3))
    state[3, 1:-1, 1:-1] = 2.0 + 0.1 * rng.random((4, 3))
    state = model.apply_boundary_conditions(state)

    previous_rhs = model._linear_rhs(
        state, kappa, beta_x, beta_y
    ) + model._reaction_rhs(state, scaled_e)
    x = state.reshape(-1).copy()
    linear_jacobian = model._build_linear_residual_jacobian(
        kappa, beta_x, beta_y
    )
    jacobian = (
        linear_jacobian
        - 0.5 * model.dt * model._reaction_jacobian(x, scaled_e)
    )

    direction = rng.normal(size=x.size)
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-6
    residual_plus = model._residual(
        x + epsilon * direction,
        state,
        previous_rhs,
        kappa,
        scaled_e,
        beta_x,
        beta_y,
    )
    residual_minus = model._residual(
        x - epsilon * direction,
        state,
        previous_rhs,
        kappa,
        scaled_e,
        beta_x,
        beta_y,
    )
    finite_difference = (residual_plus - residual_minus) / (2.0 * epsilon)

    relative_error = np.linalg.norm(
        finite_difference - jacobian @ direction
    ) / np.linalg.norm(finite_difference)
    assert relative_error < 1.0e-6


def test_qoi_model_wrapper(tmp_path):
    model = H2AirFlameQoiModel(
        nx=8,
        ny=6,
        dt=1.0e-4,
        t_end=2.0e-4,
        snapshot_stride=1,
        spatial_sensor_stride=2,
    )
    sample = {
        "kappa": 2.0,
        "scaled_activation_energy": 8.0,
        "beta_x": 40.0,
        "beta_y": 7.0,
    }

    model.populate_run_directory(str(tmp_path), sample)
    assert model.run_model(str(tmp_path), sample) == 0
    qoi = model.compute_qoi(str(tmp_path), sample)

    assert qoi.ndim == 1
    assert qoi.size > 0
    assert np.all(np.isfinite(qoi))
    with np.load(tmp_path / "solution.npz") as data:
        assert data["states"].shape == (3, 4, 8, 6)
        assert data["parameter_names"].tolist() == list(model.parameter_names)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        H2AirFlame(nx=3)
    with pytest.raises(ValueError):
        H2AirFlame(dt=-1.0e-4)
    with pytest.raises(ValueError):
        H2AirFlame(dt=1.0e-4, t_end=1.5e-4)

    model = _small_model()
    with pytest.raises(ValueError):
        model.solve(-1.0, 8.0, 40.0, 7.0)
    with pytest.raises(ValueError):
        model.solve(2.0, 8.0, -1.0, 7.0)
