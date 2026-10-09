import numpy as np
import pytest

from romtools.workflows.inverse.vi_optimization_methods import (
    NewtonSolver,
    _normalize_newton_curvature_strategy,
    _normalize_newton_regularization_strategy,
)
from romtools.workflows.inverse.vi_drivers import (
    _average_hessians,
    _compute_newton_step,
    _restore_adaptive_newton_regularization,
    _update_adaptive_newton_controls,
)


def test_diagonal_newton_step_uses_raw_diagonal_for_matrix_input():
    gradient = np.array([2.0, -3.0])
    hessian = np.array([[0.0, 10.0], [10.0, -2.0]])
    solver = NewtonSolver(regularization=0.5, hessian_type='diagonal')

    matrix_step = solver.step(gradient, hessian)
    diagonal_step = solver.step(gradient, np.diag(hessian))

    expected = gradient / np.array([100.0, 2.0])
    np.testing.assert_allclose(matrix_step, expected)
    np.testing.assert_allclose(diagonal_step, expected)


def test_full_newton_step_uses_sign_preserving_curvature_projection():
    gradient = np.array([1.0, 2.0])
    hessian = np.array([[0.0, 1.0], [1.0, 0.0]])
    solver = NewtonSolver(regularization=0.1, hessian_type='full')

    np.testing.assert_allclose(solver.step(gradient, hessian), [-0.485, 0.515])


@pytest.mark.parametrize('hessian_type', ['diagonal', 'full'])
def test_newton_step_uses_tolerance_relative_to_spectral_norm(hessian_type):
    gradient = np.array([8.0, 8.0])
    diagonal = np.array([-4.0, -1.0])
    hessian = diagonal if hessian_type == 'diagonal' else np.diag(diagonal)
    solver = NewtonSolver(
        regularization=0.5,
        hessian_type=hessian_type,
        regularization_strategy='hessian_norm',
    )

    step = solver.step(gradient, hessian)
    np.testing.assert_allclose(step, np.array([2.0, 0.08]))
    np.testing.assert_allclose(solver.step(gradient, 10.0 * hessian), np.array([0.2, 0.08]))


@pytest.mark.parametrize('hessian_type', ['diagonal', 'full'])
def test_hessian_norm_regularization_safeguards_zero_hessian(hessian_type):
    hessian = np.zeros(2) if hessian_type == 'diagonal' else np.zeros((2, 2))
    solver = NewtonSolver(
        regularization=0.5,
        hessian_type=hessian_type,
        regularization_strategy='hessian_norm',
    )

    assert np.all(np.isfinite(solver.step(np.ones(2), hessian)))


@pytest.mark.parametrize('regularization_strategy', ['absolute', 'hessian_norm'])
@pytest.mark.parametrize('hessian_type', ['diagonal', 'full'])
def test_scalar_additive_regularization_is_applied_after_replacement(
        regularization_strategy, hessian_type):
    diagonal = np.array([-4.0, 1.0])
    hessian = diagonal if hessian_type == 'diagonal' else np.diag(diagonal)
    solver = NewtonSolver(
        regularization=0.5,
        hessian_type=hessian_type,
        regularization_strategy=regularization_strategy,
        additive_regularization=0.25,
    )

    floor = 2.0 if regularization_strategy == 'hessian_norm' else 0.5
    expected_curvature = np.where(-diagonal < floor, 100.0, -diagonal) + 0.25
    np.testing.assert_allclose(
        solver.step(expected_curvature, hessian), np.ones(2)
    )


def test_per_parameter_diagonal_regularization_uses_signed_curvature():
    gradient = np.array([6.1, 3.4, 0.4])
    hessian = np.array([-4.0, 3.0, 0.0])
    solver = NewtonSolver(
        regularization=0.1,
        hessian_type='diagonal',
        regularization_strategy='per_parameter',
        additive_regularization=0.5,
        regularization_epsilon=0.2,
    )

    # Replace curvature below 0.1 with 100, then add parameter damping.
    expected_curvature = np.array([6.1, 101.6, 100.1])
    np.testing.assert_allclose(
        solver.step(gradient, hessian), gradient / expected_curvature
    )


def test_per_parameter_full_regularization_replaces_negative_curvature():
    hessian = np.array([[1.0, 2.0], [2.0, 1.0]])
    solver = NewtonSolver(
        regularization=0.25,
        hessian_type='full',
        regularization_strategy='per_parameter',
        additive_regularization=0.5,
        regularization_epsilon=0.5,
    )

    # The [1, 1] mode has raw Hessian eigenvalue 3. Its positive-curvature
    # eigenvalue is replaced with 100, then receives 0.75 diagonal damping.
    np.testing.assert_allclose(solver.step(np.ones(2), hessian), np.ones(2) / 100.75)


@pytest.mark.parametrize('value', [0.0, -1.0, np.inf, np.nan])
def test_regularization_epsilon_must_be_finite_and_positive(value):
    with pytest.raises(ValueError, match='must be finite and positive'):
        NewtonSolver(
            regularization=0.1,
            regularization_strategy='per_parameter',
            regularization_epsilon=value,
        )


@pytest.mark.parametrize('value', [-1.0, -np.inf, np.inf, np.nan])
def test_additive_regularization_must_be_finite_and_nonnegative(value):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        NewtonSolver(regularization=0.1, additive_regularization=value)


def test_natural_newton_regularization_uses_transformed_hessian_norm():
    state = {
        'gradient_mean': np.array([1.0]),
        'gradient_log_std': np.array([1.0]),
        'hessian_diagonal_mean': np.array([4.0]),
        'hessian_diagonal_log_std': np.array([1.0]),
    }
    metric_scale = np.array([2.0, np.sqrt(0.5)])

    mean_step, log_std_step = _compute_newton_step(
        state,
        newton_regularization=0.5,
        newton_regularization_strategy='hessian_norm',
        metric_scale=metric_scale,
    )

    np.testing.assert_allclose(mean_step, np.array([0.04]))
    np.testing.assert_allclose(log_std_step, np.array([0.005]))


def test_natural_per_parameter_regularization_uses_transformed_hessian():
    state = {
        'gradient_mean': np.array([1.0]),
        'gradient_log_std': np.array([1.0]),
        'hessian_diagonal_mean': np.array([-4.0]),
        'hessian_diagonal_log_std': np.array([1.0]),
    }
    metric_scale = np.array([2.0, np.sqrt(0.5)])

    mean_step, log_std_step = _compute_newton_step(
        state,
        newton_regularization=0.1,
        newton_regularization_strategy='per_parameter',
        newton_additive_regularization=0.5,
        newton_regularization_epsilon=0.2,
        metric_scale=metric_scale,
    )

    np.testing.assert_allclose(mean_step, np.array([4.0 / 24.1]))
    np.testing.assert_allclose(log_std_step, np.array([0.5 / 100.35]))


@pytest.mark.parametrize('hessian', [np.zeros((2, 3)), np.zeros((2, 2, 2))])
def test_diagonal_newton_step_rejects_invalid_hessian_shape(hessian):
    solver = NewtonSolver(regularization=0.1, hessian_type='diagonal')

    with pytest.raises(ValueError, match='Hessian must be'):
        solver.step(np.ones(2), hessian)


def test_newton_curvature_strategy_validation():
    assert _normalize_newton_curvature_strategy(' Independent ') == 'independent'
    with pytest.raises(ValueError, match='newton_curvature_strategy'):
        _normalize_newton_curvature_strategy('unknown')


def test_newton_regularization_strategy_validation():
    assert _normalize_newton_regularization_strategy(' Hessian_Norm ') == 'hessian_norm'
    assert _normalize_newton_regularization_strategy(' Per_Parameter ') == 'per_parameter'
    with pytest.raises(ValueError, match='newton_regularization_strategy'):
        _normalize_newton_regularization_strategy('unknown')


def test_adaptive_newton_controls_relax_when_elbo_drops():
    step_size, regularization = _update_adaptive_newton_controls(
        step_size=2.0,
        current_additive_regularization=3.0,
        initial_additive_regularization=1.0,
        accept_step=False,
        elbo_dropped=True,
        step_size_decay_factor=2.0,
        step_size_growth_factor=1.5,
        max_step_size=10.0,
        regularization_increase_factor=5.0,
        regularization_decrease_factor=1.25,
        regularization_max_multiplier=1e4,
    )

    assert step_size == 1.0
    assert regularization == 15.0


def test_adaptive_newton_controls_unchanged_when_elbo_does_not_drop():
    step_size, regularization = _update_adaptive_newton_controls(
        step_size=2.0,
        current_additive_regularization=5.0,
        initial_additive_regularization=1.0,
        accept_step=True,
        elbo_dropped=False,
        step_size_decay_factor=2.0,
        step_size_growth_factor=1.5,
        max_step_size=10.0,
        regularization_increase_factor=5.0,
        regularization_decrease_factor=1.25,
        regularization_max_multiplier=1e4,
    )

    assert step_size == 2.0
    assert regularization == 5.0


def test_adaptive_newton_controls_relax_accepted_elbo_drop():
    step_size, regularization = _update_adaptive_newton_controls(
        2.0, 5.0, 1.0, True, True, 2.0, 1.5, 3.0, 5.0, 1.25, 1e4
    )
    assert step_size == 3.0
    assert regularization == 4.0


def test_adaptive_newton_controls_do_not_reduce_below_initial_regularization():
    _, regularization = _update_adaptive_newton_controls(
        1.0, 1.0, 1.0, True, True, 2.0, 1.5, 3.0, 5.0, 1.25, 1e4
    )
    assert regularization == 1.0


def test_adaptive_newton_regularization_restores_current_value():
    restart_data = {
        'newton_adaptive_regularization': np.array(True),
        'newton_regularization_increase_factor': np.array(5.0),
        'newton_regularization_decrease_factor': np.array(1.25),
        'newton_regularization_max_multiplier': np.array(1e4),
        'current_newton_additive_regularization': np.array(25.0),
    }

    restored = _restore_adaptive_newton_regularization(
        restart_data, True, 1.0, True, 5.0, 1.25, 1e4
    )

    assert restored == 25.0


def test_exponential_hessian_average():
    previous = np.array([2.0, 4.0])
    current = np.array([10.0, -2.0])
    np.testing.assert_allclose(
        _average_hessians(previous, current, 0.75),
        np.array([4.0, 2.5]),
    )


@pytest.mark.parametrize('strategy', ['absolute', 'hessian_norm', 'per_parameter'])
@pytest.mark.parametrize('hessian_type', ['diagonal', 'full'])
def test_gradient_fallback_boundary_and_retained_curvature(strategy, hessian_type):
    curvature = np.array([-2.0, 0.0, 0.49, 0.5, 1.0])
    hessian = -curvature if hessian_type == 'diagonal' else -np.diag(curvature)
    tolerance = 0.25 if strategy == 'hessian_norm' else 0.5
    solver = NewtonSolver(tolerance, hessian_type=hessian_type,
                          regularization_strategy=strategy, fallback_learning_rate=0.02)
    np.testing.assert_allclose(solver.step(np.ones(5), hessian), [0.02, 0.02, 0.02, 2.0, 1.0])


def test_rotated_dense_gradient_fallback():
    rotation = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2.0)
    hessian = -rotation @ np.diag([-3.0, 2.0]) @ rotation.T
    gradient = rotation @ np.array([2.0, 4.0])
    solver = NewtonSolver(0.1, hessian_type='full', fallback_learning_rate=0.03)
    np.testing.assert_allclose(solver.step(gradient, hessian), rotation @ np.array([0.06, 2.0]))


@pytest.mark.parametrize('value', [0.0, -1.0, np.nan, np.inf, np.nextafter(0.0, 1.0)])
def test_fallback_learning_rate_requires_finite_positive_reciprocal(value):
    with pytest.raises(ValueError, match='fallback_learning_rate'):
        NewtonSolver(0.1, fallback_learning_rate=value)


def test_newton_step_passes_custom_fallback_learning_rate():
    state = {'gradient_mean': np.array([2.0]), 'gradient_log_std': np.array([3.0]),
             'hessian_diagonal_mean': np.array([1.0]), 'hessian_diagonal_log_std': np.array([0.0])}
    mean_step, std_step = _compute_newton_step(state, 0.1, newton_fallback_learning_rate=0.02)
    np.testing.assert_allclose(mean_step, [0.04])
    np.testing.assert_allclose(std_step, [0.06])
