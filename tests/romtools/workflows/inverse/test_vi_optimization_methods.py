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
)


def test_diagonal_newton_step_uses_raw_diagonal_for_matrix_input():
    gradient = np.array([2.0, -3.0])
    hessian = np.array([[0.0, 10.0], [10.0, -2.0]])
    solver = NewtonSolver(regularization=0.5, hessian_type='diagonal')

    matrix_step = solver.step(gradient, hessian)
    diagonal_step = solver.step(gradient, np.diag(hessian))

    expected = gradient / np.array([0.5, 2.0])
    np.testing.assert_allclose(matrix_step, expected)
    np.testing.assert_allclose(diagonal_step, expected)


def test_full_newton_step_retains_spectral_absolute_value_projection():
    gradient = np.array([1.0, 2.0])
    hessian = np.array([[0.0, 1.0], [1.0, 0.0]])
    solver = NewtonSolver(regularization=0.1, hessian_type='full')

    np.testing.assert_allclose(solver.step(gradient, hessian), gradient)


@pytest.mark.parametrize('hessian_type', ['diagonal', 'full'])
def test_newton_step_can_floor_eigenvalues_relative_to_spectral_norm(hessian_type):
    gradient = np.array([8.0, 8.0])
    diagonal = np.array([4.0, 1.0])
    hessian = diagonal if hessian_type == 'diagonal' else np.diag(diagonal)
    solver = NewtonSolver(
        regularization=0.5,
        hessian_type=hessian_type,
        regularization_strategy='hessian_norm',
    )

    step = solver.step(gradient, hessian)
    np.testing.assert_allclose(step, np.array([2.0, 4.0]))
    np.testing.assert_allclose(solver.step(gradient, 10.0 * hessian), step / 10.0)


@pytest.mark.parametrize('hessian_type', ['diagonal', 'full'])
def test_hessian_norm_regularization_safeguards_zero_hessian(hessian_type):
    hessian = np.zeros(2) if hessian_type == 'diagonal' else np.zeros((2, 2))
    solver = NewtonSolver(
        regularization=0.5,
        hessian_type=hessian_type,
        regularization_strategy='hessian_norm',
    )

    assert np.all(np.isfinite(solver.step(np.ones(2), hessian)))


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

    np.testing.assert_allclose(mean_step, np.array([0.25]))
    np.testing.assert_allclose(log_std_step, np.array([0.0625]))


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
    with pytest.raises(ValueError, match='newton_regularization_strategy'):
        _normalize_newton_regularization_strategy('unknown')


def test_exponential_hessian_average():
    previous = np.array([2.0, 4.0])
    current = np.array([10.0, -2.0])
    np.testing.assert_allclose(
        _average_hessians(previous, current, 0.75),
        np.array([4.0, 2.5]),
    )
