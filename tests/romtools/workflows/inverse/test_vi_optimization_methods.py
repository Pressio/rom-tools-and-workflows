import numpy as np
import pytest

from romtools.workflows.inverse.vi_optimization_methods import NewtonSolver


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


@pytest.mark.parametrize('hessian', [np.zeros((2, 3)), np.zeros((2, 2, 2))])
def test_diagonal_newton_step_rejects_invalid_hessian_shape(hessian):
    solver = NewtonSolver(regularization=0.1, hessian_type='diagonal')

    with pytest.raises(ValueError, match='Hessian must be'):
        solver.step(np.ones(2), hessian)
