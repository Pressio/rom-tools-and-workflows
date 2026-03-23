import numpy as np

from romtools.workflows.inverse.vi_optimization_methods import (
    AdaGradSolver,
    NewtonSolver,
    SteepestDescentSolver,
)


def test_steepest_descent_solver_returns_gradient_direction():
    gradient = np.array([1.0, -2.0, 0.5, 0.25])
    solver = SteepestDescentSolver()
    step = solver.step(gradient)
    assert np.allclose(step, gradient)


def test_newton_solver_regularizes_hessian_diagonal():
    gradient = np.array([2.0, 4.0])
    hessian_diagonal = np.array([0.0, -2.0])
    solver = NewtonSolver(regularization=1e-2)
    step = solver.step(gradient, hessian_diagonal)
    assert np.allclose(step, np.array([200.0, 2.0]))


def test_newton_solver_can_use_full_hessian():
    gradient = np.array([1.0, 2.0])
    hessian = np.array([
        [2.0, 1.0],
        [1.0, 3.0],
    ])
    solver = NewtonSolver(regularization=1e-2, hessian_type="full")
    step = solver.step(gradient, hessian)
    assert np.allclose(step, np.linalg.solve(hessian, gradient))


def test_adagrad_solver_scales_by_accumulator():
    gradient = np.array([2.0, -3.0])
    accumulator = np.array([4.0, 9.0])
    solver = AdaGradSolver(epsilon=0.0)
    step = solver.step(gradient, accumulator)
    assert np.allclose(step, np.array([1.0, -1.0]))
