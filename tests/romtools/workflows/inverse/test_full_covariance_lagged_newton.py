import numpy as np

from romtools.workflows.inverse.vi import newton as base
from romtools.workflows.inverse.vi.optimization import VINewtonOptimizerConfig


def _state_with_hessian(hessian):
    return {
        "newton_hessian_full": hessian,
        "natural_gradient_mean": np.zeros(1),
        "natural_gradient_covariance_svec": np.zeros(1),
    }


def test_lagged_hessian_commits_only_accepted_states():
    config = VINewtonOptimizerConfig(
        newton_curvature_strategy="lagged",
        newton_hessian_averaging_factor=0.25,
    )
    optimizer = base.FullCovarianceNewton(config)
    initial = optimizer.average_hessian(np.eye(2))
    np.testing.assert_allclose(initial, np.eye(2))
    optimizer.gradient_norm(_state_with_hessian(initial))

    rejected = optimizer.average_hessian(2.0 * np.eye(2))
    np.testing.assert_allclose(rejected, 1.75 * np.eye(2))
    # A rejected candidate never commits its prospective curvature.
    accepted = optimizer.average_hessian(3.0 * np.eye(2))
    np.testing.assert_allclose(accepted, 2.5 * np.eye(2))
    optimizer.gradient_norm(_state_with_hessian(accepted))
    np.testing.assert_allclose(optimizer.running_hessian, accepted)


def test_full_covariance_lagged_newton_rejects_independent_curvature():
    config = VINewtonOptimizerConfig(newton_curvature_strategy="independent")
    try:
        base._validate_newton_config(config)
    except NotImplementedError as error:
        assert "same_sample" in str(error)
        assert "lagged" in str(error)
    else:
        raise AssertionError("Expected independent full-covariance curvature to fail")
