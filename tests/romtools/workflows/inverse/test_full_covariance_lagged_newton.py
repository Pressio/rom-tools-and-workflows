import numpy as np

from romtools.workflows.inverse import full_covariance_lagged_newton as lagged
from romtools.workflows.inverse import full_covariance_newton as base
from romtools.workflows.inverse.vi_optimization_methods import VINewtonOptimizerConfig


def _state_with_hessian(hessian):
    return {
        "newton_hessian_full": hessian,
        "natural_gradient_mean": np.zeros(1),
        "natural_gradient_covariance_svec": np.zeros(1),
    }


def test_lagged_hessian_commits_only_accepted_states(monkeypatch):
    config = VINewtonOptimizerConfig(
        newton_curvature_strategy="lagged",
        newton_hessian_averaging_factor=0.25,
    )
    raw_hessians = iter(
        [
            np.eye(2),
            2.0 * np.eye(2),
            3.0 * np.eye(2),
        ]
    )

    def fake_estimator(*args, **kwargs):
        return next(raw_hessians).copy()

    monkeypatch.setattr(base, "estimate_ordinary_hessian", fake_estimator)

    with lagged._accepted_state_hessian_averaging(
        config, "estimate_ordinary_hessian"
    ):
        initial = base.estimate_ordinary_hessian()
        np.testing.assert_allclose(initial, np.eye(2))
        base._natural_gradient_norm(_state_with_hessian(initial))

        rejected = base.estimate_ordinary_hessian()
        np.testing.assert_allclose(rejected, 1.75 * np.eye(2))
        # No gradient-norm call: this line-search candidate is rejected and the
        # running Hessian must remain equal to the initial accepted Hessian.

        accepted = base.estimate_ordinary_hessian()
        np.testing.assert_allclose(accepted, 2.5 * np.eye(2))
        base._natural_gradient_norm(_state_with_hessian(accepted))


def test_full_covariance_lagged_newton_rejects_independent_curvature():
    config = VINewtonOptimizerConfig(newton_curvature_strategy="independent")
    try:
        lagged._validate_newton_config(config)
    except NotImplementedError as error:
        assert "same_sample" in str(error)
        assert "lagged" in str(error)
    else:
        raise AssertionError("Expected independent full-covariance curvature to fail")
