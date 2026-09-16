import numpy as np
import pytest

from romtools.workflows.inverse.vi_jackknife_regularization import (
    VIJackknifeRegularizationConfig,
    _jackknife_spectral_sigma,
    _project_unregularized_hessian,
    _relative_hessian_uncertainty,
)


def test_jackknife_spectral_sigma_matches_scalar_formula():
    hessians = [
        np.array([[1.0, 0.0], [0.0, 2.0]]),
        np.array([[2.0, 0.0], [0.0, 2.0]]),
        np.array([[3.0, 0.0], [0.0, 2.0]]),
    ]
    sigma = _jackknife_spectral_sigma(hessians)
    expected = np.sqrt((2.0 / 3.0) * (1.0 + 0.0 + 1.0))
    np.testing.assert_allclose(sigma, expected)


def test_unregularized_projection_uses_absolute_eigenvalues():
    hessian = np.array([[-2.0, 0.0], [0.0, 0.5]])
    projected = _project_unregularized_hessian(hessian)
    np.testing.assert_allclose(projected, np.diag([2.0, 0.5]))


def test_relative_hessian_uncertainty_is_dimensionless_ratio():
    np.testing.assert_allclose(_relative_hessian_uncertainty(0.25, 2.0), 0.125)


def test_jackknife_regularization_config_validation():
    with pytest.raises(ValueError, match="scale"):
        VIJackknifeRegularizationConfig(scale=-1.0)
    with pytest.raises(ValueError, match="maximum"):
        VIJackknifeRegularizationConfig(minimum=1.0e-2, maximum=1.0e-3)
