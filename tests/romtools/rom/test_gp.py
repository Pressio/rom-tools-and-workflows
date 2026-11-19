import numpy as np
import pytest

from romtools.rom.qoi_surrogates import *

def reference_gp_mean_and_std(
    kernel,
    x_train,
    y_train,
    x_query,
    noise_variance,
    jitter,
):
    """
    Independent reference implementation of GP posterior mean and marginal std.
    """
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1, 1)
    x_query = np.asarray(x_query, dtype=float)

    kxx = kernel(x_train, x_train)
    kxx = kxx + (noise_variance + jitter) * np.eye(kxx.shape[0])

    chol = np.linalg.cholesky(kxx)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_train))

    kx = kernel(x_query, x_train)
    kqq = kernel(x_query, x_query)

    mean = kx @ alpha

    v = np.linalg.solve(chol, kx.T)
    cov = kqq - v.T @ v

    # Numerical roundoff can produce tiny negative values on the diagonal.
    var = np.clip(np.diag(cov), 0.0, None)
    std = np.sqrt(var)

    return mean.ravel(), std.ravel()

def test_predict_mean_and_std_single_query_matches_reference():
    kernel = GaussianProcessKernel()
    noise_variance = 1e-8
    jitter = 1e-12

    gp = GaussianProcessRegressorLite(
        kernel=kernel,
        noise_variance=noise_variance,
        jitter=jitter,
    )

    x_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0.0, 1.0, 0.0])
    x_query = np.array([[0.5]])

    gp.fit(x_train, y_train)

    mean, std = gp.predict_mean_and_std(x_query)
    expected_mean, expected_std = reference_gp_mean_and_std(
        kernel,
        x_train,
        y_train,
        x_query,
        noise_variance,
        jitter,
    )

    assert mean.shape == (1,)
    assert std.shape == (1,)

    np.testing.assert_allclose(mean, expected_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(std, expected_std, rtol=1e-12, atol=1e-12)

def test_predict_mean_and_std_vector_query_matches_reference():
    kernel = GaussianProcessKernel()
    noise_variance = 1e-8
    jitter = 1e-12

    gp = GaussianProcessRegressorLite(
        kernel=kernel,
        noise_variance=noise_variance,
        jitter=jitter,
    )

    x_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0.0, 1.0, 0.0])
    x_query = np.array([[0.25], [0.75], [1.50]])

    gp.fit(x_train, y_train)

    mean, std = gp.predict_mean_and_std(x_query)
    expected_mean, expected_std = reference_gp_mean_and_std(
        kernel,
        x_train,
        y_train,
        x_query,
        noise_variance,
        jitter,
    )

    assert mean.shape == (x_query.shape[0],)
    assert std.shape == (x_query.shape[0],)

    np.testing.assert_allclose(mean, expected_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(std, expected_std, rtol=1e-12, atol=1e-12)

def test_predict_mean_and_std_mean_matches_predict():
    kernel = GaussianProcessKernel()

    gp = GaussianProcessRegressorLite(
        kernel=kernel,
        noise_variance=1e-8,
        jitter=1e-12,
    )

    x_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0.0, 1.0, 0.0])
    x_query = np.array([[0.25], [0.75], [1.50]])

    gp.fit(x_train, y_train)

    mean_from_predict = gp.predict(x_query)
    mean, std = gp.predict_mean_and_std(x_query)

    assert mean.shape == mean_from_predict.shape
    assert std.shape == mean_from_predict.shape

    np.testing.assert_allclose(mean, mean_from_predict, rtol=1e-12, atol=1e-12)

def test_predict_mean_and_std_raises_before_fit():
    gp = GaussianProcessRegressorLite(kernel=GaussianProcessKernel())

    with pytest.raises(RuntimeError, match="has not been fit"):
        gp.predict_mean_and_std(np.array([[0.0]]))

if __name__=='__main__':
    test_predict_mean_and_std_single_query_matches_reference()
    test_predict_mean_and_std_vector_query_matches_reference()
    test_predict_mean_and_std_mean_matches_predict()
    test_predict_mean_and_std_raises_before_fit()


def test_ard_kernel_matches_manual_covariance():
    kernel = GaussianProcessKernel(length_scale=[0.2, 2.0], signal_variance=1.5)
    x = np.array([[0.0, 0.0], [0.2, 2.0]])
    expected = 1.5 * np.exp(-1.0)
    np.testing.assert_allclose(kernel(x, x)[0, 1], expected)


def test_scalar_and_repeated_vector_length_scales_agree():
    x = np.array([[0.0, 0.5], [1.0, 2.0], [0.3, 1.0]])
    np.testing.assert_allclose(
        GaussianProcessKernel(length_scale=0.7)(x, x),
        GaussianProcessKernel(length_scale=[0.7, 0.7])(x, x))


@pytest.mark.parametrize("length_scale", [[1.0], [1.0, -2.0], [1.0, np.nan]])
def test_ard_rejects_invalid_length_scales(length_scale):
    with pytest.raises(ValueError, match="length_scale"):
        GaussianProcessKernel(length_scale=length_scale)(
            np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]))


def test_ard_hyperparameter_tuning_fits_anisotropic_function():
    rng = np.random.default_rng(123)
    x = rng.uniform(0, 1, (28, 2))
    y = np.sin(12 * x[:, 0]) + 0.05 * x[:, 1]
    model = GaussianProcessQoiModel(
        x, y, tune_hyperparameters=True, anisotropic=True,
        normalize_parameters=True, normalize_targets=True,
        optimizer_restarts=3)
    scales = np.asarray(model._gps[0].kernel.length_scale)
    assert scales.shape == (2,)
    assert np.all(scales > 0)
    assert scales[0] < scales[1]


def test_ard_kernel_preserves_small_separations_at_large_offsets():
    # Norm-square/dot-product distance formulas incorrectly round this
    # squared separation to zero after division by a small length scale.
    x = np.array([[1e5, 1e5], [1e5 + 1e-3, 1e5]])
    kernel = GaussianProcessKernel(length_scale=1e-3)
    covariance = kernel(x, x)
    expected = np.exp(-0.5)
    np.testing.assert_allclose(covariance[0, 1], expected, rtol=1e-7)
    np.testing.assert_allclose(covariance[1, 0], expected, rtol=1e-7)
    np.testing.assert_allclose(np.diag(covariance), 1.0, rtol=0, atol=1e-14)


def test_ard_kernel_large_offset_with_anisotropic_scales_and_cross_covariance():
    x = np.array([[1e5, -2e5], [1e5 + 1e-3, -2e5 + 2e-3]])
    kernel = GaussianProcessKernel(length_scale=[1e-3, 2e-3], signal_variance=1.5)
    expected = 1.5 * np.exp(-1.0)
    np.testing.assert_allclose(kernel(x[:1], x[1:])[0, 0], expected, rtol=1e-7)
