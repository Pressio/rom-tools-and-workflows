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
