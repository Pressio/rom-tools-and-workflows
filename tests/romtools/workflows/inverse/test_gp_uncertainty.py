"""Regression tests for Gaussian-process QoI uncertainty reconstruction."""

import numpy as np
import pytest

from romtools.rom.qoi_surrogates import GaussianProcessQoiModel


@pytest.mark.parametrize("normalize_targets", [False, True])
def test_gp_scalar_std_uses_scale_but_not_offset(normalize_targets):
    parameters = np.array([[0.0], [0.5], [1.0]])
    qois = np.array([2.0, 3.0, 5.0])
    model = GaussianProcessQoiModel(
        parameters, qois, normalize_targets=normalize_targets
    )
    query = np.array([0.25])
    scaled_query = model._parameter_sample_to_array(query)
    mean, scaled_std = model._gps[0].predict_mean_and_std(scaled_query[None, :])
    expected_mean = model._unscale_targets(mean)[0]
    expected_std = scaled_std[0]
    if normalize_targets:
        expected_std *= abs(model._target_scale[0])
    actual_mean, actual_std = model.compute_qoi_and_var("", query)
    np.testing.assert_allclose(actual_mean, [expected_mean], rtol=1e-12)
    np.testing.assert_allclose(actual_std, [expected_std], rtol=1e-12)
    assert actual_std[0] >= 0.0


@pytest.mark.parametrize("normalize_targets", [False, True])
def test_gp_vector_std_squares_pod_modes(normalize_targets):
    parameters = np.array([[0.0], [0.3], [0.7], [1.0]])
    qois = np.array([
        [1.0, 3.0, 4.0],
        [2.0, 2.0, 5.0],
        [3.0, 6.0, 2.0],
        [4.0, 5.0, 7.0],
    ])
    model = GaussianProcessQoiModel(
        parameters, qois, normalize_targets=normalize_targets,
        pod_energy_fraction=1.0,
    )
    query = np.array([0.45])
    scaled_query = model._parameter_sample_to_array(query)
    predictions = [
        gp.predict_mean_and_std(scaled_query[None, :]) for gp in model._gps
    ]
    reduced_mean = np.array([p[0][0] for p in predictions])
    reduced_std = np.array([p[1][0] for p in predictions])
    expected_mean = model._mean_qoi + model._pod_modes @ model._unscale_targets(reduced_mean)
    if normalize_targets:
        reduced_std *= np.abs(model._target_scale)
    expected_std = np.sqrt((model._pod_modes ** 2) @ (reduced_std ** 2))
    actual_mean, actual_std = model.compute_qoi_and_var("", query)
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-12)
    np.testing.assert_allclose(actual_std, expected_std, rtol=1e-12)
    assert np.all(np.isfinite(actual_std))
    assert np.all(actual_std >= 0.0)
