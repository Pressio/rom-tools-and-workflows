"""Tests for jointly trained multifidelity GP and its QoI protocol."""
import numpy as np
import pytest

from romtools.rom.multifidelity import (
    FidelityData, GPLevelConfig, AutoregressiveGPConfig,
    AutoregressiveGPBuilder, AutoregressiveGPQoiModel,
)
from romtools.rom.qoi_surrogates import GaussianProcessKernel
from romtools.workflows.models import (
    MultiFidelityQoiModel, MultiFidelityQoiModelWithErrorEstimate,
)


def make_config(n, tune=False):
    return AutoregressiveGPConfig(
        levels=[
            GPLevelConfig(
                kernel=GaussianProcessKernel(length_scale=0.35,
                                             signal_variance=1.0),
                rho=0.75 if i else 1.0,
                noise_variance=1e-9,
            )
            for i in range(n)
        ],
        tune_hyperparameters=tune,
        optimizer_restarts=2,
    )


def test_two_level_non_nested_gp_and_protocol():
    x0 = np.linspace(0.0, 1.0, 15)[:, None]
    x1 = np.array([0.08, 0.29, 0.61, 0.93])[:, None]
    f0 = np.sin(4 * x0[:, 0])
    f1 = 0.75 * np.sin(4 * x1[:, 0]) + 0.15 * x1[:, 0]
    model = AutoregressiveGPQoiModel(
        [FidelityData(x0, f0), FidelityData(x1, f1)],
        make_config(2), parameter_names=["mu"],
    )
    assert isinstance(model, MultiFidelityQoiModelWithErrorEstimate)
    assert isinstance(model, MultiFidelityQoiModel)
    assert model.num_fidelities == 2
    for fidelity, x, y in [(0, x0, f0), (1, x1, f1)]:
        mean, std = model.predict(x, fidelity=fidelity, return_std=True)
        np.testing.assert_allclose(mean, y, atol=5e-5)
        assert np.all(std >= 0)
        assert np.max(std) < 1e-3
    probe = {"mu": 0.35}
    assert model.compute_qoi("", probe).shape == (1,)
    assert model.compute_qoi("", probe, fidelity=0).shape == (1,)
    assert model.compute_error_estimate("", probe, fidelity=0) >= 0
    assert model.compute_error_estimate("", probe, fidelity=1) >= 0
    np.testing.assert_allclose(
        model.compute_qoi("", probe),
        model.compute_qoi("", probe, fidelity=1),
    )
    with pytest.raises(ValueError, match="out of range"):
        model.predict(x1, fidelity=2)


def test_three_level_cross_fidelity_joint_conditioning():
    x0 = np.array([0., 0.3, 0.6, 1.])[:, None]
    x1 = np.array([0.2, 0.5, 0.8])[:, None]
    x2 = np.array([0.45, 0.9])[:, None]
    data = [
        FidelityData(x0, np.sin(2 * x0[:, 0])),
        FidelityData(x1, 0.7 * np.sin(2 * x1[:, 0]) + x1[:, 0]),
        FidelityData(x2, 0.5 * (0.7 * np.sin(2 * x2[:, 0]) + x2[:, 0])
                     + 0.2 * x2[:, 0]),
    ]
    model = AutoregressiveGPQoiModel(data, make_config(3))
    assert model.num_fidelities == 3
    for fidelity in range(3):
        mean, std = model.predict(np.array([[0.4], [0.7]]),
                                  fidelity=fidelity, return_std=True)
        assert mean.shape == std.shape == (2,)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0)
    # Additional high fidelity observations must update the low-level
    # posterior through the joint cross-fidelity covariance.
    baseline = AutoregressiveGPQoiModel(
        [data[0], data[1], FidelityData(
            np.array([[0.9]]), data[2].qois[-1:])], make_config(3)
    )
    assert abs(model.predict([0.45], fidelity=0)[0]
               - baseline.predict([0.45], fidelity=0)[0]) > 1e-6


def test_builder_and_joint_hyperparameter_tuning():
    x0 = np.linspace(0, 1, 10)[:, None]
    x1 = np.array([0.12, 0.41, 0.77])[:, None]
    data = [
        FidelityData(x0, np.sin(3 * x0[:, 0]), name="low"),
        FidelityData(x1, 1.2 * np.sin(3 * x1[:, 0]), name="high"),
    ]
    model = AutoregressiveGPBuilder(
        config=make_config(2, tune=True), parameter_names=["mu"],
    ).build("/unused", data)
    prediction, std = model.predict(np.array([[0.5]]), return_std=True)
    assert prediction.shape == std.shape == (1,)
    assert np.isfinite(prediction[0]) and np.isfinite(std[0])
    assert all(np.all(np.asarray(k.length_scale) > 0)
               for k in model._kernels)


@pytest.mark.parametrize("bad_data", [
    [FidelityData(np.zeros((3, 1)), np.ones(3))],
    [FidelityData(np.zeros((3, 1)), np.ones(3)),
     FidelityData(np.zeros((2, 2)), np.ones(2))],
    [FidelityData(np.zeros((3, 1)), np.ones((3, 2))),
     FidelityData(np.zeros((2, 1)), np.ones(2))],
])
def test_rejects_incompatible_fidelity_data(bad_data):
    with pytest.raises(ValueError):
        AutoregressiveGPQoiModel(bad_data, make_config(len(bad_data)))
