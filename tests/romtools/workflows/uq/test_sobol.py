import numpy as np
import pytest

from romtools.workflows.parameter_spaces import (
    GaussianParameterSpace,
    MultivariateGaussianParameterSpace,
    UniformParameterSpace,
)
from romtools.workflows.sampling_methods import MonteCarloSampler
from romtools.workflows.uq import run_multifidelity_sobol_indices, run_sobol_indices
from romtools.workflows.uq.multifidelity_sobol import _estimate_multifidelity_indices
from romtools.workflows.uq.sobol import _estimate_indices


class SensitivityModel:
    def populate_run_directory(self, run_directory, parameter_sample):
        pass

    def run_model(self, run_directory, parameter_sample):
        return 0

    def compute_qoi(self, run_directory, parameter_sample):
        x, y, z = (parameter_sample[name] for name in ("x", "y", "z"))
        ishigami = np.sin(x) + 7 * np.sin(y) ** 2 + 0.1 * z ** 4 * np.sin(x)
        return np.array([ishigami, x + 2 * y + 100.0, 0.1])


def parameter_space():
    return UniformParameterSpace(
        ["x", "y", "z"], [-np.pi] * 3, [np.pi] * 3, MonteCarloSampler
    )


@pytest.mark.mpi_skip
def test_sobol_estimates_with_known_interactions(tmp_path):
    result = run_sobol_indices(SensitivityModel(), parameter_space(), tmp_path, 512)

    assert result.parameter_names == ("x", "y", "z")
    assert result.parameter_samples.shape == (512 * 5, 3)
    np.testing.assert_allclose(result.first_order[:, 0], [0.314, 0.442, 0], atol=0.04)
    np.testing.assert_allclose(result.total_effect[:, 0], [0.558, 0.442, 0.244], atol=0.04)
    np.testing.assert_allclose(result.first_order[:, 1], [0.2, 0.8, 0], atol=0.01)
    np.testing.assert_allclose(result.total_effect[:, 1], [0.2, 0.8, 0], atol=0.01)
    assert np.isnan(result.first_order[:, 2]).all()
    assert np.isnan(result.total_effect[:, 2]).all()

    with np.load(tmp_path / "sobol_stats.npz") as saved:
        np.testing.assert_equal(saved["parameter_samples"], result.parameter_samples)
        np.testing.assert_equal(saved["first_order"], result.first_order)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("nonblocking", [False, True], ids=["blocking", "nonblocking"])
def test_sobol_dry_run_and_restart(tmp_path, nonblocking, deferred_model):
    model = SensitivityModel()
    if nonblocking:
        model = deferred_model(model)
    space = GaussianParameterSpace(["x", "y", "z"], [0] * 3, [1] * 3, MonteCarloSampler)
    prepared = run_sobol_indices(model, space, tmp_path, 8, dry_run=True)
    assert len(prepared.run_directories) == 40
    assert not (tmp_path / "sobol_stats.npz").exists()

    result = run_sobol_indices(model, space, tmp_path, 8, evaluation_concurrency=2)
    np.testing.assert_equal(result.parameter_samples, prepared.parameter_samples)
    a, b, *hybrids = result.parameter_samples.reshape(5, 8, 3)
    for i, hybrid in enumerate(hybrids):
        expected = a.copy()
        expected[:, i] = b[:, i]
        np.testing.assert_equal(hybrid, expected)
    assert np.isfinite(result.run_times).all()

    restarted = run_sobol_indices(model, space, tmp_path, 8)
    assert np.isnan(restarted.run_times).all()
    np.testing.assert_equal(result.qoi_values, restarted.qoi_values)

    changed = run_sobol_indices(model, space, tmp_path, 8, random_seed=2)
    assert np.isfinite(changed.run_times).all()
    assert not np.array_equal(changed.parameter_samples, result.parameter_samples)


def test_sobol_rejects_correlated_inputs(tmp_path):
    space = MultivariateGaussianParameterSpace(["x", "y"], [0, 0], [[1, 0.5], [0.5, 1]])
    with pytest.raises(ValueError, match="independent inputs"):
        run_sobol_indices(SensitivityModel(), space, tmp_path, 8)


def test_sobol_requires_power_of_two(tmp_path):
    with pytest.raises(ValueError, match="power of two"):
        run_sobol_indices(SensitivityModel(), parameter_space(), tmp_path, 6)


class InteractionBlindModel(SensitivityModel):
    def compute_qoi(self, run_directory, parameter_sample):
        x, y = (parameter_sample[name] for name in ("x", "y"))
        return np.array([np.sin(x) + 7 * np.sin(y) ** 2, 1.0, 0.1])


@pytest.mark.mpi_skip
@pytest.mark.parametrize("nonblocking", [False, True], ids=["blocking", "nonblocking"])
def test_multifidelity_sobol_workflow(tmp_path, nonblocking, deferred_model):
    high, low = SensitivityModel(), InteractionBlindModel()
    if nonblocking:
        high, low = deferred_model(high), deferred_model(low)
    space = parameter_space()
    options = dict(
        number_of_high_fidelity_base_samples=8,
        number_of_low_fidelity_base_samples=16,
        pilot_sample_count=8,
        high_fidelity_evaluation_concurrency=2,
        low_fidelity_evaluation_concurrency=2,
    )
    prepared = run_multifidelity_sobol_indices(high, low, space, tmp_path, dry_run=True, **options)
    assert len(prepared.high_fidelity_pilot.run_directories) == 40
    assert len(prepared.low_fidelity_pilot.run_directories) == 40
    assert len(prepared.high_fidelity.run_directories) == 40
    assert len(prepared.low_fidelity.run_directories) == 80
    assert not (tmp_path / "multifidelity_sobol_stats.npz").exists()

    result = run_multifidelity_sobol_indices(high, low, space, tmp_path, **options)
    np.testing.assert_equal(
        result.high_fidelity_parameter_samples,
        result.low_fidelity_parameter_samples.reshape(5, 16, 3)[:, :8].reshape(-1, 3),
    )
    np.testing.assert_equal(prepared.high_fidelity_pilot.parameter_samples, result.pilot_parameter_samples)
    np.testing.assert_equal(prepared.low_fidelity_pilot.parameter_samples, result.pilot_parameter_samples)
    assert not np.array_equal(result.pilot_parameter_samples, result.high_fidelity_parameter_samples)
    assert result.control_variate_coefficients.shape == (7, 3)
    np.testing.assert_equal(result.control_variate_coefficients[[3, 6], 0], [0.0, 0.0])
    assert np.any(result.control_variate_coefficients[:, 0] != 0)
    np.testing.assert_equal(result.used_high_fidelity_only[1:], [True, True])
    expected_first, expected_total = _estimate_indices(result.high_fidelity_qoi_values, 8)
    np.testing.assert_allclose(result.first_order[:, 1:], expected_first[:, 1:])
    np.testing.assert_allclose(result.total_effect[:, 1:], expected_total[:, 1:])
    assert np.isfinite(result.high_fidelity_run_times).all()
    assert np.isfinite(result.low_fidelity_run_times).all()
    assert np.isfinite(result.high_fidelity_pilot_run_times).all()
    assert np.isfinite(result.low_fidelity_pilot_run_times).all()

    restarted = run_multifidelity_sobol_indices(high, low, space, tmp_path, **options)
    assert np.isnan(restarted.high_fidelity_run_times).all()
    assert np.isnan(restarted.low_fidelity_run_times).all()
    assert np.isnan(restarted.high_fidelity_pilot_run_times).all()
    assert np.isnan(restarted.low_fidelity_pilot_run_times).all()
    np.testing.assert_equal(result.first_order, restarted.first_order)
    np.testing.assert_equal(result.total_effect, restarted.total_effect)
    with np.load(tmp_path / "multifidelity_sobol_stats.npz") as saved:
        np.testing.assert_equal(saved["first_order"], result.first_order)
        np.testing.assert_equal(saved["pilot_correlations"], result.pilot_correlations)


@pytest.mark.mpi_skip
def test_matching_fidelity_recovers_large_design_estimate(tmp_path):
    model = SensitivityModel()
    result = run_multifidelity_sobol_indices(
        model, model, parameter_space(), tmp_path,
        number_of_high_fidelity_base_samples=8,
        number_of_low_fidelity_base_samples=256,
        pilot_sample_count=16,
    )
    a, b, *hybrids = result.low_fidelity_qoi_values.reshape(5, 256, 3)
    pilot_center = result.high_fidelity_pilot_qoi_values[:32].mean(axis=0)
    variance = 0.5 * np.mean((a - b) ** 2, axis=0)
    differences = np.asarray(hybrids) - a
    np.testing.assert_allclose(
        result.first_order[:, :2],
        np.mean((b - pilot_center) * differences, axis=1)[:, :2] / variance[:2],
    )
    np.testing.assert_allclose(
        result.total_effect[:, :2],
        0.5 * np.mean(differences ** 2, axis=1)[:, :2] / variance[:2],
    )


def test_multifidelity_sobol_rejects_smaller_low_fidelity_design(tmp_path):
    model = SensitivityModel()
    with pytest.raises(ValueError, match="low-fidelity base sample count"):
        run_multifidelity_sobol_indices(model, model, parameter_space(), tmp_path, 8, 4)


def test_multifidelity_sobol_falls_back_for_negative_corrected_variance():
    # One input implies AB_0 == B. The matching pilot yields a variance
    # coefficient of one, but the low-fidelity variance correction here is
    # negative and larger in magnitude than the high-fidelity variance.
    pilot = np.array([0, 0, 1, 2, 1, 2], dtype=float)[:, None]
    high = np.array([0, 0, 0.1, 0.2, 0.1, 0.2])[:, None]
    low = np.array([0, 0, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0], dtype=float)[:, None]

    first, total, coefficients, _, fallback, _, _ = _estimate_multifidelity_indices(
        high, low, pilot, pilot, 2, 4, 2
    )
    np.testing.assert_allclose(coefficients[0], [1.0])
    np.testing.assert_equal(fallback, [True])
    expected_first, expected_total = _estimate_indices(high, 2)
    np.testing.assert_allclose(first, expected_first)
    np.testing.assert_allclose(total, expected_total)
