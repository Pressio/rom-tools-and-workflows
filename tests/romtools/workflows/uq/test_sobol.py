import numpy as np
import pytest

from romtools.workflows.parameter_spaces import (
    GaussianParameterSpace,
    MultivariateGaussianParameterSpace,
    UniformParameterSpace,
)
from romtools.workflows.sampling_methods import MonteCarloSampler
from romtools.workflows.uq import run_sobol_indices


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
