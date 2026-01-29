import numpy as np
import pytest

from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.parameter_spaces import MonteCarloSampler
import romtools.workflows.inverse.mf_eki_drivers as mf_module


class DummyQoiModel:
    def populate_run_directory(self, run_directory, parameter_sample):
        return None

    def run_model(self, run_directory, parameter_sample):
        return 0

    def compute_qoi(self, run_directory, parameter_sample):
        return np.array([0.0])


@pytest.mark.mpi_skip
def test_mf_eki_with_auto_rom_gp_builder(monkeypatch):
    captured = {}

    def fake_run_mf_eki(**kwargs):
        captured["rom_model_builder"] = kwargs["rom_model_builder"]
        return "ok", None

    monkeypatch.setattr(mf_module, "run_mf_eki", fake_run_mf_eki)

    parameter_space = UniformParameterSpace(
        ["u", "v"],
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        sampler=MonteCarloSampler,
    )

    kernel = mf_module.GaussianProcessKernel(length_scale=2.5, signal_variance=0.3)
    mf_module.mf_eki_with_auto_rom(
        model=DummyQoiModel(),
        parameter_space=parameter_space,
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
        rom_args={
            "pod_energy_fraction": 0.9,
            "max_pod_modes": 3,
            "kernel": kernel,
            "noise_variance": 1e-6,
        },
    )

    builder = captured["rom_model_builder"]
    assert isinstance(builder, mf_module.GaussianProcessQoiModelBuilderWithTrainingData)
    assert builder.parameter_names == ["u", "v"]
    assert builder.pod_energy_fraction == 0.9
    assert builder.max_pod_modes == 3
    assert builder.kernel is kernel
    assert builder.noise_variance == 1e-6


@pytest.mark.mpi_skip
def test_mf_eki_with_auto_rom_invalid_type():
    parameter_space = UniformParameterSpace(
        ["u", "v"],
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        sampler=MonteCarloSampler,
    )

    with pytest.raises(ValueError, match="Unsupported rom_type"):
        mf_module.mf_eki_with_auto_rom(
            model=DummyQoiModel(),
            parameter_space=parameter_space,
            observations=np.array([0.0]),
            observations_covariance=np.eye(1),
            rom_type="not-a-rom",
        )
