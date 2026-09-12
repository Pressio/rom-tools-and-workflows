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


def _parameter_space():
    return UniformParameterSpace(
        ["u", "v"],
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        sampler=MonteCarloSampler,
    )


@pytest.mark.mpi_skip
def test_mf_eki_with_auto_rom_gp_builder(monkeypatch):
    captured = {}

    def fake_run_mf_eki(**kwargs):
        captured["rom_model_builder"] = kwargs["rom_model_builder"]
        return "ok", None

    monkeypatch.setattr(mf_module, "run_mf_eki", fake_run_mf_eki)

    parameter_space = _parameter_space()
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
def test_mf_eki_with_auto_rom_nn_builder(monkeypatch):
    captured = {}

    def fake_run_mf_eki(**kwargs):
        captured["rom_model_builder"] = kwargs["rom_model_builder"]
        return "ok", None

    monkeypatch.setattr(mf_module, "run_mf_eki", fake_run_mf_eki)

    network_config = mf_module.NeuralNetworkConfig(training_iterations=17)
    lipschitz_config = mf_module.LipschitzConfig(enabled=True, safety_factor=1.2)
    mf_module.mf_eki_with_auto_rom(
        model=DummyQoiModel(),
        parameter_space=_parameter_space(),
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
        rom_type="nn",
        rom_args={
            "pod_energy_fraction": 0.95,
            "max_pod_modes": 4,
            "network_config": network_config,
            "lipschitz_config": lipschitz_config,
            "normalize_parameters": False,
            "normalize_targets": False,
        },
    )

    builder = captured["rom_model_builder"]
    assert isinstance(builder, mf_module.NeuralNetworkQoiModelBuilderWithTrainingData)
    assert builder.parameter_names == ["u", "v"]
    assert builder.pod_energy_fraction == 0.95
    assert builder.max_pod_modes == 4
    assert builder.network_config is network_config
    assert builder.lipschitz_config is lipschitz_config
    assert builder.normalize_parameters is False
    assert builder.normalize_targets is False


@pytest.mark.mpi_skip
@pytest.mark.parametrize("rom_type", ["neural_network", "neural-network"])
def test_mf_eki_with_auto_rom_nn_aliases(monkeypatch, rom_type):
    captured = {}

    def fake_run_mf_eki(**kwargs):
        captured["rom_model_builder"] = kwargs["rom_model_builder"]
        return "ok", None

    monkeypatch.setattr(mf_module, "run_mf_eki", fake_run_mf_eki)
    mf_module.mf_eki_with_auto_rom(
        model=DummyQoiModel(),
        parameter_space=_parameter_space(),
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
        rom_type=rom_type,
    )

    assert isinstance(
        captured["rom_model_builder"],
        mf_module.NeuralNetworkQoiModelBuilderWithTrainingData,
    )


@pytest.mark.mpi_skip
def test_mf_eki_with_auto_rom_invalid_type():
    with pytest.raises(ValueError, match="Unsupported rom_type"):
        mf_module.mf_eki_with_auto_rom(
            model=DummyQoiModel(),
            parameter_space=_parameter_space(),
            observations=np.array([0.0]),
            observations_covariance=np.eye(1),
            rom_type="not-a-rom",
        )
