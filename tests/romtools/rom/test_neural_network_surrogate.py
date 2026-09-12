import numpy as np
import pytest

from romtools.rom.neural_network_surrogate import (
    NeuralNetworkQoiModelBuilderWithTrainingData,
)
from romtools.rom.qoi_surrogates import NeuralNetworkConfig, NeuralNetworkQoiModel


torch = pytest.importorskip("torch")
pytestmark = pytest.mark.mpi_skip


def test_neural_network_config_baseline_defaults():
    config = NeuralNetworkConfig()

    assert config.num_hidden_layers == 2
    assert config.hidden_layer_width(4) == 12
    assert config.optimizer == "adam"
    assert config.training_iterations == 5000


def test_neural_network_scalar_qoi_uses_configured_architecture():
    parameters = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
        ]
    )
    qois = parameters[:, 0] + 2.0 * parameters[:, 1]
    config = NeuralNetworkConfig(
        num_hidden_layers=2,
        hidden_neurons_per_layer=8,
        training_iterations=1000,
        learning_rate=1.0e-2,
        random_seed=3,
    )

    model = NeuralNetworkQoiModel(
        parameters=parameters,
        qois=qois,
        parameter_names=["x", "y"],
        network_config=config,
    )

    linear_layers = [
        layer for layer in model._network if isinstance(layer, torch.nn.Linear)
    ]
    assert len(linear_layers) == 3
    assert linear_layers[0].in_features == 2
    assert linear_layers[0].out_features == 8
    assert linear_layers[1].in_features == 8
    assert linear_layers[1].out_features == 8
    assert linear_layers[2].out_features == 1

    prediction = model.compute_qoi("", {"x": 0.25, "y": 0.75})
    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])
    assert model.final_training_loss < 1.0e-3


def test_neural_network_vector_qoi_uses_pod_coefficients():
    parameters = np.linspace(0.0, 1.0, 9)[:, None]
    coefficient = 2.0 * parameters[:, 0] - 0.5
    qois = np.column_stack(
        [
            1.0 + coefficient,
            -2.0 + 2.0 * coefficient,
            0.5 - coefficient,
        ]
    )
    config = NeuralNetworkConfig(
        training_iterations=750,
        learning_rate=1.0e-2,
        random_seed=5,
    )

    model = NeuralNetworkQoiModel(
        parameters=parameters,
        qois=qois,
        parameter_names=["mu"],
        pod_energy_fraction=0.999999,
        max_pod_modes=1,
        network_config=config,
    )

    assert model._pod_modes.shape == (3, 1)
    assert model._network[-1].out_features == 1
    prediction = model.compute_qoi("", {"mu": 0.35})
    assert prediction.shape == (3,)
    assert np.all(np.isfinite(prediction))


def test_neural_network_builder_forwards_pod_and_training_configuration():
    config = NeuralNetworkConfig(
        hidden_neurons_per_layer=11,
        num_hidden_layers=3,
        training_iterations=123,
    )
    builder = NeuralNetworkQoiModelBuilderWithTrainingData(
        parameter_names=["a", "b"],
        pod_energy_fraction=0.95,
        max_pod_modes=4,
        network_config=config,
        normalize_parameters=False,
        normalize_targets=False,
    )

    assert builder.parameter_names == ["a", "b"]
    assert builder.pod_energy_fraction == 0.95
    assert builder.max_pod_modes == 4
    assert builder.network_config is config
    assert not builder.normalize_parameters
    assert not builder.normalize_targets
