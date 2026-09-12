from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from romtools.rom.qoi_surrogates import (
    NeuralNetworkConfig,
    NeuralNetworkQoiModel,
    _import_torch,
)


@dataclass
class LipschitzConfig:
    """Configuration for a hard Lipschitz constraint on the neural surrogate.

    When enabled, spectral normalization constrains every linear operator to
    unit spectral norm and a scalar multiplier is applied to each layer. If
    ``lipschitz_constant`` is not supplied, the global constant is estimated
    from pairwise slopes in the transformed training coordinates and multiplied
    by ``safety_factor``.
    """

    enabled: bool = False
    lipschitz_constant: Optional[float] = None
    safety_factor: float = 1.1
    spectral_norm_power_iterations: int = 5
    pairwise_distance_tolerance: float = 1.0e-12
    layer_lipschitz_constants: Optional[Sequence[float]] = None

    def validate(self) -> None:
        if self.lipschitz_constant is not None and self.lipschitz_constant < 0.0:
            raise ValueError("lipschitz_constant must be nonnegative")
        if self.safety_factor <= 0.0:
            raise ValueError("safety_factor must be positive")
        if self.spectral_norm_power_iterations < 1:
            raise ValueError("spectral_norm_power_iterations must be positive")
        if self.pairwise_distance_tolerance < 0.0:
            raise ValueError("pairwise_distance_tolerance must be nonnegative")
        if self.layer_lipschitz_constants is not None:
            if len(self.layer_lipschitz_constants) < 1:
                raise ValueError("layer_lipschitz_constants must not be empty")
            if any(value < 0.0 for value in self.layer_lipschitz_constants):
                raise ValueError("layer_lipschitz_constants must be nonnegative")


def estimate_pairwise_lipschitz_constant(
    parameters: np.ndarray,
    targets: np.ndarray,
    distance_tolerance: float = 1.0e-12,
) -> float:
    """Estimate the maximum pairwise Lipschitz slope from training data."""
    parameters = np.asarray(parameters, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if parameters.ndim != 2 or targets.ndim != 2:
        raise ValueError("parameters and targets must both be two-dimensional")
    if parameters.shape[0] != targets.shape[0]:
        raise ValueError("parameters and targets must contain the same samples")

    maximum_slope = 0.0
    for first in range(parameters.shape[0]):
        for second in range(first + 1, parameters.shape[0]):
            parameter_distance = np.linalg.norm(
                parameters[first] - parameters[second]
            )
            target_distance = np.linalg.norm(targets[first] - targets[second])
            if parameter_distance <= distance_tolerance:
                if target_distance > distance_tolerance:
                    raise ValueError(
                        "Training data contain coincident parameter samples with "
                        "different targets, so a finite Lipschitz constant cannot "
                        "be estimated."
                    )
                continue
            maximum_slope = max(
                maximum_slope,
                float(target_distance / parameter_distance),
            )
    return maximum_slope


class LipschitzNeuralNetworkQoiModel(NeuralNetworkQoiModel):
    """Neural-network QoI surrogate with a spectral Lipschitz constraint."""

    def __init__(self,
                 parameters: np.ndarray,
                 qois: np.ndarray,
                 parameter_names: Optional[Sequence[str]] = None,
                 pod_energy_fraction: float = 0.999999,
                 max_pod_modes: Optional[int] = None,
                 network_config: Optional[NeuralNetworkConfig] = None,
                 lipschitz_config: Optional[LipschitzConfig] = None,
                 normalize_parameters: bool = True,
                 normalize_targets: bool = True) -> None:
        self.lipschitz_config = (
            lipschitz_config if lipschitz_config is not None else LipschitzConfig(enabled=True)
        )
        self.lipschitz_config.validate()
        if not self.lipschitz_config.enabled:
            raise ValueError(
                "LipschitzNeuralNetworkQoiModel requires lipschitz_config.enabled=True"
            )
        self.estimated_lipschitz_constant: Optional[float] = None
        self.lipschitz_constant: Optional[float] = None
        self.layer_lipschitz_constants = None
        self._lipschitz_linear_layers = []
        super().__init__(
            parameters=parameters,
            qois=qois,
            parameter_names=parameter_names,
            pod_energy_fraction=pod_energy_fraction,
            max_pod_modes=max_pod_modes,
            network_config=network_config,
            normalize_parameters=normalize_parameters,
            normalize_targets=normalize_targets,
        )

    def _fit(self) -> None:
        torch = _import_torch()
        self._torch = torch
        try:
            self._torch_dtype = getattr(torch, self.network_config.dtype)
        except AttributeError as exc:
            raise ValueError(
                f"Unsupported torch dtype '{self.network_config.dtype}'."
            ) from exc
        device = torch.device(self.network_config.device)

        parameters = self._initialize_parameter_scaling()
        targets = self._build_reduced_targets()
        targets = self._initialize_target_scaling(targets)

        self._resolve_lipschitz_constraint(parameters, targets)

        torch.manual_seed(self.network_config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.network_config.random_seed)

        self._network = self._build_lipschitz_network(
            parameters.shape[1], targets.shape[1], torch
        ).to(device=device, dtype=self._torch_dtype)

        x_train = torch.as_tensor(parameters, dtype=self._torch_dtype, device=device)
        y_train = torch.as_tensor(targets, dtype=self._torch_dtype, device=device)
        optimizer = self._build_optimizer(torch)
        loss_function = torch.nn.MSELoss()

        self._network.train()
        for _ in range(self.network_config.training_iterations):
            optimizer.zero_grad()
            prediction = self._network(x_train)
            loss = loss_function(prediction, y_train)
            loss.backward()
            optimizer.step()
        self.final_training_loss = float(loss.detach().cpu().item())
        self._network.eval()

    def _resolve_lipschitz_constraint(
        self,
        parameters: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        activation = self.network_config.activation.strip().lower()
        if activation not in ("tanh", "relu"):
            raise ValueError(
                "A hard Lipschitz constraint currently requires a 1-Lipschitz "
                "activation ('tanh' or 'relu')."
            )

        self.estimated_lipschitz_constant = estimate_pairwise_lipschitz_constant(
            parameters,
            targets,
            self.lipschitz_config.pairwise_distance_tolerance,
        )
        if self.lipschitz_config.lipschitz_constant is None:
            self.lipschitz_constant = (
                self.lipschitz_config.safety_factor
                * self.estimated_lipschitz_constant
            )
        else:
            self.lipschitz_constant = float(
                self.lipschitz_config.lipschitz_constant
            )

        num_linear_layers = self.network_config.num_hidden_layers + 1
        configured_layers = self.lipschitz_config.layer_lipschitz_constants
        if configured_layers is None:
            if self.lipschitz_constant == 0.0:
                layer_constant = 0.0
            else:
                layer_constant = self.lipschitz_constant ** (1.0 / num_linear_layers)
            self.layer_lipschitz_constants = [
                layer_constant for _ in range(num_linear_layers)
            ]
        else:
            if len(configured_layers) != num_linear_layers:
                raise ValueError(
                    "layer_lipschitz_constants must contain one value per linear layer"
                )
            self.layer_lipschitz_constants = [
                float(value) for value in configured_layers
            ]
            layer_product = float(np.prod(self.layer_lipschitz_constants))
            tolerance = 1.0e-12 * max(1.0, self.lipschitz_constant)
            if layer_product > self.lipschitz_constant + tolerance:
                raise ValueError(
                    "The product of layer_lipschitz_constants cannot exceed the "
                    "global lipschitz_constant."
                )

    def _build_lipschitz_network(
        self,
        input_dimension: int,
        output_dimension: int,
        torch,
    ):
        activation_type = self._activation_type(torch)
        power_iterations = self.lipschitz_config.spectral_norm_power_iterations
        self._lipschitz_linear_layers = []

        class ScaledSpectralLinear(torch.nn.Module):
            def __init__(self, in_features, out_features, scale):
                super().__init__()
                linear = torch.nn.Linear(in_features, out_features)
                self.linear = torch.nn.utils.parametrizations.spectral_norm(
                    linear,
                    n_power_iterations=power_iterations,
                )
                self.lipschitz_scale = float(scale)

            def forward(self, inputs):
                return torch.nn.functional.linear(
                    inputs,
                    self.lipschitz_scale * self.linear.weight,
                    self.linear.bias,
                )

        layers = []
        previous_width = input_dimension
        for hidden_layer in range(self.network_config.num_hidden_layers):
            linear = ScaledSpectralLinear(
                previous_width,
                self.hidden_neurons_per_layer,
                self.layer_lipschitz_constants[hidden_layer],
            )
            layers.append(linear)
            self._lipschitz_linear_layers.append(linear)
            layers.append(activation_type())
            previous_width = self.hidden_neurons_per_layer

        output_layer = ScaledSpectralLinear(
            previous_width,
            output_dimension,
            self.layer_lipschitz_constants[-1],
        )
        layers.append(output_layer)
        self._lipschitz_linear_layers.append(output_layer)
        return torch.nn.Sequential(*layers)


class NeuralNetworkQoiModelBuilderWithTrainingData:
    """Build a torch neural-network QoI surrogate from workflow training data.

    The builder mirrors the POD controls used by the Gaussian-process
    surrogate while keeping neural-network architecture and optimization
    settings in :class:`NeuralNetworkConfig`. A hard Lipschitz constraint can
    be enabled independently with :class:`LipschitzConfig`.
    """

    def __init__(self,
                 parameter_names: Optional[Sequence[str]],
                 pod_energy_fraction: float = 0.999999,
                 max_pod_modes: Optional[int] = None,
                 network_config: Optional[NeuralNetworkConfig] = None,
                 lipschitz_config: Optional[LipschitzConfig] = None,
                 normalize_parameters: bool = True,
                 normalize_targets: bool = True) -> None:
        self.parameter_names = (
            list(parameter_names) if parameter_names is not None else None
        )
        self.pod_energy_fraction = pod_energy_fraction
        self.max_pod_modes = max_pod_modes
        self.network_config = (
            network_config if network_config is not None else NeuralNetworkConfig()
        )
        self.lipschitz_config = (
            lipschitz_config if lipschitz_config is not None else LipschitzConfig()
        )
        self.lipschitz_config.validate()
        self.normalize_parameters = normalize_parameters
        self.normalize_targets = normalize_targets

    def build_from_training_dirs(self,
                                 offline_data_dir: str,
                                 training_data_dirs,
                                 training_parameters: np.ndarray,
                                 training_qois: np.ndarray):
        model_type = (
            LipschitzNeuralNetworkQoiModel
            if self.lipschitz_config.enabled
            else NeuralNetworkQoiModel
        )
        model_kwargs = dict(
            parameters=training_parameters,
            qois=training_qois,
            parameter_names=self.parameter_names,
            pod_energy_fraction=self.pod_energy_fraction,
            max_pod_modes=self.max_pod_modes,
            network_config=self.network_config,
            normalize_parameters=self.normalize_parameters,
            normalize_targets=self.normalize_targets,
        )
        if self.lipschitz_config.enabled:
            model_kwargs["lipschitz_config"] = self.lipschitz_config
        return model_type(**model_kwargs)
