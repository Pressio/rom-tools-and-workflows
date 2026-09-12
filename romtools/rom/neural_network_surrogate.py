from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from romtools.rom.qoi_surrogates import NeuralNetworkConfig, NeuralNetworkQoiModel


class NeuralNetworkQoiModelBuilderWithTrainingData:
    """Build a torch neural-network QoI surrogate from workflow training data.

    The builder mirrors the POD controls used by the Gaussian-process
    surrogate while keeping neural-network architecture and optimization
    settings in :class:`NeuralNetworkConfig`.
    """

    def __init__(self,
                 parameter_names: Optional[Sequence[str]],
                 pod_energy_fraction: float = 0.999999,
                 max_pod_modes: Optional[int] = None,
                 network_config: Optional[NeuralNetworkConfig] = None,
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
        self.normalize_parameters = normalize_parameters
        self.normalize_targets = normalize_targets

    def build_from_training_dirs(self,
                                 offline_data_dir: str,
                                 training_data_dirs,
                                 training_parameters: np.ndarray,
                                 training_qois: np.ndarray):
        return NeuralNetworkQoiModel(
            parameters=training_parameters,
            qois=training_qois,
            parameter_names=self.parameter_names,
            pod_energy_fraction=self.pod_energy_fraction,
            max_pod_modes=self.max_pod_modes,
            network_config=self.network_config,
            normalize_parameters=self.normalize_parameters,
            normalize_targets=self.normalize_targets,
        )
