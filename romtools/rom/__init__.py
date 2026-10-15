"""Reduced-order model projection and QoI surrogate utilities.

The names in ``__all__`` define the supported public API for this package.
Implementation helpers remain accessible from their defining modules but are
not part of the compatibility guarantee.
"""

from . import neural_network_surrogate, projections, qoi_surrogates
from .neural_network_surrogate import (
    LipschitzConfig,
    LipschitzNeuralNetworkQoiModel,
    NeuralNetworkQoiModelBuilderWithTrainingData,
    estimate_pairwise_lipschitz_constant,
)
from .projections import optimal_l2_projection
from .qoi_surrogates import (
    GaussianProcessKernel,
    GaussianProcessQoiModel,
    GaussianProcessRegressorLite,
    NeuralNetworkConfig,
    NeuralNetworkQoiModel,
)

__all__ = [
    "projections",
    "qoi_surrogates",
    "neural_network_surrogate",
    "optimal_l2_projection",
    "GaussianProcessKernel",
    "GaussianProcessRegressorLite",
    "GaussianProcessQoiModel",
    "NeuralNetworkConfig",
    "NeuralNetworkQoiModel",
    "LipschitzConfig",
    "estimate_pairwise_lipschitz_constant",
    "LipschitzNeuralNetworkQoiModel",
    "NeuralNetworkQoiModelBuilderWithTrainingData",
]
