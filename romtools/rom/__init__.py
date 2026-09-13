"""Reduced-order model projection and QoI surrogate utilities.

The names in ``__all__`` define the supported public API for this package.
Implementation helpers remain accessible from their defining modules but are
not part of the compatibility guarantee.
"""

from romtools.rom import neural_network_surrogate, projections, qoi_surrogates
from romtools.rom.projections import *
from romtools.rom.qoi_surrogates import *
from romtools.rom.neural_network_surrogate import *

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
