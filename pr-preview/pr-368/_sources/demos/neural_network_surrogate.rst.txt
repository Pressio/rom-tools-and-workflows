Neural-network QoI surrogate
============================

romtools provides a PyTorch-based neural-network surrogate that implements the
standard ``QoiModel`` interface. The surrogate maps model parameters to scalar
or vector quantities of interest and can be used anywhere a data-driven
``QoiModelBuilderWithTrainingData`` is accepted.

Installation
------------

PyTorch is an optional dependency. Install the neural-network support with

.. code-block:: bash

   pip install "romtools[WithTorch]"

Baseline configuration
----------------------

The baseline network is a fully connected feed-forward network with

* two hidden layers,
* three times the parameter dimension neurons in each hidden layer,
* ``tanh`` activations,
* Adam optimization, and
* 5,000 full-batch optimization iterations.

These choices are collected in ``NeuralNetworkConfig`` and can be changed when
the surrogate builder is constructed.

.. code-block:: python

   from romtools.rom import (
       NeuralNetworkConfig,
       NeuralNetworkQoiModelBuilderWithTrainingData,
   )

   network_config = NeuralNetworkConfig(
       num_hidden_layers=2,
       hidden_neurons_per_layer=None,  # defaults to 3 * parameter dimension
       optimizer="adam",
       learning_rate=1.0e-3,
       training_iterations=5000,
       random_seed=1,
   )

   builder = NeuralNetworkQoiModelBuilderWithTrainingData(
       parameter_names=["mu_1", "mu_2"],
       network_config=network_config,
       normalize_parameters=True,
       normalize_targets=True,
   )

Setting ``hidden_neurons_per_layer`` to an integer overrides the default
``3 * parameter_dimension`` width. The number of hidden layers, activation,
optimizer, learning rate, number of training iterations, weight decay, random
seed, floating-point dtype, and torch device are also configurable through
``NeuralNetworkConfig``.

MF-EKI auto-ROM interface
-------------------------

The MF-EKI convenience driver can construct the neural surrogate directly with
``rom_type="nn"``. Neural-network and Lipschitz configuration objects are
passed through ``rom_args`` together with the POD and normalization controls.
This is the same public auto-ROM interface used by the Gaussian-process
surrogate.

.. code-block:: python

   from romtools.rom import LipschitzConfig, NeuralNetworkConfig
   from romtools.workflows.inverse.mf_eki_drivers import mf_eki_with_auto_rom

   mf_eki_with_auto_rom(
       model=fom_model,
       parameter_space=parameter_space,
       observations=observations,
       observations_covariance=observations_covariance,
       rom_type="nn",
       rom_args={
           "network_config": NeuralNetworkConfig(
               training_iterations=5000,
           ),
           "lipschitz_config": LipschitzConfig(
               enabled=True,
               safety_factor=1.1,
           ),
           "normalize_parameters": True,
           "normalize_targets": True,
       },
   )

``"neural_network"`` and ``"neural-network"`` are accepted aliases, while
``"nn"`` is the canonical short form used in the examples.

POD for vector QoIs
-------------------

Vector-valued QoIs use the same POD reduction strategy as the Gaussian-process
surrogate. The training QoIs are mean-centered, an SVD is computed, and the
basis is truncated using ``pod_energy_fraction`` and, optionally,
``max_pod_modes``. The neural network predicts the retained POD coefficients
and the full QoI is reconstructed from those coefficients.

Unlike the Gaussian-process implementation, which trains one scalar GP for
each retained coefficient, the neural-network surrogate uses one multi-output
network for all retained POD coefficients.

.. code-block:: python

   builder = NeuralNetworkQoiModelBuilderWithTrainingData(
       parameter_names=["mu_1", "mu_2", "mu_3"],
       pod_energy_fraction=0.999999,
       max_pod_modes=20,
       network_config=NeuralNetworkConfig(),
   )

For a scalar QoI, no POD reduction is performed and the network predicts the
QoI directly.

Lipschitz-constrained networks
------------------------------

A hard Lipschitz constraint can optionally be applied using spectral
normalization. For a network with ``L`` linear layers, each linear operator is
spectrally normalized and then multiplied by a layer constant ``K_l``. With a
1-Lipschitz activation, the resulting network satisfies the approximate bound

.. math::

   \operatorname{Lip}(f) \lesssim \prod_{l=1}^{L} K_l.

The approximation reflects the finite power iteration used by PyTorch spectral
normalization. By default, the global constant ``K`` is distributed uniformly
across the linear layers,

.. math::

   K_l = K^{1/L}.

If ``lipschitz_constant`` is not supplied, romtools estimates a data-based
constant from the maximum pairwise slope

.. math::

   K_{\mathrm{data}}
   = \max_{i<j}
   \frac{\|z_i-z_j\|_2}{\|x_i-x_j\|_2},

and uses ``K = safety_factor * K_data``. The estimate is computed in the exact
coordinates learned by the network: after parameter normalization and, for
vector QoIs, after POD projection and optional target normalization.

.. code-block:: python

   from romtools.rom import LipschitzConfig

   lipschitz_config = LipschitzConfig(
       enabled=True,
       lipschitz_constant=None,       # estimate from the training data
       safety_factor=1.1,
       spectral_norm_power_iterations=5,
   )

   builder = NeuralNetworkQoiModelBuilderWithTrainingData(
       parameter_names=["mu_1", "mu_2"],
       network_config=NeuralNetworkConfig(activation="tanh"),
       lipschitz_config=lipschitz_config,
       normalize_parameters=True,
       normalize_targets=True,
   )

An explicit global constant can be supplied with ``lipschitz_constant``.
Advanced users may also provide ``layer_lipschitz_constants`` directly; there
must be one value for each linear layer and their product cannot exceed the
global constant.

The hard-constraint mode currently accepts ``tanh`` and ReLU activations,
which are 1-Lipschitz. GELU and SiLU are intentionally rejected because the
linear-layer spectral constraints alone would not imply the requested global
bound. Coincident training parameter samples with different transformed
targets are also rejected because they imply no finite data-based Lipschitz
constant.
