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
