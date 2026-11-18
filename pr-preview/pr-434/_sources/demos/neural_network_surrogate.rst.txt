Neural-network QoI surrogate
============================

**Method.** A neural-network surrogate approximates quantities of interest as functions of model parameters. The implementation can constrain the model's Lipschitz constant.

**Example.** The example constructs a PyTorch QoI surrogate, including a configuration with explicit Lipschitz constraints.

Install optional support with ``pip install 'romtools[WithTorch]'``.

For algorithm details, settings, and full source listings, see
:doc:`neural_network_surrogate_details`.

.. toctree::
   :hidden:

   neural_network_surrogate_details
