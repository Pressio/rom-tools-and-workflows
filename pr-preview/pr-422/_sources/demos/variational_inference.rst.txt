Variational inference
=====================

Variational inference (VI) approximates a Bayesian posterior by optimizing a
parameterized probability distribution. ROM Tools supports mean-field and
full-covariance Gaussian VI. Multifidelity VI (MF-VI) combines full-order and
surrogate-model evaluations to reduce the cost of estimating the optimization
objective and its derivatives.

Examples
--------

- :doc:`Analytic sine VI and MF-VI <vi_mf_vi_demo>`: compare diagonal and
  full-covariance approximations against a known Gaussian posterior.
- :doc:`H2-air flame VI and MF-VI <h2_air_flame_vi_mf_vi_demo>`: compare
  full-order inference with Gaussian-process surrogate-assisted inference.

For signatures and configuration options, see :doc:`../api_inverse`.
Detailed algorithm and restart controls are documented in
:doc:`../vi_implementation_notes`.

.. toctree::
   :hidden:
   :maxdepth: 1

   vi_mf_vi_demo
   h2_air_flame_vi_mf_vi_demo
