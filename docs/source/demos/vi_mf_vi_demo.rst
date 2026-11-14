Analytic sine VI and MF-VI examples
===================================

Variational inference (VI) approximates a posterior distribution by optimizing
a Gaussian approximation. Multifidelity VI (MF-VI) uses additional inexpensive
surrogate evaluations to reduce estimator cost.

These examples infer seven coefficients in a one-dimensional sine-series
problem with a known Gaussian posterior, allowing the numerical results to be
compared with the exact solution.

Diagonal Gaussian posterior
---------------------------

Equispaced observations produce a nearly diagonal posterior. This example
compares mean-field VI and MF-VI with the exact posterior.

Run ``python examples/vi_mf_vi_demo/diagonal_example.py``.

.. figure:: notebooks/analytic_sine_diagonal_convergence.png
   :alt: KL convergence for diagonal VI and MF-VI
   :width: 78%

   Convergence toward the exact posterior.

.. figure:: notebooks/analytic_sine_diagonal_posterior.png
   :alt: Diagonal VI and MF-VI posterior marginals
   :width: 96%

   Exact and approximate posterior marginals.

Correlated Gaussian posterior
-----------------------------

Non-equispaced observations induce posterior correlations. This example
compares full-covariance and mean-field VI/MF-VI, highlighting the approximation
error caused by a diagonal covariance restriction.

Run ``python examples/vi_mf_vi_demo/full_covariance_example.py``.

.. figure:: notebooks/analytic_sine_correlated_convergence.png
   :alt: KL convergence for correlated VI and MF-VI
   :width: 78%

   Full-covariance VI and MF-VI convergence.

.. figure:: notebooks/analytic_sine_correlated_family_correlation.png
   :alt: Posterior correlation matrices for exact and approximate distributions
   :width: 96%

   Comparison of posterior correlation structure.

Source and further details
--------------------------

- `Diagonal example source <https://github.com/Pressio/rom-tools-and-workflows/blob/develop/examples/vi_mf_vi_demo/diagonal_example.py>`_
- `Full-covariance example source <https://github.com/Pressio/rom-tools-and-workflows/blob/develop/examples/vi_mf_vi_demo/full_covariance_example.py>`_
- :doc:`Detailed benchmark configuration <../vi_benchmark_details>`
- :doc:`VI implementation and optimizer notes <../vi_implementation_notes>`

To regenerate the figures, run
``python examples/vi_mf_vi_demo/generate_docs_media.py``.
