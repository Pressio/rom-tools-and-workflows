Inverse Problems
================

Parameter inference with ensemble Kalman inversion and variational inference.

.. autosummary::
   :toctree: generated

   romtools.workflows.inverse.run_eki
   romtools.workflows.inverse.run_mf_eki
   romtools.workflows.inverse.run_vi
   romtools.workflows.inverse.run_mf_vi

For an introduction to the EKI workflows, see :doc:`api_inverse_eki`.
For background on full-covariance VI, see :doc:`full_covariance_vi`.

.. toctree::
   :hidden:

   api_inverse_eki
   vi_implementation_notes
   vi_benchmark_details

Examples
--------

- :doc:`EKI and multifidelity EKI <demos/eki_mf_eki_demo>`
- :doc:`H2-air flame EKI and MF-EKI <demos/h2_air_flame_eki_mf_eki_demo>`
- :doc:`Variational inference examples <demos/variational_inference>`
