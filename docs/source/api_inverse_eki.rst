Ensemble Kalman Inversion (EKI)
===============================

The EKI workflows provide single- and multi-fidelity ensemble Kalman inversion
utilities for parameter inference problems. Both workflows are derivative-free
with respect to the forward model, support concurrent sample evaluation, and
write restart files that can be used to resume long-running solves.

Single-fidelity EKI updates a parameter ensemble with Kalman-style covariance
estimates formed from forward-model QoIs. Multi-fidelity EKI preserves the
same high-fidelity inverse target while augmenting the update with a
surrogate-based control-variate correction built from shared FOM/ROM samples
and additional ROM-only samples.

For more detailed algorithmic background, see the function-level API
documentation for the single- and multi-fidelity drivers below.

Single-fidelity EKI
-------------------

.. autofunction:: romtools.workflows.inverse.eki_drivers.run_eki
   :no-index:

See also:

- :doc:`generated/romtools.workflows.inverse.run_eki`

Multi-fidelity EKI
------------------

.. autofunction:: romtools.workflows.inverse.mf_eki_drivers.run_mf_eki
   :no-index:

See also:

- :doc:`generated/romtools.workflows.inverse.run_mf_eki`
