H2-air flame EKI and MF-EKI rejuvenation benchmark
==================================================

This example uses the four-parameter :doc:`h2_air_flame` model to study
ensemble collapse in ensemble Kalman inversion and the effect of adaptive
MF-EKI rejuvenation. The inferred parameters are diffusivity ``kappa``, scaled
activation energy, and the two advection components ``beta_x`` and ``beta_y``.

Benchmark design
----------------

The benchmark compares three cases using the same synthetic observation and
random seed:

* single-fidelity EKI with four FOM ensemble members,
* GP auto-ROM MF-EKI with four FOM members and 32 additional ROM-only members,
  with rejuvenation disabled, and
* the same MF-EKI configuration with ``rejuvenation_strategy="adaptive"``.

The full model uses a ``64 x 32`` mesh, ``dt = 1e-3``, and ``t_end = 6e-2``.
The full benchmark permits up to 30 outer iterations. Adaptive rejuvenation is
triggered when the parameter-update norm falls below ``1e-4`` while the
observation residual is still above its convergence tolerance. At most three
rejuvenations are allowed.

When the empirical ensemble covariance collapses, the default parameter-scaled
reference term corresponds to a 5% standard-deviation perturbation around the
current parameter mean. Parameter bounds supply the fallback scale for a
near-zero mean.

Diagnostics
-----------

The benchmark records three diagnostics at every saved outer iteration:

* mean observation residual,
* RMS componentwise relative error of the FOM-ensemble parameter mean, and
* range-scaled ensemble spread. For MF-EKI, the spread uses the combined FOM
  and ROM-only parameter ensembles.

.. figure:: ../_static/h2_air_flame_eki_rejuvenation_benchmark.svg
   :alt: H2-air flame EKI and MF-EKI rejuvenation benchmark
   :align: center
   :width: 100%

   Observation residual, parameter error, and ensemble spread for
   single-fidelity EKI, baseline MF-EKI, and MF-EKI with adaptive rejuvenation.

Run the example
---------------

From the repository root:

.. code-block:: bash

   python examples/h2_air_flame_eki_mf_eki/example.py

A reduced API-validation run is available with:

.. code-block:: bash

   python examples/h2_air_flame_eki_mf_eki/example.py --smoke

The full benchmark writes an SVG/PNG figure and a JSON file containing the
complete histories and final summary values.

Implementation
--------------

.. literalinclude:: ../../../examples/h2_air_flame_eki_mf_eki/example.py
   :language: python
   :linenos:
