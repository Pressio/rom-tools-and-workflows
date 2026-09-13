Solid-dynamics EKI and MF-EKI benchmark
=======================================

This example uses the nonlinear cantilever in the :doc:`solid dynamics model <nonlinear_solid_dynamics>` as a controlled inverse problem for comparing
single-fidelity ensemble Kalman inversion (EKI) and multifidelity EKI (MF-EKI).
Synthetic truth and the inference high-fidelity model intentionally use the
same finite-element mesh and time step.  MF-EKI then constructs a
Gaussian-process QoI surrogate automatically from the high-fidelity evaluations
and uses that GP as the low-fidelity model.

Inverse parameters
------------------

The unknown physical parameters are Young's modulus, Poisson's ratio, and the
amplitude of the transient end load,

.. math::

   \theta = (E,\nu,F_0).

Positive parameters are represented internally in logarithmic coordinates,

.. math::

   \xi = (\log E,\nu,\log F_0),

which keeps the EKI updates well scaled and guarantees positive values when the
model maps back to physical units.  The prior ranges are

.. math::

   E \in [0.6,1.4]\ \mathrm{MPa},\qquad
   \nu \in [0.20,0.40],\qquad
   F_0 \in [700,1300]\ \mathrm{N}.

The synthetic truth is deliberately away from the center of these ranges,

.. math::

   E^*=1.15\ \mathrm{MPa},\qquad
   \nu^*=0.33,\qquad
   F_0^*=900\ \mathrm{N}.

Observation model
-----------------

The beam is clamped on the left and driven by the same smooth half-sine
transient used in the solid-dynamics cantilever example.  The observation
vector concatenates transverse-displacement histories at

.. math::

   x/L = 0.50,\ 0.75,\ 1.00,

sampled every ``0.1 s`` from ``0.1`` through ``1.0 s``.  Synthetic data are
generated with the same high-fidelity model used during inversion and perturbed
with reproducible Gaussian noise.  A single standard deviation equal to one
percent of the largest clean observed displacement is used,

.. math::

   y = \mathcal G_H(\theta^*) + \eta,
   \qquad
   \eta\sim\mathcal N(0,\sigma_{\mathrm{obs}}^2 I).

This is intentionally a controlled algorithmic benchmark rather than a study of
discretization mismatch: truth and high fidelity share the same discrete
forward model.

High-fidelity model
-------------------

The default non-smoke benchmark uses the same discretization for synthetic truth
and every high-fidelity evaluation:

.. list-table::
   :header-rows: 1

   * - Model
     - Q4 mesh
     - Time step
     - Time horizon
   * - Synthetic truth / high fidelity
     - ``33 x 16``
     - ``0.005 s``
     - ``1.0 s``

Both EKI and MF-EKI therefore query the same ``33 x 16`` Neo-Hookean FOM.  The
``--smoke`` path uses a much smaller discretization solely to keep CI validation
lightweight.

Automatic GP low-fidelity model
--------------------------------

MF-EKI is run through ``mf_eki_with_auto_rom`` with ``rom_type="gp"``.  The GP
maps the three transformed parameters

.. math::

   (\log E,\nu,\log F_0)

onto the concatenated sensor time histories.  It is rebuilt from accumulated
high-fidelity parameter/QoI pairs when its error exceeds the MF-EKI ROM
tolerance.  Parameters and QoIs are normalized before GP construction.

The default benchmark uses

* 8 shared FOM ensemble members;
* 24 additional GP-only ensemble members;
* up to 4 GP-only EKI substeps after each outer high-fidelity update;
* a relative ROM tolerance of ``0.005``; and
* at most 5 recent high-fidelity training batches when rebuilding the GP.

This setup isolates the benefit of an automatically constructed data-driven
surrogate: there is no separate coarse finite-element model to tune.

Cost metric
-----------

The dominant cost is the ``33 x 16`` nonlinear finite-element solve.  The main
comparison therefore plots observation error against cumulative high-fidelity
model evaluations.  GP evaluations are treated as negligible in this primary
cost metric.  The example also records cumulative high-fidelity wall-clock time
for both EKI and MF-EKI in the summary JSON.

Running the benchmark
---------------------

Run the full study with

.. code-block:: bash

   python examples/solid_dynamics_eki_mf_eki/example.py

A reduced configuration is available for CI:

.. code-block:: bash

   python examples/solid_dynamics_eki_mf_eki/example.py --smoke

The script writes

* ``solid_dynamics_observations.png`` for the clean/noisy sensor histories;
* ``solid_dynamics_parameter_convergence.png`` for the EKI and GP MF-EKI
  parameter estimates;
* ``solid_dynamics_error_vs_cost.png`` for convergence versus cumulative
  high-fidelity evaluations; and
* ``solid_dynamics_eki_mf_eki_summary.json`` containing the truth, FOM and GP
  settings, final estimates, high-fidelity evaluation counts, and high-fidelity
  wall-clock totals.

Implementation
--------------

.. literalinclude:: ../../../examples/solid_dynamics_eki_mf_eki/example.py
   :language: python
   :linenos:
