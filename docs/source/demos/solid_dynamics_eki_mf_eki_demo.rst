Solid-dynamics EKI and MF-EKI benchmark
=======================================

This example uses the nonlinear cantilever in the :doc:`solid dynamics model
<nonlinear_solid_dynamics>` as a controlled inverse problem for comparing
single-fidelity ensemble Kalman inversion (EKI) and multifidelity EKI (MF-EKI).
The first benchmark intentionally changes only numerical resolution between the
high- and low-fidelity models; both use the same compressible Neo-Hookean
physics.  This makes it possible to study the effect of fidelity correlation
without introducing model-form error at the same time.

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
generated on the truth discretization and perturbed with reproducible Gaussian
noise.  A single standard deviation equal to one percent of the largest clean
observed displacement is used for the initial benchmark,

.. math::

   y = \mathcal G_{\mathrm{truth}}(\theta^*) + \eta,
   \qquad
   \eta\sim\mathcal N(0,\sigma_{\mathrm{obs}}^2 I).

The truth discretization is finer than the inference high-fidelity model, so
the inverse problem does not use exactly the same discrete model to generate and
fit the observations.

Fidelity hierarchy
------------------

The default non-smoke configuration uses

.. list-table::
   :header-rows: 1

   * - Model
     - Q4 mesh
     - Time step
     - Time horizon
   * - Synthetic truth
     - ``12 x 4``
     - ``0.005 s``
     - ``1.0 s``
   * - High fidelity
     - ``8 x 2``
     - ``0.01 s``
     - ``1.0 s``
   * - Low fidelity
     - ``4 x 2``
     - ``0.02 s``
     - ``1.0 s``

The MF-EKI low-fidelity model is passed through the normal
``QoiModelBuilderWithTrainingData``-style interface, but the builder returns a
fixed coarse discretization rather than learning a ROM from the high-fidelity
snapshots.  This is deliberate: the benchmark first asks how MF-EKI behaves
when the low-fidelity error is ordinary spatial/temporal discretization error.
A later experiment can replace the coarse Neo-Hookean model with linear
elasticity to introduce controlled model-form error.

Cost metric
-----------

Every model evaluation stores both its wall-clock time and a deterministic work
proxy

.. math::

   W \propto N_{\mathrm{elem}}N_{\mathrm{step}}.

The high-fidelity model has unit cost and the low-fidelity evaluations are
weighted by their ratio to the high-fidelity work proxy.  The main comparison
therefore plots observation error against cumulative **HF-equivalent work**.
Wall-clock totals are also written to the summary JSON for reference, but they
are not used as the primary algorithmic metric because they are machine
dependent.

Pilot correlation
-----------------

Before the full solve, the example can evaluate a small pilot ensemble on the
high- and low-fidelity models and report componentwise HF/LF QoI correlations.
This provides a direct diagnostic for whether the coarse model is useful to
MF-EKI.  The example reports the minimum, median, and maximum correlation across
all sensor/time components.

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
* ``solid_dynamics_parameter_convergence.png`` for the EKI and MF-EKI parameter
  estimates;
* ``solid_dynamics_error_vs_cost.png`` for convergence versus HF-equivalent
  work; and
* ``solid_dynamics_eki_mf_eki_summary.json`` containing the truth, final
  estimates, pilot correlations, nominal costs, and wall-clock totals.

Implementation
--------------

.. literalinclude:: ../../../examples/solid_dynamics_eki_mf_eki/example.py
   :language: python
   :linenos:
