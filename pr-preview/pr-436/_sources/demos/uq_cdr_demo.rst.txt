Monte Carlo UQ for convection-diffusion-reaction
================================================

**Method.** Monte Carlo estimates statistics from repeated model evaluations; multifidelity Monte Carlo uses a correlated cheaper model to reduce estimation cost.

**Example.** The example compares standard and two-level multifidelity estimates of the mean right-boundary flux for a steady convection-diffusion-reaction problem.

Run: ``python examples/uq_cdr_demo/example.py``.

For algorithm details, settings, and full source listings, see
:doc:`uq_cdr_demo_details`.

.. toctree::
   :hidden:

   uq_cdr_demo_details
