Monte Carlo UQ for Convection-Diffusion-Reaction
================================================

This example estimates the expected integrated right-boundary flux of the
steady convection-diffusion-reaction (CDR) equation. Four uncertain inputs are
sampled independently: advection magnitude, advection direction, diffusion
coefficient, and reaction coefficient.

The standard Monte Carlo calculation uses the existing 21-by-21 CDR model. The
multifidelity calculation treats that grid as high fidelity and a 9-by-9 grid
as low fidelity. Four paired pilot samples estimate their correlation and
select the final high- and low-fidelity counts under a budget of 12
high-fidelity-equivalent evaluations. The low-fidelity evaluation cost is set
to 5% of a high-fidelity evaluation for this demonstration.

Run the example
---------------

From the repository root:

.. code-block:: bash

   python examples/uq_cdr_demo/example.py

Both workflows save their statistics, allocation metadata, samples, and model
run directories below ``examples/uq_cdr_demo/uq_cdr_output``.

Implementation
--------------

.. literalinclude:: ../../../examples/uq_cdr_demo/example.py
   :language: python
   :linenos:
