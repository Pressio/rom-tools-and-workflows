VI and MF-VI Demo
=================

This example compares single-fidelity variational inference with multifidelity
variational inference for a steady convection-diffusion-reaction model. Both
methods approximate the posterior over the diffusion coefficient ``nu`` and
reaction rate ``sigma`` with a Gaussian distribution.

The multifidelity method supplements its full-order-model samples with
additional evaluations of a Gaussian-process surrogate. The surrogate is
constructed automatically from the full-order-model parameter and QoI data,
with both inputs and targets normalized before fitting.

Run the example
---------------

The canonical implementation lives under ``examples/`` and is exercised
against the current romtools checkout in CI.

.. code-block:: bash

   python examples/vi_mf_vi_demo/example.py

A reduced configuration is available for quick validation:

.. code-block:: bash

   python examples/vi_mf_vi_demo/example.py --smoke

The full run uses 50 optimization iterations, eight FOM samples per iteration,
and 64 additional GP-ROM samples for MF-VI. The smoke configuration reduces
the grid, sample counts, and optimization iterations without changing the
workflow API being tested.

Regenerate the published figures
--------------------------------

The Sphinx build does not execute scientific examples implicitly. Regenerate
the checked-in figures explicitly after changes that affect this benchmark:

.. code-block:: bash

   python examples/vi_mf_vi_demo/example.py \
       --output-dir docs/source/demos/notebooks

Results
-------

.. figure:: notebooks/vi_mf_vi_elbo_convergence.png
   :alt: ELBO convergence for single-fidelity VI and multifidelity VI
   :align: center
   :width: 80%

   ELBO histories for single-fidelity VI and MF-VI with automatic GP-ROM
   construction.

.. figure:: notebooks/vi_mf_vi_parameter_convergence.png
   :alt: Posterior parameter convergence for VI and multifidelity VI
   :align: center
   :width: 95%

   Variational means and one-standard-deviation bands for the inferred
   diffusion coefficient and reaction rate.

Implementation
--------------

.. literalinclude:: ../../../examples/vi_mf_vi_demo/example.py
   :language: python
   :linenos:
