EKI and MF-EKI Demo
===================

This demo compares single-fidelity EKI with two multifidelity variants on a
convection-diffusion-reaction (CDR) model with two inferred parameters:

* EKI using only the full-order model (FOM),
* MF-EKI using a tailored projection-based ROM, and
* MF-EKI using automatic on-the-fly Gaussian-process ROM construction.

The forward model solves a steady 2D CDR equation on a structured grid, and the
QoI is the right-boundary flux functional used in the existing UQ examples.
The example estimates the diffusion coefficient ``nu`` and reaction rate
``sigma`` from a synthetic observation.

Run the demo
------------

The canonical implementation lives under ``examples/`` and is validated against
the current romtools checkout in CI.

.. code-block:: bash

   python examples/eki_mf_eki_demo/example.py

A reduced configuration is available for quick validation:

.. code-block:: bash

   python examples/eki_mf_eki_demo/example.py --smoke

Regenerate the published figure
-------------------------------

The documentation build does not execute this scientific demo implicitly.
Regenerate the checked-in figure explicitly after changes that affect the
benchmark:

.. code-block:: bash

   python examples/eki_mf_eki_demo/example.py \
       --output docs/source/demos/notebooks/eki_mf_eki_demo.png

Results
-------

.. figure:: notebooks/eki_mf_eki_demo.png
   :alt: EKI and MF-EKI error convergence
   :align: center
   :width: 90%

   Mean observation error across iterations for single-fidelity EKI,
   multifidelity EKI with a tailored ROM, and multifidelity EKI with automatic
   Gaussian-process ROM construction.

Implementation
--------------

.. literalinclude:: ../../../examples/eki_mf_eki_demo/example.py
   :language: python
   :linenos:
