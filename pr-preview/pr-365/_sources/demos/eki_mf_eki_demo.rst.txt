EKI and MF-EKI Demo
===================

This demo compares single-fidelity EKI with three multifidelity variants on a
convection-diffusion-reaction (CDR) model with two inferred parameters:

* EKI using only the full-order model (FOM),
* MF-EKI using a tailored projection-based ROM,
* MF-EKI using automatic on-the-fly Gaussian-process surrogate construction,
  and
* MF-EKI using an automatically rebuilt Lipschitz-constrained neural-network
  surrogate.

The forward model solves a steady 2D CDR equation on a structured grid, and the
QoI is the right-boundary flux functional used in the existing UQ examples.
The example estimates the diffusion coefficient ``nu`` and reaction rate
``sigma`` from a synthetic observation.

Consistent solver configuration
-------------------------------

All four curves use the same EKI solver controls: an initial step size of
``0.05``, regularization parameter ``1e-4``, step-size growth and decay factors
of ``1.25`` and ``2.0``, relaxation parameter ``1.05``, observation-error
tolerance ``1e-5``, parameter-update tolerance ``1e-6``, and random seed ``1``.
The three MF-EKI cases also use the same FOM and auxiliary ensemble sizes, ROM
substep schedule, ROM error tolerance, training-history length, and policy for
using the previous surrogate in an update when a rebuild occurs. This keeps the
comparison focused on surrogate construction rather than differences in
workflow defaults.

Neural-network surrogate
------------------------

The neural-network case uses the standard romtools neural surrogate with two
hidden layers and a width of three times the parameter dimension. The published
benchmark uses 5,000 Adam iterations for each surrogate rebuild. Parameters and
targets are normalized before training.

The network is constrained to be Lipschitz using spectral normalization. The
global Lipschitz constant is estimated from the maximum pairwise slope in the
normalized training data and multiplied by a safety factor of ``1.1``. The
resulting global bound is distributed evenly over the linear layers. This gives
the neural surrogate an explicit smoothness constraint while retaining the same
adaptive rebuild logic used by the other MF-EKI surrogates.

PyTorch is an optional dependency. Install neural-network support with

.. code-block:: bash

   pip install "romtools[WithTorch]"

Run the demo
------------

The canonical implementation lives under ``examples/`` and is validated against
the current romtools checkout in CI.

.. code-block:: bash

   python examples/eki_mf_eki_demo/example.py

A reduced configuration is available for quick validation. It uses the same
four model paths but reduces the neural-network training to 100 Adam iterations:

.. code-block:: bash

   python examples/eki_mf_eki_demo/example.py --smoke

Regenerate the published figure
-------------------------------

The documentation build does not execute this scientific demo implicitly.
Regenerate the checked-in figure explicitly after changes that affect the
benchmark:

.. code-block:: bash

   python examples/eki_mf_eki_demo/example.py \
       --output docs/source/demos/notebooks/eki_mf_eki_demo.svg

Results
-------

.. figure:: notebooks/eki_mf_eki_demo.svg
   :alt: EKI and MF-EKI error convergence
   :align: center
   :width: 90%

   Mean observation error across iterations for single-fidelity EKI,
   multifidelity EKI with a tailored ROM, multifidelity EKI with automatic
   Gaussian-process surrogate construction, and multifidelity EKI with a
   Lipschitz-constrained neural-network surrogate.

Implementation
--------------

.. literalinclude:: ../../../examples/eki_mf_eki_demo/example.py
   :language: python
   :linenos:
