Steady convection-diffusion-reaction model
==========================================

The examples include a self-contained steady convection-diffusion-reaction
(CDR) model that is used throughout the romtools workflow demonstrations. It
is intentionally small and inexpensive, making it useful for testing romtools
interfaces and workflows without an external simulation code.

The implementation lives in ``examples/models`` and requires only NumPy and
SciPy.

Problem definition
------------------

The model solves a scalar steady CDR equation on the unit square,

.. math::

   \nu \nabla^2 u - \boldsymbol{b}\cdot\nabla u - \sigma u = -1,

with homogeneous boundary data represented by the discrete operators. The
four model parameters are

``bmag``
   Magnitude of the advection velocity.

``theta``
   Advection direction in radians. The velocity vector is

   .. math::

      \boldsymbol{b} = b_{\mathrm{mag}}
      [\cos(\theta),\,\sin(\theta)]^T.

``nu``
   Diffusion coefficient.

``sigma``
   Reaction coefficient.

The spatial discretization uses a second-order central stencil for diffusion
and upwind finite differences for advection. The resulting sparse linear
system is assembled with SciPy and solved with ``scipy.sparse.linalg.spsolve``.

Direct use
----------

The PDE solver can be used independently of the romtools workflow interfaces.
From ``examples/models``:

.. code-block:: python

   import numpy as np
   import steady_cdr as cdr

   system = cdr.AdvectionDiffusionSystem(nx=21, ny=21)
   bmag = 0.5
   theta = np.pi / 3.0
   b = bmag * np.array([np.cos(theta), np.sin(theta)])

   state = cdr.solveFom(
       system,
       b=b,
       nu=1.0e-3,
       sigma=1.0,
   )

``state`` contains the solution values at the ``nx * ny`` interior grid
points.

romtools QoI model
------------------

``SteadyCdrQoiModel`` wraps the solver using the standard romtools QoI-model
protocol. It implements ``populate_run_directory``, ``run_model``, and
``compute_qoi`` so the model can be passed directly to romtools workflows.
For each evaluation, the wrapper stores the state, model parameters, and a
one-sided boundary-derivative QoI in ``solution.npz``.

A representative parameter sample is

.. code-block:: python

   sample = {
       "bmag": 0.5,
       "theta": np.pi / 3.0,
       "nu": 1.0e-3,
       "sigma": 1.0,
   }

The complete wrapper can be exercised from the repository root with

.. code-block:: bash

   python examples/models/steady_cdr_model.py

The script performs one model evaluation and reports the size and norm of the
resulting QoI.

Related examples
----------------

Several demos use this model or closely related CDR configurations. See the
uncertainty-quantification and inverse-workflow sections for examples of how a
``QoiModel`` is consumed by higher-level romtools algorithms.

Implementation
--------------

Core CDR operators and solver:

.. literalinclude:: ../../../examples/models/steady_cdr.py
   :language: python
   :linenos:

romtools QoI wrapper:

.. literalinclude:: ../../../examples/models/steady_cdr_model.py
   :language: python
   :linenos:
