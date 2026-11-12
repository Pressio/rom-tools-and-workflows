Hyper-reduction
===============

``romtools`` provides discrete empirical interpolation utilities for
approximating high-dimensional model terms from a small set of sampled entries.
The examples below apply DEIM and QDEIM to the complete semi-discrete right-hand
side of the H2-air flame model rather than to only the reaction term. This keeps
the example close to the operation required by a hyper-reduced projection ROM:
construct a basis for the full RHS, evaluate only selected entries, and
reconstruct the full tensor.

Tensor convention and multistate systems
----------------------------------------

The ``DEIM`` and ``QDEIM`` class APIs use the standard ``romtools`` tensor
layout. Function snapshots have shape
``(n_vars, n_dofs, n_snapshots)`` and tensor-form function bases have shape
``(n_vars, n_dofs, n_basis)``. Users therefore do not need to flatten
multistate data before constructing a hyper-reduction operator.

Per-state basis construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default ``basis_mode="per_state"`` is intended for multistate PDE systems.
A separate POD basis is constructed for each state variable,

.. math::

   F_i \approx U_i A_i, \qquad i=1,\ldots,n_{vars},

and DEIM or QDEIM is applied independently to each :math:`U_i`. The resulting
state-specific spatial sample sets :math:`S_i` are combined into one sample
mesh,

.. math::

   S = \bigcup_i S_i.

Every state variable is evaluated on this union mesh. Each state is then
reconstructed with all points in the union,

.. math::

   f_i \approx U_i (U_i[S,:])^{\dagger} f_i[S].

Thus the union acts as an oversampled DEIM system for each state rather than
throwing away points selected by another state. ``sample_indices`` contains the
union of spatial sample points and a sampled RHS has shape
``(n_vars, n_sample_points)``. ``state_sample_indices`` exposes the individual
pre-union DEIM/QDEIM selections for diagnostics.

The per-state formulation avoids combining variables with different physical
units in a single POD/SVD. It is also invariant to independent constant scaling
of the state variables, and an energy-based truncater can naturally retain a
different number of modes for each state. The retained dimensions are available
through ``basis_sizes``.

Global coupled basis
~~~~~~~~~~~~~~~~~~~~

A coupled basis remains available with ``basis_mode="global"``. In this mode,
the first two tensor axes are collapsed internally using explicit C ordering,

.. math::

   F \in \mathbb{R}^{(n_{vars}n_{dofs})\times n_{snapshots}},

and one POD/DEIM problem is constructed for the full multistate vector. This can
exploit correlations between state variables, but the variables should first be
nondimensionalized or weighted consistently so that the Euclidean inner product
used by the SVD is physically meaningful.

With a global basis, ``sample_all_states=True`` remains the default. A scalar
DEIM/QDEIM row selected for any state is converted to its spatial index and all
states are sampled at that point. Independent flattened state-DOF sampling is
available only in global mode with ``sample_all_states=False``.

All tensor-to-matrix and matrix-to-tensor transformations in the global path use
explicit ``order="C"`` so that the state-major ordering is unambiguous.

DEIM
----

The shortest path through the public API is:

.. code-block:: python

   from romtools.hyper_reduction import DEIM
   from romtools.vector_space.utils.truncater import BasisSizeTruncater

   # rhs_training_snapshots.shape == (n_vars, n_dofs, n_snapshots)
   deim = DEIM.from_snapshots(
       rhs_training_snapshots,
       truncater=BasisSizeTruncater(10),
       basis_mode="per_state",
   )

   # full_rhs.shape == (n_vars, n_dofs)
   sampled_rhs = full_rhs[:, deim.sample_indices]
   reconstructed_rhs = deim.reconstruct(sampled_rhs)

``DEIM.from_snapshots`` performs a separate POD for each state by default,
applies DEIM to each retained basis, unions the selected spatial points, and
constructs an oversampled reconstruction operator for each state. With
``BasisSizeTruncater(10)`` each state retains ten modes. With an energy-based
truncater, the individual states may retain different numbers of modes.

In the H2-air flame example, the four RHS fields are kept as the first tensor
axis while the two spatial dimensions are collapsed into the ``n_dofs`` axis
using C ordering. The training snapshots come from one flame trajectory, while
reconstruction is measured on a second trajectory at different parameter
values. This avoids measuring only interpolation of the data used to construct
the basis.

Run the example with

.. code-block:: bash

   python examples/h2_air_flame_deim/example.py

Full implementation
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/h2_air_flame_deim/example.py
   :language: python
   :linenos:

QDEIM
-----

QDEIM uses the same tensor-native and per-state API. Replacing ``DEIM`` with
``QDEIM`` changes the sample selection performed on each state basis to pivoted
QR:

.. code-block:: python

   from romtools.hyper_reduction import QDEIM
   from romtools.vector_space.utils.truncater import BasisSizeTruncater

   qdeim = QDEIM.from_snapshots(
       rhs_training_snapshots,
       truncater=BasisSizeTruncater(10),
       basis_mode="per_state",
   )

   sampled_rhs = full_rhs[:, qdeim.sample_indices]
   reconstructed_rhs = qdeim.reconstruct(sampled_rhs)

The same spatial ``sample_indices`` and tensor ``reconstruct`` interface can
therefore be used when comparing DEIM and QDEIM in an application.

Run the example with

.. code-block:: bash

   python examples/h2_air_flame_qdeim/example.py

Full implementation
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/h2_air_flame_qdeim/example.py
   :language: python
   :linenos:
