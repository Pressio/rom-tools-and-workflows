Hyper-reduction
===============

``romtools`` provides discrete empirical interpolation utilities for
approximating high-dimensional model terms from a small set of sampled entries.
The examples below apply DEIM and QDEIM to the complete semi-discrete right-hand
side of the H2-air flame model rather than to only the reaction term. This keeps
the example close to the operation required by a hyper-reduced projection ROM:
construct a basis for the full RHS, evaluate only selected entries, and
reconstruct the full tensor.

Tensor convention and shared-state sampling
-------------------------------------------

The ``DEIM`` and ``QDEIM`` class APIs use the standard ``romtools`` tensor
layout. Function snapshots have shape
``(n_vars, n_dofs, n_snapshots)`` and function bases have shape
``(n_vars, n_dofs, n_basis)``. The first two axes are collapsed internally in
explicit C order whenever the DEIM algebra requires a matrix representation.
Users therefore do not need to flatten multistate data before constructing a
hyper-reduction operator.

For a state :math:`x` and parameters :math:`\mu`, let
:math:`f(x;\mu)` denote the full multistate RHS. Internally, a POD basis
:math:`U \in \mathbb{R}^{N \times r}` is formed from the tensor snapshots,
where :math:`N = n_{vars} n_{dofs}`. DEIM or QDEIM first selects scalar rows of
this matrix, and the full RHS is approximated by

.. math::

   f(x;\mu) \approx U (P^T U)^{\dagger} P^T f(x;\mu).

By default, ``sample_all_states=True``. A scalar row selected for any state is
converted to its spatial index, and every state at that spatial point is then
included in the sampling operator. Thus ``sample_indices`` contains spatial
sample points, and a sampled RHS has shape ``(n_vars, n_sample_points)``. This
is usually the desired behavior for coupled PDE discretizations because an
application evaluates all state components at a selected mesh point.

Independent flattened state-DOF sampling remains available with
``sample_all_states=False`` for applications that explicitly need it.

The main difference between DEIM and QDEIM is the initial scalar point
selection. DEIM uses the classical greedy selection algorithm, while QDEIM uses
column-pivoted QR. Their tensor handling and reconstruction APIs are otherwise
the same.

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
   )

   # full_rhs.shape == (n_vars, n_dofs)
   sampled_rhs = full_rhs[:, deim.sample_indices]
   reconstructed_rhs = deim.reconstruct(sampled_rhs)

``DEIM.from_snapshots`` performs the POD internally, selects the interpolation
points, expands those spatial points across all state variables by default, and
constructs the reconstruction operator. In the H2-air flame example, the four
RHS fields are kept as the first tensor axis while the two spatial dimensions
are collapsed into the ``n_dofs`` axis using C ordering.

The training snapshots come from one flame trajectory, while reconstruction is
measured on a second trajectory at different parameter values. This avoids
measuring only interpolation of the data used to construct the basis.

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

QDEIM uses the same tensor-native API. Replacing ``DEIM`` with ``QDEIM`` changes
the initial scalar sample selection to pivoted QR:

.. code-block:: python

   from romtools.hyper_reduction import QDEIM
   from romtools.vector_space.utils.truncater import BasisSizeTruncater

   qdeim = QDEIM.from_snapshots(
       rhs_training_snapshots,
       truncater=BasisSizeTruncater(10),
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
