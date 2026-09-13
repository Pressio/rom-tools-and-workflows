Hyper-reduction
===============

``romtools`` provides discrete empirical interpolation utilities for
approximating high-dimensional model terms from a small set of sampled entries.
The examples below apply DEIM and QDEIM to the complete semi-discrete right-hand
side of the H2-air flame model rather than to only the reaction term. This keeps
the example close to the operation required by a hyper-reduced projection ROM:
construct a basis for the full RHS, evaluate only selected entries, and
reconstruct the full vector.

For a state :math:`x` and parameters :math:`\mu`, let
:math:`f(x;\mu) \in \mathbb{R}^N` denote the full RHS. A POD basis
:math:`U \in \mathbb{R}^{N \times r}` is built from RHS snapshots. DEIM or
QDEIM then selects interpolation indices :math:`P`, and the full RHS is
approximated from the sampled entries by

.. math::

   f(x;\mu) \approx U (P^T U)^{\dagger} P^T f(x;\mu).

The main difference between the two methods in ``romtools`` is the default
sample-point selection. DEIM uses the classical greedy selection algorithm,
while QDEIM uses column-pivoted QR. Their reconstruction APIs are otherwise the
same.

DEIM
----

The shortest path through the public API is:

.. code-block:: python

   from romtools.hyper_reduction import DEIM
   from romtools.vector_space.utils.truncater import BasisSizeTruncater

   deim = DEIM.from_snapshots(
       rhs_training_snapshots,
       truncater=BasisSizeTruncater(10),
   )

   sampled_rhs = full_rhs[deim.sample_indices]
   reconstructed_rhs = deim.reconstruct(sampled_rhs)

``DEIM.from_snapshots`` performs the POD internally, selects the interpolation
points, and constructs the reconstruction operator. The example deliberately
flattens all four H2-air flame fields into one RHS vector, so
``reconstructed_rhs`` approximates the entire coupled RHS.

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

QDEIM uses the same API. Replacing ``DEIM`` with ``QDEIM`` changes the default
sample selection to pivoted QR:

.. code-block:: python

   from romtools.hyper_reduction import QDEIM
   from romtools.vector_space.utils.truncater import BasisSizeTruncater

   qdeim = QDEIM.from_snapshots(
       rhs_training_snapshots,
       truncater=BasisSizeTruncater(10),
   )

   sampled_rhs = full_rhs[qdeim.sample_indices]
   reconstructed_rhs = qdeim.reconstruct(sampled_rhs)

The same ``sample_indices`` and ``reconstruct`` interface can therefore be used
when comparing DEIM and QDEIM in an application.

Run the example with

.. code-block:: bash

   python examples/h2_air_flame_qdeim/example.py

Full implementation
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/h2_air_flame_qdeim/example.py
   :language: python
   :linenos:
