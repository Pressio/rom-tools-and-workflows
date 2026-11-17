DEIM and QDEIM hyper-reduction
==============================

**Method.** Discrete empirical interpolation (DEIM) reconstructs a high-dimensional model term from a small set of sampled entries; QDEIM uses pivoted QR to choose those entries.

**Example.** The examples apply DEIM and QDEIM to the full right-hand side of a two-dimensional H2-air flame model and compare reconstructions on a separate trajectory.

Run: ``python examples/h2_air_flame_deim/example.py`` or
``python examples/h2_air_flame_qdeim/example.py``.

For algorithm details, settings, and full source listings, see
:doc:`hyper_reduction_details`.

.. toctree::
   :hidden:

   hyper_reduction_details
