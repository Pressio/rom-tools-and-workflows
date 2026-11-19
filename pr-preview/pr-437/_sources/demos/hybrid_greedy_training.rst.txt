Hybrid greedy and space-filling training
========================================

**Method.** Hybrid training combines one error-indicator-selected point with additional maximin space-filling points in each batch. Validation determines when training stops.

**Example.** The example uses an adaptive batch to balance refinement in inaccurate regions with broader parameter-domain coverage.

Run: ``python examples/hybrid_greedy_demo.py --work-dir /tmp/hybrid-greedy-demo``.

For algorithm details, settings, and full source listings, see
:doc:`hybrid_greedy_training_details`.

.. toctree::
   :hidden:

   hybrid_greedy_training_details
