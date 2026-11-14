Batch space-filling ROM training
================================

**Method.** Space-filling training selects parameter points to cover the domain without relying on a ROM error indicator. New full-order simulations are run in batches and the ROM is validated after each batch.

**Example.** The example trains a Gaussian-process surrogate on an irregular disk-shaped parameter domain using file-backed model evaluations and cross-validation.

Run: ``python examples/space_filling_demo.py --work-dir /tmp/space-filling-demo``.

For algorithm details, settings, and full source listings, see
:doc:`space_filling_training_details`.

.. toctree::
   :hidden:

   space_filling_training_details
