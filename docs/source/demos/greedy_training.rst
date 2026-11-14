Greedy training
===============

**Method.** Greedy ROM training adds full-order samples where the current ROM has the largest estimated error. The model is rebuilt until a stopping criterion is reached.

**Example.** The notebook demonstrates iterative sample selection, ROM rebuilding, and error monitoring.

Notebook
--------

.. toctree::
   :maxdepth: 1

   notebooks/greedy_training.ipynb

For algorithm details, settings, and full source listings, see
:doc:`greedy_training_details`.

.. toctree::
   :hidden:

   greedy_training_details
