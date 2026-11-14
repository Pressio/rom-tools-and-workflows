Sampling & Execution
====================

Model protocols, parameter spaces, sampling, and workflow execution.

.. autosummary::
   :toctree: generated

   romtools.workflows.models
   romtools.workflows.parameter_spaces
   romtools.workflows.sampling

Workflow working directories
----------------------------

Workflow drivers use ``absolute_work_dir`` for their working-directory
argument. Previous workflow-specific keyword names remain accepted for
backwards compatibility and emit a ``DeprecationWarning``. New code should
use ``absolute_work_dir``.

Examples
--------

- :doc:`Model interfaces <demos/models>`
- :doc:`Parameter-space sampling <demos/notebooks/parameter_space>`
- :doc:`Basic sampling workflow <demos/notebooks/basic_sampling>`
- :doc:`Space-filling training <demos/space_filling_training>`
- :doc:`Hybrid greedy training <demos/hybrid_greedy_training>`
