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

For worked sampling examples, see :doc:`demos/space_filling_training` and
:doc:`demos/hybrid_greedy_training`.
