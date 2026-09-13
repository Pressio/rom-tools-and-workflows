Workflow working-directory argument
===================================

Workflow drivers use ``absolute_work_dir`` for their working-directory argument.
The previous workflow-specific keyword names remain accepted for backwards
compatibility and emit a ``DeprecationWarning``. New code should use
``absolute_work_dir``.
