API Reference
=============

Workflow working directories
----------------------------

Workflow drivers use ``absolute_work_dir`` for their working-directory argument.
Previous workflow-specific keyword names remain accepted for backwards
compatibility and emit a ``DeprecationWarning``. New code should use
``absolute_work_dir``.

.. autosummary::
   :toctree: generated
   :recursive:

   romtools
   romtools.composite_vector_space
   romtools.hyper_reduction
   romtools.linalg
   romtools.rom
   romtools.vector_space
   romtools.workflows

.. toctree::
   :maxdepth: 1
   :caption: Utilities

   formatting
   Distributed SVD <distributed_svd>

.. toctree::
   :maxdepth: 1
   :caption: ROM Construction

   Vector Space <generated/romtools.vector_space>
   Composite Vector Space <generated/romtools.composite_vector_space>
   Hyper Reduction <generated/romtools.hyper_reduction>

.. toctree::
   :maxdepth: 1
   :caption: Inverse Workflows

   api_inverse_eki

.. toctree::
   :maxdepth: 1
   :caption: Uncertainty Quantification

   api_uq

.. toctree::
   :hidden:

   Ensemble Kalman Inversion <generated/romtools.workflows.inverse.run_eki>
   Multifidelity Ensemble Kalman Inversion <generated/romtools.workflows.inverse.run_mf_eki>
   Variational Inference <generated/romtools.workflows.inverse.vi_drivers>
   Multifidelity Variational Inference <generated/romtools.workflows.inverse.mf_vi_drivers>
