API Reference
=============

Public API and compatibility
----------------------------

For the 1.x release series, the supported public API consists of documented
interfaces in this API reference and names explicitly listed in package
``__all__`` definitions. Public interfaces should not be removed or changed
incompatibly without a deprecation cycle.

New code should prefer domain-oriented imports, for example::

   from romtools.vector_space import VectorSpaceFromPOD
   from romtools.hyper_reduction import DEIM
   from romtools.rom import GaussianProcessQoiModel
   from romtools.workflows import run_sampling
   from romtools.workflows.inverse import run_eki

The top-level ``romtools`` namespace primarily exposes package metadata and
major subpackages. Historical flat aliases remain available for backwards
compatibility, but they are not the preferred import style.

Names and modules beginning with an underscore, along with implementation
names that are neither documented as public nor listed in ``__all__``, are
internal and may change without a deprecation cycle.

Hyper-reduction import compatibility
------------------------------------

The procedural DEIM and ECSW functions remain supported at their historical
package and root import paths throughout the 1.x series. For example,
``from romtools import deim_get_indices`` and
``from romtools.hyper_reduction import deim_get_indices`` refer to the same
function as ``romtools.hyper_reduction.deim.deim_get_indices``. No migration is
required, and no deprecation warning is emitted. New code should prefer the
hyper-reduction namespace or the ``DEIM``/``QDEIM`` classes where appropriate.

The retained procedural exports are ``qdeim_get_indices``, ``deim_get_indices``,
``multi_state_deim_get_indices``, ``deim_get_approximation_matrix``,
``multi_state_deim_get_test_basis``, ``deim_get_test_basis``,
``ecsw_fixed_test_basis``, ``ecsw_varying_test_basis``, and
``ecsw_lspg_zero_residual``. Any future incompatible change to these supported
imports must follow the public deprecation policy.

The reference below lists explicit package exports rather than recursively
listing implementation modules. A submodule exposed for navigation is not a
promise that all of its helpers are public; the compatibility guarantee applies
to its explicitly exported interfaces and interfaces documented as public.

Workflow working directories
----------------------------

Workflow drivers use ``absolute_work_dir`` for their working-directory argument.
Previous workflow-specific keyword names remain accepted for backwards
compatibility and emit a ``DeprecationWarning``. New code should use
``absolute_work_dir``.

.. autosummary::
   :toctree: generated

   romtools
   romtools.composite_vector_space
   romtools.hyper_reduction
   romtools.linalg
   romtools.rom
   romtools.vector_space
   romtools.workflows
   romtools.vector_space.utils
   romtools.workflows.inverse
   romtools.workflows.sampling
   romtools.workflows.greedy
   romtools.workflows.uq
   romtools.workflows.models
   romtools.workflows.parameter_spaces
   romtools.vector_space.utils.scaler
   romtools.vector_space.utils.orthogonalizer

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
   Variational Inference <generated/romtools.workflows.inverse.run_vi>
   Multifidelity Variational Inference <generated/romtools.workflows.inverse.run_mf_vi>
