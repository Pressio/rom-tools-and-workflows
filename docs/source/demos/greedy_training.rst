Greedy Training
=======================

Greedy training is an iterative training process where we iteratively add samples at which we believe the ROM is inaccurate.

Initial training and iteration limits
------------------------------------

Without explicit initial samples, ``initial_sample_size`` defaults to 2. Greedy
runs those first samples from the initial candidate pool before constructing
the initial ROM. This is separate
from ``testing_sample_size`` (default 10), which sets the candidate-pool size.

``max_iterations`` defaults to 100 and counts FOM enrichments **after** the
initial ROM. Thus at most ``initial_sample_size + max_iterations`` training
FOM simulations run. Setting ``max_iterations=0`` constructs only the initial
ROM. Both arguments are keyword-only, preserving existing positional calls:

.. code-block:: python

   run_greedy(
       fom_model, rom_model_builder, parameter_space,
       absolute_work_dir=work_dir,
       tolerance=1e-4,
       testing_sample_size=50,
       initial_sample_size=8,
       max_iterations=40,
   )

Sizes must be positive integers; the iteration limit must be a nonnegative
integer. When ``initial_samples`` is absent, ``initial_sample_size`` cannot
exceed ``testing_sample_size``; if enrichment is enabled, the candidate pool
must be strictly larger than the initial training set. Convergence checks retain
their previous calibration behavior: the first check follows one enrichment.
The iteration cap stops after the final allowed rebuild; a cap exit does not
claim convergence or perform another candidate evaluation.

FOM directories are populated once, immediately before their simulation, and
unselected FOM candidates are not prepared. Each ROM evaluation directory is
also populated once in its iteration. ``greedy_stats.npz`` is written even for
an initial-only run and records ``initial_sample_size``, ``max_iterations``,
completed ``iterations``, ``converged``, and ``termination_reason`` (``tolerance``,
``max_iterations``, or ``candidate_pool_exhausted``). Error histories describe
completed enrichments.

Explicit initial samples
------------------------

Pass ``initial_samples`` as a nonempty two-dimensional array of parameter rows,
with columns in ``parameter_space.get_names()`` order:

.. code-block:: python

   result = run_greedy(
       fom_model, rom_model_builder, parameter_space,
       absolute_work_dir=work_dir,
       tolerance=1e-4,
       initial_samples=np.array([[0.1, 0.01], [0.8, 0.05], [0.5, 0.02]]),
       testing_sample_size=50,
   )

The workflow runs FOM simulations at these points in supplied order before any
adaptation. This supplies parameter values, not previously computed FOM outputs.
The row count overrides ``initial_sample_size``. The workflow draws a separate
pool of ``testing_sample_size`` adaptation candidates, so the supplied training
set may be larger than that pool. Supplied rows are copied and are not projected
into the parameter domain; numeric values must be finite and the caller is
responsible for domain validity.

When explicit initial samples are used, exact matches to accumulated training
rows are excluded from adaptation candidates, including replenished candidates.
If no untrained candidates remain, the workflow returns the current ROM with
``candidate_pool_exhausted`` and ``converged=False``. This describes the finite
sampled pool, not a proof of domain coverage. Repeated rows explicitly supplied
by the caller are run as requested.

In-memory ROM evaluation
------------------------

Set ``create_rom_run_directories=False`` for ROMs that do not require per-run
files, such as an in-memory GP with a suitable error-indicator wrapper:

.. code-block:: python

   result = run_greedy(
       fom_model, rom_model_builder, parameter_space,
       absolute_work_dir=work_dir,
       tolerance=1e-4,
       create_rom_run_directories=False,
   )

The default is ``True``. Only per-candidate ``rom_iteration_*/run_*`` creation
is skipped; FOM run directories, offline-data directories, logs, and statistics
are still created. ROM methods are still called with their nominal absolute
run paths, and must work without those directories or writing per-run files.
Builders still receive an existing offline-data directory. A GP used by greedy
must provide ``compute_error_estimate`` in addition to its usual QoI interface.

Returned result
---------------

``run_greedy`` returns ``GreedyResult`` on convergence, iteration-limit exit,
candidate-pool exhaustion, and initial-only runs. Existing callers can continue
to ignore the return value. Access the final model as ``result.rom_model``.

The result includes aligned ``training_parameters``, ``training_qois``, and
``training_dirs``, plus ``training_sample_indices`` into the workflow's combined
sample array. It also provides ``sample_counts`` (initial build and each rebuild),
``max_error_indicators``, ``qoi_errors``, ``predicted_qoi_errors``, ``iterations``,
``converged``, and ``termination_reason``. Arrays are independent copies; the
final model is the actual object returned by the last builder call.

Error histories describe the ROMs before their corresponding enrichments and
are not independent validation of the final ROM. Training data, directories,
and sample-count history are also saved in ``greedy_stats.npz``. File-backed
returned models may still depend on their offline-data directories.

.. autoclass:: romtools.workflows.greedy.GreedyResult
   :members:

Builder interface
-----------------

``run_greedy`` uses ``QoiModelBuilderWithTrainingData``, the same data-aware
builder interface as batch space-filling training. Every build receives all
completed training FOM runs:

.. code-block:: python

   rom = builder.build_from_training_dirs(
       offline_data_dir,
       training_data_dirs,
       training_parameters,  # shape (N, parameter_dimension)
       training_qois,        # shape (N, qoi_dimension)
   )

Directory entries, parameter rows, and QoI rows have matching order, including
the initial training samples and every subsequent greedy enrichment. Only selected
training samples are passed, rather than the whole candidate pool. Scalar QoIs
become one-component rows; array QoIs are flattened. The arrays passed to the
builder are independent copies of the workflow's accumulated data. Full-state
snapshots remain available in the training directories.

The returned ROM must still provide ``compute_error_estimate`` as well as the
``QoiModel`` methods. This change does not alter greedy's sample selection,
calibration, or convergence criteria.

Migration from directory-only builders
-------------------------------------

Greedy now always calls the four-argument interface. Existing builders must add
the two array arguments or use an adapter when their construction only needs
directory-based snapshots:

.. code-block:: python

   class DataAwareBuilder:
       def __init__(self, directory_builder):
           self.directory_builder = directory_builder

       def build_from_training_dirs(self, offline_data_dir, training_data_dirs,
                                    training_parameters, training_qois):
           return self.directory_builder.build_from_training_dirs(
               offline_data_dir, training_data_dirs
           )

The existing ``QoiModelWithErrorEstimateBuilder`` remains a directory-only
protocol for callers that need it; it is no longer the builder interface used
by ``run_greedy``.

Tutorial
--------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   notebooks/greedy_training.ipynb





