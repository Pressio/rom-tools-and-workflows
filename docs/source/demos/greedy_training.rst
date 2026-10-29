Greedy Training
=======================

Greedy training is an iterative training process where we iteratively add samples at which we believe the ROM is inaccurate.

Initial training and iteration limits
------------------------------------

``initial_sample_size`` defaults to 2. Greedy runs those first samples from the
initial candidate pool before constructing the initial ROM. This is separate
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
integer. ``initial_sample_size`` cannot exceed ``testing_sample_size``. If
enrichment is enabled, the candidate pool must be strictly larger than the
initial training set so there is a point to test. Convergence checks retain
their previous calibration behavior: the first check follows one enrichment.
The iteration cap stops after the final allowed rebuild; a cap exit does not
claim convergence or perform another candidate evaluation.

FOM directories are populated once, immediately before their simulation, and
unselected FOM candidates are not prepared. Each ROM evaluation directory is
also populated once in its iteration. ``greedy_stats.npz`` is written even for
an initial-only run and records ``initial_sample_size``, ``max_iterations``,
completed ``iterations``, ``converged``, and ``termination_reason`` (``tolerance``
or ``max_iterations``). Error histories describe completed enrichments.

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





