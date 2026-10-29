Batch space-filling ROM training
===============================

``run_space_filling`` constructs a ROM through sequential batches of geometrically
selected FOM simulations. Unlike greedy training, sample selection uses parameter
coverage rather than an error indicator. Validation controls when training stops.

Each round selects its complete batch, executes the FOMs concurrently, builds a
ROM from all accumulated successful training runs, and validates it. The initial
batch defaults to 10 samples; subsequent batches default to 5. ``batch_size``
and ``max_concurrent_fom_runs`` are independent: a batch of 10 can execute with
at most 3 active FOM tasks.

Public API
----------

.. autofunction:: romtools.workflows.space_filling.run_space_filling

.. autofunction:: romtools.workflows.space_filling.select_space_filling_samples

.. autoclass:: romtools.workflows.space_filling.SpaceFillingResult
   :members:

.. _space-filling-distances:

Parameter spaces and distances
------------------------------

Candidates come exclusively from ``ParameterSpace.generate_samples``. The
workflow does not inspect bounds, project into a box, or assume a uniform
sampling distribution. Constrained domains can implement their own sampler;
unbounded distributions are also accepted. Coverage is always relative to the
finite candidate pool: rare regions not sampled by the space cannot be selected.

Maximin selection maintains each candidate's nearest distance to existing and
already-selected training points. The first point is seeded randomly when there
is no training data; subsequent points maximize that nearest distance. Euclidean
distances use vectorized NumPy operations without a candidate-by-training matrix.
A custom metric uses a slower general path.

There is **no automatic parameter normalization**. For different units/scales,
provide a suitable metric, for example:

.. code-block:: python

   scales = np.array([10.0, 0.01])
   distance_metric = lambda a, b: np.linalg.norm((a - b) / scales)

The default metric requires finite numeric parameters. Custom metrics can handle
string/object rows, including heterogeneous spaces. Each sample row must have
one column per parameter name for FOM dictionary construction. Distances must be
finite, nonnegative scalars. Zero distance identifies duplicate/equivalent points.

The sampling utility returns **up to** the requested batch size. Short or empty
results mean the candidate pool lacks enough distinct points; this does not prove
that the whole domain is exhausted. Constant and zero-dimensional spaces contain
at most one distinct point under the default metric. The default five-fold
validation needs at least five distinct initial points; choose a different
validation strategy for smaller datasets.

``random_seed`` controls independent sampling, validation, build, and FOM NumPy
seed streams. Sampling does not change the caller's NumPy RNG state. Legacy
``generate_samples(n)`` implementations using NumPy's global RNG are supported;
custom spaces owning independent unseeded RNGs must supply their own reproducible
behavior. Reproducibility of external solvers remains the model's responsibility.

.. _space-filling-execution:

FOM execution contract
----------------------

For each sample, the workflow creates an independent absolute run directory and
calls ``populate_run_directory``, ``run_model``, then ``compute_qoi`` in the same
worker task. ``run_model`` must return 0 for success. Nonzero return codes and
exceptions fail the batch. QoIs must be nonempty finite real numeric values;
scalars are flattened to one-element vectors, with consistent dimensions across
all samples and rounds.

Execution uses ``ProcessPoolExecutor`` with ``spawn``, isolating working-directory
changes and Python object state. Concurrent FOM objects must be serializable;
define model classes at module scope and guard executable scripts with
``if __name__ == "__main__":``. With concurrency one, execution is direct and
does not require serialization. Model calls restore the previous working
directory, including on failure.

Per-run state can live in the worker's model or its run directory, but model
mutations are not returned to the parent process. Do not rely on shared mutable
state across samples or write to common external filenames. Models launching MPI
or threaded solvers must account for their own resource use: the concurrency
limit counts FOM tasks, not MPI ranks or BLAS threads.

Builders and training data
--------------------------

Use ``QoiModelBuilderWithTrainingData``. Each build receives:

.. code-block:: python

   rom = builder.build_from_training_dirs(
       offline_data_dir,
       training_data_dirs,
       training_parameters,  # (N, parameter_dimension)
       training_qois,        # (N, qoi_dimension)
   )

Directory, parameter, and QoI entries have matching row order. Directories retain
application-specific full-state snapshots and other files; QoIs supplement those
artifacts and do not replace the snapshots a projection-based builder needs.
Builders must read only their supplied training directories, fit independent
models from only their supplied data, and write artifacts under their supplied
offline directory. Two-argument directory-only builders need a user adapter.

Builders, validation strategies, ROM evaluations, and error functors execute
sequentially in the parent process; they need not be serializable.

.. _space-filling-validation:

Validation and error functors
----------------------------

The default strategy is five-fold cross-validation. Every fold builds a new ROM
from the complementary training subset and predicts the held-out samples using
their cached FOM reference QoIs. All fitted preprocessing, including POD bases,
normalization, truncation, and hyperparameters, must use only that fold's training
subset. The returned production ROM is built from the complete dataset.

The default per-sample error is relative L2:

.. math::

   e_i = \frac{\lVert q_{\mathrm{ROM},i} - q_{\mathrm{FOM},i}\rVert_2}
                   {\max(\lVert q_{\mathrm{FOM},i}\rVert_2, 10^{-12})}.

The denominator floor avoids division by zero; errors near zero reference norms
can consequently be large. The maximum held-out sample error is compared with
``tolerance``, and convergence occurs when error is less than or equal to it.

To change the per-sample measure, supply ``error_functor(qoi_rom, qoi_fom)``:

.. code-block:: python

   result = run_space_filling(
       ...,
       error_functor=lambda q_rom, q_fom: np.linalg.norm(q_rom - q_fom, ord=np.inf),
   )

Arguments are matching flattened QoI vectors. The functor returns the complete,
finite nonnegative scalar error; **no further normalization** is applied.
Tolerance follows that error's units/meaning. Functor failures report fold and
sample context.

.. autoclass:: romtools.workflows.space_filling.KFoldValidation
   :members:

For example, ``validation_strategy=KFoldValidation(num_folds=3)`` uses three
folds. Each round costs one production build, one build per fold, and one
held-out ROM evaluation per training sample, with no extra FOM evaluations.

For a custom validation procedure, pass a callable returning a finite nonnegative
scalar. It receives the production ROM, builder, read-only training arrays,
directories, parameter names, validation seed/directory, iteration index, and
resolved error functor:

.. autoclass:: romtools.workflows.space_filling.SpaceFillingValidationContext
   :members:

A callback can close over independently prepared holdout samples and reference
QoIs and use ``context.error_functor`` for comparisons. It owns aggregation and
any extra FOM costs; the workflow's FOM budget covers training runs only.

Cross-validation on a geometric design is not a distribution-weighted estimate
or a worst-case error certificate. Repeated stopping checks can introduce
selection bias. Independent holdout references offer a separate accuracy check
at additional FOM cost. QoI convergence does not establish full-state accuracy.

Termination, failures, and output
---------------------------------

``max_iterations`` defaults to 20 and includes the initial training round.
``max_fom_samples`` optionally caps training FOM evaluations; the final batch is
shortened to respect the cap. Results distinguish ``tolerance``,
``max_iterations``, ``max_fom_samples``, and ``candidate_pool_exhausted``.
Tolerance takes precedence when limits are reached in a converged round;
otherwise the sample limit takes precedence over the iteration limit.

The absolute output directory must be absent or empty. No overwrite, restart,
remote dispatcher, or automatic retry is supported. Layout:

.. code-block:: text

   fom/run_<global_sample_id>/
   iteration_<k>/offline_data/
   iteration_<k>/validation/fold_<f>/offline_data/
   iteration_<k>/validation/fold_<f>/run_<sample_id>/
   space_filling_status.log
   space_filling_stats.npz

Only the parent writes campaign logs/history. Each successful sample has its own
``passed.txt`` marker. All submitted FOM tasks finish before a batch failure is
reported, and a failed batch never enters a ROM build. Successful and failed run
files remain for diagnosis; workers are shut down. Builder or validation errors
also propagate, preserving the previous completed round's statistics and
artifacts. Statistics are diagnostic output, not a restart interface. Object
parameter arrays stored in NPZ require NumPy's pickle-enabled loading; load such
files only when trusted.

Complete example
----------------

``examples/space_filling_demo.py`` demonstrates an irregular disk domain,
file-backed FOM tasks, the existing Gaussian-process data-array builder, and
maximum relative L2 K-fold validation. From the checkout:

.. code-block:: bash

   python examples/space_filling_demo.py --work-dir /tmp/space-filling-demo

The example prints sample counts, validation errors, and the termination reason,
and saves its full training history. Use a fresh output directory for each run.
