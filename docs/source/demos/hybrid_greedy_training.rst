Hybrid greedy / space-filling training
=====================================

``run_hybrid_greedy`` combines one error-indicator-selected point per adaptation
batch with geometric maximin coverage. An independent validation error controls
termination. It does not calibrate error indicators against QoI errors.

Algorithm and interfaces
------------------------

Initialization draws ``initial_sample_size`` rows directly from
``ParameterSpace.generate_samples``. The parameter space controls the distribution,
including LHS when configured; the workflow adds no LHS construction. Repeated or
metric-equivalent initial rows are removed in sampler order. This differs from
:doc:`space_filling_training`, whose initialization remains maximin-based.

The workflow executes the initial FOM batch, constructs a ROM, and validates it
**before any adaptation**. Each subsequent round:

#. Draws a fresh ``candidate_pool_size`` pool and excludes points equivalent to
   historical training samples.
#. Evaluates the current ROM's raw error indicator at every remaining candidate
   and selects the largest, breaking ties by input order.
#. Selects up to ``batch_size - 1`` further points by maximin distance, accounting
   for all historical samples, the greedy point, and points already selected in
   this batch. Both selection steps use the same candidate pool.
#. Executes the entire selected FOM batch concurrently, rebuilds using all
   accumulated training data, and validates again.

``batch_size=1`` gives purely greedy adaptation with the same validation stopping
rule. A short distinct candidate pool or remaining FOM budget shortens a batch.
An empty admissible pool stops with ``candidate_pool_exhausted``; it does not
prove the entire parameter domain is exhausted. Accepted FOM training runs are
retained and never repeated.

The FOM implements ``QoiModel``. The builder implements
``QoiModelBuilderWithTrainingData`` and receives aligned training directories,
parameter rows, and QoI rows from **all completed batches**. Its production ROM
must implement ``QoiModelWithErrorEstimate``, including
``compute_error_estimate(run_directory, parameter_dict)``. Missing methods produce
a diagnostic error. Indicators must be finite, nonnegative scalars. They rank
candidates only and need not have the units or normalization of the validation
error. Indicator failures report candidate parameters, index, path, and model stage.

Validation and shared behavior
------------------------------

Validation reuses :ref:`space-filling-validation` unchanged: by default, five
independent fold-specific ROM builds compute the maximum held-out relative L2
QoI error. Training stops when that error is **less than or equal to**
``tolerance``. At least five distinct initial samples are required by default.
``error_functor`` replaces the per-sample error; ``validation_strategy`` accepts
the existing ``SpaceFillingValidationContext`` and owns error aggregation.

``max_iterations`` includes initialization. ``max_fom_samples`` caps training
FOM evaluations. Termination reasons are ``tolerance``, ``max_fom_samples``,
``max_iterations``, and ``candidate_pool_exhausted``, with tolerance taking
precedence. Cross-validation costs, fitting independence, and limitations are
those described in the linked validation documentation.

Candidates may come from irregular domains. NumPy Euclidean distance is the
default; there is no automatic normalization. Supply ``distance_metric(a, b)``
for differing scales, units, non-Euclidean geometry, or nonnumeric parameter
rows. Zero distance identifies equivalent points. See
:ref:`space-filling-distances` for metric, parameter-shape, and seed conventions.

``max_concurrent_fom_runs`` limits active FOM processes independently of
``batch_size``. Each FOM uses an independent directory. All submitted tasks finish
successfully before rebuilding; failed batches retain diagnostic FOM files and
never enter a build. Concurrent FOMs must support multiprocessing spawn. See
:ref:`space-filling-execution` for the complete execution contract.

Candidate ROM calls run serially through ``populate_run_directory``,
``run_model`` (which must return zero), and ``compute_error_estimate``. By default,
independent temporary run directories support filesystem-backed ROMs. They are
removed after selection, including on failure, so models must finish all work
before returning. ``create_rom_run_directories=False`` skips candidate directory
creation for in-memory ROMs; methods still receive nominal absolute paths.
K-fold validation retains its existing directory behavior.

Public API and example
----------------------

.. autofunction:: romtools.workflows.hybrid_greedy.run_hybrid_greedy

.. autoclass:: romtools.workflows.hybrid_greedy.HybridGreedyResult
   :members:

The result holds the final ROM, accumulated training arrays/directories,
validation errors and cumulative sample counts (including initialization), and
``greedy_parameters`` / ``greedy_error_indicators`` (one entry per completed
adaptation round). ``hybrid_greedy_stats.npz`` records the same histories plus FOM
run times; ``hybrid_greedy_status.log`` records progress and failures. Output must
be an absent or empty absolute directory. There is no restart/resume support.

``examples/hybrid_greedy_demo.py`` is a complete surrogate example. It adapts the
existing scalar Gaussian-process builder to expose predictive standard deviation
as the greedy indicator. This indicator is distinct from the measured K-fold
QoI error. Its initial design uses the existing LHS sampler:

.. code-block:: python

   from pathlib import Path
   from examples.hybrid_greedy_demo import ScalarFom, UncertaintyBuilder
   from romtools.workflows import run_hybrid_greedy
   from romtools.workflows.parameter_spaces import UniformParameterSpace
   from romtools.workflows.sampling_methods import LatinHypercubeSampler

   if __name__ == "__main__":
       result = run_hybrid_greedy(
           ScalarFom(), UncertaintyBuilder(),
           UniformParameterSpace(["x"], [0], [1], LatinHypercubeSampler),
           str(Path("hybrid_training").resolve()), tolerance=0.02,
           initial_sample_size=5, batch_size=3, candidate_pool_size=256,
           max_concurrent_fom_runs=3, max_iterations=6, max_fom_samples=20,
           create_rom_run_directories=False,
       )
       print(result.termination_reason, result.validation_errors)

Run the complete example from the checkout with a fresh output directory:

.. code-block:: bash

   python examples/hybrid_greedy_demo.py --work-dir /tmp/hybrid-greedy-demo

For a reusable geometric selector that accepts an explicit candidate pool, use
``select_maximin_samples`` (also exported from ``romtools.workflows``):

.. autofunction:: romtools.workflows.space_filling.select_maximin_samples
