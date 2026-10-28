Greedy Training
=======================

Greedy training is an iterative training process where we iteratively add samples at which we believe the ROM is inaccurate.

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
the initial two samples and every subsequent greedy enrichment. Only selected
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





