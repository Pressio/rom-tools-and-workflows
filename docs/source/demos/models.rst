Models
======

In ``romtools``, a model is the primary interface to an application. A model can
be an in-process Python function or an external MPI application running on a
cluster. Workflow algorithms operate on this common model abstraction.

Examples
--------

The first examples demonstrate the model interfaces. The CDR and hydrogen-air
flame examples document reusable application models used throughout the
workflow demonstrations.

.. toctree::
   :maxdepth: 1

   notebooks/basic_model.ipynb
   notebooks/external_model.ipynb
   notebooks/external_qoi_model.ipynb
   steady_cdr
   h2_air_flame
