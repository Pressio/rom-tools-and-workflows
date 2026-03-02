model_builders
=======================

A model builder is a method use to build a (typically data-driven) model in a workflow like greedy.

As an example, we can consider a simple training algorithm where we iteratively
run FOM samples, use them to construct a ROM, evaulate the ROM, and then
continue to add samples until the ROM is of a desired accuracy. In this context,
the model_builder interface will provide context on how to construct the ROM
given the FOM samples. 

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   notebooks/model_builder.ipynb


