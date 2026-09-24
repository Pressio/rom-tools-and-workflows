Model builders
==============

A model builder defines how a reduced-order model (ROM) is constructed from
available full-order model (FOM) samples.

For example, consider an iterative training workflow such as a greedy algorithm.
The workflow runs FOM simulations, uses the resulting samples to build a ROM,
and evaluates its accuracy. Additional FOM samples are then generated and
incorporated until the ROM reaches the desired accuracy.

In this setting, the `model_builder` interface defines how the ROM is
constructed from the collected FOM samples.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   notebooks/model_builder.ipynb
