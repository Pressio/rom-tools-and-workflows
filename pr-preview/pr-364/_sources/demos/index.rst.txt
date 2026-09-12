ROM Tools and Workflows: Examples and Demos
===========================================

``romtools`` provides reusable building blocks for projection-based reduced-order
models, surrogate models, and model-based computational workflows. The examples
are organized around those three concepts so that each capability can grow
without turning this page into a flat catalog of notebooks.

Projection-based ROM utilities
------------------------------

Construct and manipulate the ingredients of projection-based reduced-order
models, including reduced vector spaces and basis construction.

:doc:`Explore projection-based ROM utilities <projection_rom_utilities>`

Surrogates
----------

Build data-driven approximations to model responses and quantities of interest.
The current surrogate capability is based on Gaussian processes.

:doc:`Explore surrogates <surrogates>`

Workflows
---------

Organize and execute parameterized computational studies. Workflow concepts
include models, parameter spaces, model execution, ROM construction and
adaptation, uncertainty quantification, and inverse problems.

:doc:`Explore workflows <workflow_concepts>`

Getting started
---------------

If you are new to ``romtools``, start with projection-based ROM utilities to
learn the reduced-space abstractions, then continue to the workflow examples.
If you already have an application that you want to run through ``romtools``,
start with :doc:`Models <models>` and :doc:`Model execution <workflows>`.

.. toctree::
   :hidden:
   :maxdepth: 3

   projection_rom_utilities
   surrogates
   workflow_concepts

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   installation
   documentation

.. toctree::
   :caption: Project
   :maxdepth: 1

   GitHub Repo <https://github.com/Pressio/rom-tools-and-workflows>
   Open an issue/feature req. <https://github.com/Pressio/rom-tools-and-workflows/issues>
   license
