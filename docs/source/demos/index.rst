ROM Tools and Workflows Demos
=============================

The ROM tools and workflows Python library comprises a set of algorithms for
constructing and exploiting ROMs. The library is designed internally in terms of
Protocols and abstract base classes that encapsulate all the information needed
to run a given algorithm. The philosophy is that, for any given application, the
user simply needs to create a class that meets the required API of the abstract base class.
Once this class is complete, the user gains access to all of our existing algorithms.



This site provides a suite of tutorials and demos on how to use the rom-tools-and-workflows package. We provide tutorials for

* Basis construction
* Parameter spaces
* Workflows



.. Important::

    rom-tools-and-workflows is only responsible for the offline and workflow aspects of model reduction. It does not deal with the construction of ROMs.



.. toctree::
    :maxdepth: 2

    installation
    documentation

.. toctree::
    :caption: Basic concept tutorials
    :maxdepth: 1

    basis_construction
    vector_space
    parameter_space

.. toctree::
    :caption: Single model workflow tutorials
    :maxdepth: 1

    your_first_problem
    models
    workflows
    holdout_sampling

.. toctree::
    :caption: ROM-FOM workflow tutorials
    :maxdepth: 1

    model_builders
    greedy_training

.. toctree::
    :caption: Inverse workflows
    :maxdepth: 1

    eki_mf_eki_demo

.. toctree::
    :caption: Miscellanea
    :maxdepth: 1

    GitHub Repo <https://github.com/Pressio/rom-tools-and-workflows>
    Open an issue/feature req. <https://github.com/Pressio/rom-tools-and-workflows/issues>
    license
