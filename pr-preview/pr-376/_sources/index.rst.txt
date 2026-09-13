ROM Tools and Workflows
=======================

The ROM tools and workflows Python library comprises a set of algorithms for
constructing and exploiting ROMs that rely on protocol classes that
encapsulate all the information needed to run a given algorithm.

.. raw:: html

   <div class="docs-home-hero">
     <div class="docs-hero-brand">
       <img src="_static/pressio-logo.png" alt="Pressio logo" />
       <span>Pressio Ecosystem</span>
     </div>
     <div class="docs-hero-kicker">ROM Tools and Workflows</div>
     <div class="docs-hero-title">Advanced ROM workflows in the Pressio ecosystem.</div>
     <div class="docs-hero-subtitle">
       A research-grade toolkit supporting basis construction, hyper-reduction,
       and advanced ROM-enabled workflows.
     </div>
   </div>

Choose your path
----------------

.. grid:: 3
   :gutter: 2
   :class-container: docs-home-grid

   .. grid-item-card:: API Reference
      :link: api
      :link-type: doc

      Explore the full Python API, organized by package.

   .. grid-item-card:: Demos
      :link: demos/index
      :link-type: doc

      Browse the example workflows and tutorials.

   .. grid-item-card:: rom-tools at scale: example applications
      :link: rom_tools_at_scale
      :link-type: doc

      Example applications with large-scale ROM results.

Capabilities
------------

.. grid:: 3
   :gutter: 2
   :class-container: docs-home-grid

   .. grid-item-card:: Trial Space Construction
      :link: generated/romtools.vector_space
      :link-type: doc

      Build reduced trial spaces through vector-space construction, shifting,
      scaling, orthogonalization, and truncation tools.

   .. grid-item-card:: Inverse Workflows
      :link: generated/romtools.workflows.inverse
      :link-type: doc

      Explore ensemble Kalman inversion, variational inference, and
      multifidelity inverse workflows.

   .. grid-item-card:: Sampling Workflows
      :link: generated/romtools.workflows.sampling
      :link-type: doc

      Access parameter sampling workflows and related workflow interfaces for
      model evaluation studies.

   .. grid-item-card:: Remote Execution
      :link: remote_execution
      :link-type: doc

      Run ROM workflows on HPC clusters from your local environment.

.. toctree::
   :hidden:

   api
   formatting
   demos/index
   rom_tools_at_scale
   remote_execution
