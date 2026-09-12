ROM Tools and Workflows: Examples and Demos
===========================================

.. raw:: html

   <div class="demos-hero">
     <div class="demos-hero-eyebrow">ROM Tools and Workflows</div>
     <div class="demos-hero-title">Build reduced models. Fit surrogates. Run workflows.</div>
     <div class="demos-hero-subtitle">
       Explore the main romtools capabilities through focused concepts and runnable examples.
       Start from reduced-space construction, data-driven surrogates, or end-to-end computational workflows.
     </div>
   </div>

Choose a capability
-------------------

.. grid:: 3
   :gutter: 2
   :class-container: demos-capability-grid

   .. grid-item-card:: Projection-based ROM utilities
      :link: projection_rom_utilities
      :link-type: doc
      :class-card: demos-capability-card demos-capability-card-rom

      Build and manipulate the ingredients of projection-based reduced-order
      models, including reduced vector spaces and basis construction.

      +++
      **Explore ROM utilities →**

   .. grid-item-card:: Surrogates
      :link: surrogates
      :link-type: doc
      :class-card: demos-capability-card demos-capability-card-surrogate

      Build data-driven approximations to model responses and quantities of
      interest. Current examples focus on Gaussian-process surrogates.

      +++
      **Explore surrogates →**

   .. grid-item-card:: Workflows
      :link: workflow_concepts
      :link-type: doc
      :class-card: demos-capability-card demos-capability-card-workflow

      Organize parameterized studies, execute models, construct ROMs, quantify
      uncertainty, and solve inverse problems.

      +++
      **Explore workflows →**

Where should I start?
---------------------

.. grid:: 2
   :gutter: 2
   :class-container: demos-start-grid

   .. grid-item-card:: I want to build a reduced basis
      :link: projection_rom_utilities
      :link-type: doc

      Start with projection-based ROM utilities and learn the reduced-space
      abstractions used throughout romtools.

   .. grid-item-card:: I already have a computational model
      :link: models
      :link-type: doc

      Start with the model interface, then move into parameter spaces, model
      execution, ROM construction, UQ, or inverse workflows.

.. raw:: html

   <div class="demos-feature-strip">
     <div><span class="demos-feature-number">01</span><strong>Learn the concept</strong><br><span>Short overview pages keep the abstraction clear.</span></div>
     <div><span class="demos-feature-number">02</span><strong>Run an example</strong><br><span>Examples live directly beneath the concept they demonstrate.</span></div>
     <div><span class="demos-feature-number">03</span><strong>Scale up</strong><br><span>Reuse the same abstractions in larger workflow studies.</span></div>
   </div>

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
