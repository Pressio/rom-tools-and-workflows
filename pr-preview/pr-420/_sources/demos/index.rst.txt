ROM Tools and Workflows: Examples and Demos
===========================================

.. raw:: html

   <div class="demos-landing">
     <section class="demos-hero-panel">
       <div class="demos-hero-eyebrow">Examples and demos</div>
       <h2 class="demos-hero-heading">romtools examples organized by capability.</h2>
       <p class="demos-hero-copy">
         The examples are grouped around three areas: projection-based ROM utilities,
         surrogate models, and computational workflows.
       </p>
     </section>

     <section class="demos-area-grid" aria-label="Main romtools areas">
       <a class="demos-area-card demos-area-rom" href="projection_rom_utilities.html">
         <div class="demos-area-topline">
           <span class="demos-area-index">01</span>
           <span class="demos-area-arrow" aria-hidden="true">↗</span>
         </div>
         <h2>Projection-based ROM utilities</h2>
         <p>Utilities for constructing reduced spaces and related offline ingredients for projection-based reduced-order models.</p>
         <div class="demos-area-tags">
           <span>Vector spaces</span>
           <span>Basis construction</span>
           <span>Hyper-reduction</span>
         </div>
       </a>

       <a class="demos-area-card demos-area-surrogates" href="surrogates.html">
         <div class="demos-area-topline">
           <span class="demos-area-index">02</span>
           <span class="demos-area-arrow" aria-hidden="true">↗</span>
         </div>
         <h2>Surrogates</h2>
         <p>Data-driven approximations of model responses and quantities of interest.</p>
         <div class="demos-area-tags">
           <span>Gaussian processes</span>
           <span>Regression</span>
           <span>Workflow integration</span>
         </div>
       </a>

       <a class="demos-area-card demos-area-workflows" href="workflow_concepts.html">
         <div class="demos-area-topline">
           <span class="demos-area-index">03</span>
           <span class="demos-area-arrow" aria-hidden="true">↗</span>
         </div>
         <h2>Workflows</h2>
         <p>Workflow abstractions for model execution, parameter studies, ROM construction, uncertainty quantification, and inverse problems.</p>
         <div class="demos-area-tags">
           <span>Models</span>
           <span>UQ</span>
           <span>Inverse problems</span>
         </div>
       </a>
     </section>

     <section class="demos-start-panel">
       <div class="demos-start-copy">
         <div class="demos-start-eyebrow">Suggested starting points</div>
         <h2>Choose an entry point based on the task of interest.</h2>
       </div>
       <div class="demos-start-links">
         <a href="projection_rom_utilities.html">
           <strong>Reduced-space construction</strong>
           <span>Projection-based ROM utilities →</span>
         </a>
         <a href="models.html">
           <strong>Model-based workflows</strong>
           <span>Model interfaces and execution →</span>
         </a>
       </div>
     </section>
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
