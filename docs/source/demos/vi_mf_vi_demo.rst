Analytic sine VI and MF-VI example
==================================

This example uses the analytically tractable sine-series inverse problem from
the VI/MF-VI benchmark. The forward model is

.. math::

   u''(x)=f(x;\theta), \qquad u(0)=u(1)=0,

with

.. math::

   f(x;\theta)=\sum_{j=1}^{7}\theta_j\sin(j\pi x),

so the exact solution is

.. math::

   u(x;\theta)
   =-\sum_{j=1}^{7}\frac{\theta_j}{(j\pi)^2}\sin(j\pi x).

The prior is

.. math::

   \theta \sim \mathcal N(0,0.75^2 I),

the synthetic truth is

.. math::

   \theta^\star
   =(1.00,-0.80,0.65,-0.50,0.35,-0.25,0.15)^T,

and the observation noise has standard deviation
:math:`\sigma_\eta=5\times10^{-3}`. Because the forward map is linear and
both the prior and likelihood are Gaussian, the exact posterior is available
in closed form,

.. math::

   C_{\rm post}^{-1}
   = C_0^{-1}+G^T\Gamma_{\rm obs}^{-1}G,

.. math::

   m_{\rm post}
   = C_{\rm post}
     \left(C_0^{-1}m_0+G^T\Gamma_{\rm obs}^{-1}y\right).

This makes the problem useful for demonstrating convergence directly against
the exact posterior.

The example compares single-fidelity VI with MF-VI using the automatic
Gaussian-process ROM. Both methods use Newton updates in natural coordinates,
with ``newton_regularization=5e-4``. The stochastic nonmonotone line search is
capped at ``max_step_size=1.0``. MF-VI uses the automatic GP ROM with
``max_rom_training_history=4``.

The example also demonstrates the revised API in which
``prior_parameter_space`` defines the prior while the required
``initial_variational_parameter_space`` defines the variational family and its
initial moments.

Problem setup
-------------

We use :math:`N=31` equispaced interior observations,

.. math::

   x_i=\frac{i}{N+1}, \qquad i=1,\ldots,N.

For these points the discrete sine basis is orthogonal, so
:math:`G^T\Gamma_{\rm obs}^{-1}G` is diagonal. Since the prior covariance is
also diagonal, the exact posterior covariance is diagonal to numerical
precision.

The variational initializer is therefore a ``GaussianParameterSpace``, which
selects the diagonal/mean-field VI implementation.

Run the example with

.. code-block:: bash

   python examples/vi_mf_vi_demo/diagonal_example.py

A reduced CI configuration is available with ``--smoke``.

Optimization configuration
--------------------------

The core optimizer configuration used for both VI and MF-VI is

.. code-block:: python

   optimizer = VINewtonOptimizerConfig(
       newton_metric="natural",
       newton_regularization=5e-4,
       gradient_norm_tolerance=0.0,
       max_iterations=max_iterations,
   )

   line_search = VIStochasticNonmonotoneLineSearchConfig(
       max_step_size=1.0,
   )

The MF-VI call uses the automatic GP ROM and retains four iterations of ROM
training history:

.. code-block:: python

   workflows.mf_vi_with_auto_rom(
       ...,
       max_rom_training_history=4,
       rom_type="gp",
   )

Convergence
-----------

The convergence plot reports the exact Gaussian
:math:`D_{\rm KL}(q\|p_{\rm post})` for the accepted VI and MF-VI states.

.. figure:: notebooks/analytic_sine_diagonal_convergence.png
   :alt: KL convergence for VI and MF-VI on the equispaced analytic sine problem
   :align: center
   :width: 78%

   VI and MF-VI convergence to the exact diagonal posterior.

Posterior mean
--------------

.. figure:: notebooks/analytic_sine_diagonal_mean.png
   :alt: Posterior mean comparison for the diagonal analytic sine problem
   :align: center
   :width: 82%

   Final VI and MF-VI posterior means compared with the exact posterior mean and
   the synthetic truth.

Posterior
---------

.. figure:: notebooks/analytic_sine_diagonal_posterior.png
   :alt: Posterior marginal densities for the diagonal analytic sine problem
   :align: center
   :width: 96%

   Exact, VI, and MF-VI marginal posterior densities for the seven sine
   coefficients. The vertical dashed line in each panel is the synthetic
   truth.

Implementation
--------------

.. literalinclude:: ../../../examples/vi_mf_vi_demo/diagonal_example.py
   :language: python
   :linenos:

Shared problem definition
-------------------------

The forward model, exact posterior calculation, history processing, and plotting
utilities are shared in

.. literalinclude:: ../../../examples/vi_mf_vi_demo/analytic_sine_support.py
   :language: python
   :linenos:

The separate full-covariance implementation and its variational-family API are
documented in :doc:`../full_covariance_vi`.

Regenerating the figures
------------------------

The documentation build regenerates these figures explicitly before Sphinx
renders the page. The same command can be run locally:

.. code-block:: bash

   python examples/vi_mf_vi_demo/generate_docs_media.py

To validate the example with the reduced CI settings, run

.. code-block:: bash

   python examples/vi_mf_vi_demo/example.py --smoke
