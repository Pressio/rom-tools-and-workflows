Analytic sine VI and MF-VI examples
===================================

These examples use the analytically tractable sine-series inverse problem from
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

with synthetic truth

.. math::

   \theta^\star
   =(1.00,-0.80,0.65,-0.50,0.35,-0.25,0.15)^T,

and observation-noise standard deviation
:math:`\sigma_\eta=5\times10^{-3}`. Because the forward map is linear and the
prior and likelihood are Gaussian, the exact posterior is available in closed
form,

.. math::

   C_{\rm post}^{-1}
   = C_0^{-1}+G^T\Gamma_{\rm obs}^{-1}G,

.. math::

   m_{\rm post}
   = C_{\rm post}
     \left(C_0^{-1}m_0+G^T\Gamma_{\rm obs}^{-1}y\right).

This lets both examples measure convergence directly against the exact
posterior.

Shared optimization configuration
---------------------------------

Both examples compare single-fidelity VI with MF-VI using the automatic
Gaussian-process ROM. The production runs use the same optimizer settings:

.. code-block:: python

   optimizer = VINewtonOptimizerConfig(
       newton_metric="natural",
       newton_regularization=1e-4,
       gradient_norm_tolerance=0.0,
       max_iterations=max_iterations,
   )

   line_search = VIStochasticNonmonotoneLineSearchConfig(
       max_step_size=1.0,
   )

The shared VI/MF-VI arguments explicitly use the joint entropy estimator and a
leave-one-out baseline:

.. code-block:: python

   common_arguments = dict(
       ...,
       optimizer_method="newton",
       optimizer_config=optimizer,
       line_search_method="stochastic_nonmonotone",
       line_search_config=line_search,
       baseline_method="loo",
       score_function_entropy_strategy="joint",
   )

The full runs use 16 FOM samples per iteration. MF-VI uses the automatic GP ROM
with four iterations of training history:

.. code-block:: python

   workflows.mf_vi_with_auto_rom(
       ...,
       fom_sample_size=16,
       max_rom_training_history=4,
       rom_type="gp",
   )

``prior_parameter_space`` defines the Bayesian prior, while the required
``initial_variational_parameter_space`` independently selects the variational
family and supplies its initial moments.

Example 1: equispaced observations and a diagonal posterior
------------------------------------------------------------

The first example uses :math:`N=31` equispaced interior observations,

.. math::

   x_i=\frac{i}{N+1}, \qquad i=1,\ldots,N.

For these points the discrete sine basis is orthogonal, so
:math:`G^T\Gamma_{\rm obs}^{-1}G` is diagonal. Since the prior covariance is
also diagonal, the exact posterior covariance is diagonal to numerical
precision.

The variational initializer is a ``GaussianParameterSpace``. This selects the
diagonal/mean-field VI implementation.

Run the example with

.. code-block:: bash

   python examples/vi_mf_vi_demo/diagonal_example.py

Convergence
~~~~~~~~~~~

The convergence plot reports the exact Gaussian
:math:`D_{\rm KL}(q\|p_{\rm post})` for the accepted VI and MF-VI states.

.. figure:: notebooks/analytic_sine_diagonal_convergence.png
   :alt: KL convergence for VI and MF-VI on the equispaced analytic sine problem
   :align: center
   :width: 78%

   VI and MF-VI convergence to the exact diagonal posterior.

Posterior mean
~~~~~~~~~~~~~~

.. figure:: notebooks/analytic_sine_diagonal_mean.png
   :alt: Posterior mean comparison for the diagonal analytic sine problem
   :align: center
   :width: 82%

   Final VI and MF-VI posterior means compared with the exact posterior mean and
   the synthetic truth.

Posterior
~~~~~~~~~

.. figure:: notebooks/analytic_sine_diagonal_posterior.png
   :alt: Posterior marginal densities for the diagonal analytic sine problem
   :align: center
   :width: 96%

   Exact, VI, and MF-VI marginal posterior densities for the seven sine
   coefficients. The vertical dashed line in each panel is the synthetic
   truth.

Implementation
~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/vi_mf_vi_demo/diagonal_example.py
   :language: python
   :linenos:

Example 2: non-equispaced observations and a correlated posterior
------------------------------------------------------------------

The second example keeps the same model, prior, truth, noise level, and number
of observations, but changes only the observation locations. Starting from

.. math::

   t_i=\frac{i}{N+1},

the locations are warped according to

.. math::

   x_i
   =0.02+0.96\,\frac{t_i^3}{t_N^3}.

This breaks discrete sine orthogonality. The exact posterior therefore has
nonzero cross-covariances (the largest absolute posterior correlation is about
0.30 for this setup).

The prior remains the same diagonal ``GaussianParameterSpace``. The variational
initializer is instead a ``MultivariateGaussianParameterSpace`` initialized
with the same diagonal prior covariance. Its *type* selects true
full-covariance VI, allowing off-diagonal covariance entries to develop during
optimization.

This case uses the same natural-coordinate Newton configuration as the diagonal
case. For the full-covariance family, Newton curvature is formed in
:math:`(\mu,\operatorname{svec}(\Sigma))` coordinates and locally whitened with
the exact Gaussian Fisher metric before the regularized Newton solve. The
covariance update is then applied with the SPD-preserving exponential
retraction described in :doc:`../full_covariance_vi`.

Run the example with

.. code-block:: bash

   python examples/vi_mf_vi_demo/full_covariance_example.py

Convergence
~~~~~~~~~~~

.. figure:: notebooks/analytic_sine_correlated_convergence.png
   :alt: KL convergence for full-covariance VI and MF-VI
   :align: center
   :width: 78%

   Exact Gaussian KL convergence for full-covariance VI and MF-VI.

Posterior mean
~~~~~~~~~~~~~~

.. figure:: notebooks/analytic_sine_correlated_mean.png
   :alt: Posterior mean comparison for the non-equispaced analytic sine problem
   :align: center
   :width: 82%

   Final full-covariance VI and MF-VI posterior means compared with the exact
   posterior mean and truth.

Posterior
~~~~~~~~~

.. figure:: notebooks/analytic_sine_correlated_posterior.png
   :alt: Posterior correlation and two-dimensional marginal for the non-equispaced sine problem
   :align: center
   :width: 96%

   The exact posterior correlation matrix and the strongest correlated
   two-dimensional marginal. The ellipse panel compares one- and
   two-standard-deviation contours from the exact posterior, VI, and MF-VI.

Implementation
~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/vi_mf_vi_demo/full_covariance_example.py
   :language: python
   :linenos:

Shared problem definition
-------------------------

The forward model, exact posterior calculation, history processing, and plotting
utilities are shared by both examples:

.. literalinclude:: ../../../examples/vi_mf_vi_demo/analytic_sine_support.py
   :language: python
   :linenos:

Regenerating the figures
------------------------

Run both production examples and regenerate all six figures with

.. code-block:: bash

   python examples/vi_mf_vi_demo/generate_docs_media.py

A reduced CI configuration exercises both examples with

.. code-block:: bash

   python examples/vi_mf_vi_demo/example.py --smoke
