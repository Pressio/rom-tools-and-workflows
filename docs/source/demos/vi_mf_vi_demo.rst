Analytic sine VI and MF-VI examples
===================================

These examples use the analytically tractable sine-series inverse problem from
the VI/MF-VI benchmark.  The forward model is

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
:math:`\sigma_\eta=5\times10^{-3}`.  Because the forward map is linear and
both the prior and likelihood are Gaussian, the exact posterior is available
in closed form,

.. math::

   C_{\rm post}^{-1}
   = C_0^{-1}+G^T\Gamma_{\rm obs}^{-1}G,

.. math::

   m_{\rm post}
   = C_{\rm post}
     \left(C_0^{-1}m_0+G^T\Gamma_{\rm obs}^{-1}y\right).

This gives a useful documentation problem: convergence can be measured against
the exact posterior rather than against a reference Monte Carlo calculation.

Both examples compare single-fidelity VI with MF-VI using the automatic
Gaussian-process ROM.  They also demonstrate the revised API in which
``prior_parameter_space`` defines the prior while the required
``initial_variational_parameter_space`` defines the variational family and its
initial moments.

Example 1: equispaced observations and a diagonal posterior
------------------------------------------------------------

The first example uses :math:`N=31` equispaced interior observations,

.. math::

   x_i=\frac{i}{N+1}, \qquad i=1,\ldots,N.

For these points the discrete sine basis is orthogonal, so
:math:`G^T\Gamma_{\rm obs}^{-1}G` is diagonal.  Since the prior covariance is
also diagonal, the exact posterior covariance is diagonal to numerical
precision.

The variational initializer is therefore a ``GaussianParameterSpace``.  This
selects the diagonal/mean-field VI implementation.

Run the example with

.. code-block:: bash

   python examples/vi_mf_vi_demo/diagonal_example.py

A reduced CI configuration is available with ``--smoke``.

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
   coefficients.  The vertical dashed line in each panel is the synthetic
   truth.

Implementation
~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/vi_mf_vi_demo/diagonal_example.py
   :language: python
   :linenos:

Example 2: non-equispaced observations and a full covariance
-------------------------------------------------------------

The second example keeps the same seven coefficients, prior, noise level, and
number of observations but changes only the observation locations.  Starting
from

.. math::

   t_i=\frac{i}{N+1},

the locations are warped according to

.. math::

   x_i
   =0.02
    +0.96\,\frac{t_i^3}{t_N^3}.

The observations now cluster toward the left side of the interval.  This
breaks discrete sine orthogonality, so the likelihood precision and exact
posterior covariance contain off-diagonal terms.

The prior remains diagonal.  The variational initializer is instead a
``MultivariateGaussianParameterSpace`` whose initial covariance is the same
diagonal prior covariance.  The *type* of the initializer selects true
full-covariance VI, so off-diagonal covariance terms are free to develop during
optimization even though the initial covariance itself is diagonal.

Run the example with

.. code-block:: bash

   python examples/vi_mf_vi_demo/full_covariance_example.py

The full-covariance implementation currently uses natural-gradient Adam; full
covariance Newton/Hessian optimization is intentionally outside the scope of
this implementation.

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
   two-dimensional marginal.  The ellipse panel compares one- and
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

The documentation build regenerates these figures explicitly before Sphinx
renders the page.  The same command can be run locally:

.. code-block:: bash

   python examples/vi_mf_vi_demo/generate_docs_media.py

To validate both examples with the reduced CI settings, run

.. code-block:: bash

   python examples/vi_mf_vi_demo/example.py --smoke
