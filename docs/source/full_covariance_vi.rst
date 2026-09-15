Full-covariance variational inference
====================================

``romtools`` supports a true full-covariance Gaussian variational family for
VI and MF-VI.  Select it explicitly with
``variational_distribution="full_covariance"``.  Omitting that option keeps
the historical behavior, including the legacy fixed-correlation multivariate
path, for backward compatibility.

Optimizer-coordinate Gaussian
-----------------------------

The full-covariance family is defined in optimizer coordinates,

.. math::

   q(x)=\mathcal N(\mu,\Sigma), \qquad \Sigma=LL^T,

where ``L`` is lower triangular with positive diagonal.  If bounded-parameter
transforms are active, the model parameters are obtained from

.. math::

   \theta=T(x).

The Gaussian score is always formed with ``x`` rather than the transformed
physical parameters.  ``L`` is used for stable sampling, density evaluation,
and restart persistence; the optimization geometry is defined in mean and
covariance coordinates.

Natural score
-------------

For :math:`\delta=x-\mu`, the Gaussian Fisher inverse maps the ordinary score
to the especially simple per-sample natural score

.. math::

   \widetilde s_\mu = \delta,
   \qquad
   \widetilde S_\Sigma = \delta\delta^T-\Sigma.

For an ELBO signal :math:`w_i`, the estimated natural gradient is therefore

.. math::

   \widetilde g_\mu
   = \frac{1}{N}\sum_i w_i\delta_i,

.. math::

   \widetilde G_\Sigma
   = \frac{1}{N}\sum_i w_i
     (\delta_i\delta_i^T-\Sigma).

This avoids assembling or inverting the full Fisher matrix and avoids
reconstructing a covariance gradient from a projected Cholesky gradient.
Symmetric covariance tangents are packed with an isometric ``svec`` mapping,
which scales off-diagonal entries by :math:`\sqrt{2}` so Euclidean vector
inner products match Frobenius matrix inner products.

Entropy estimators
------------------

Both VI entropy strategies are supported.  The ``joint`` strategy includes
``-log q`` in the sampled ELBO signal.  With a suitable baseline it retains
the exact fixed-point property: when the variational distribution equals the
target posterior, the sampled score contribution vanishes.

For ``score_function_entropy_strategy="analytic"``, the Gaussian entropy is
handled exactly.  Its covariance natural gradient is

.. math::

   \widetilde G_{\Sigma,H}=\Sigma,

with no mean contribution.  In MF-VI this exact entropy contribution is added
once after the stochastic multifidelity control-variate estimator; it is not
included in the HF/LF control-variate fit.

SPD-preserving covariance update
--------------------------------

Given a symmetric covariance tangent direction :math:`V_\Sigma`, define

.. math::

   A=L^{-1}V_\Sigma L^{-T}.

The covariance update uses

.. math::

   \Sigma_+ = L\exp(\alpha A)L^T,

followed by a Cholesky factorization of :math:`\Sigma_+`.  This preserves
positive definiteness by construction and has first-order direction
:math:`V_\Sigma`.  A configurable ``max_covariance_log_step`` uniformly limits
:math:`\max_j|\alpha\lambda_j(A)|` before exponentiation instead of clipping
covariance eigenvalues after the update.

Adam semantics
--------------

With ``gradient_method="natural"``, Adam receives the packed Gaussian natural
gradient and applies its existing componentwise moment accumulation before the
covariance block is mapped back to the SPD manifold.  This is a
Fisher-preconditioned/natural-gradient Adam method, not a fully
coordinate-invariant Riemannian Adam algorithm with tangent-space moment
transport.

MF-VI control variate
---------------------

For MF-VI the stochastic quantity passed to the multifidelity control variate
is the complete packed natural-gradient contribution

.. math::

   \Xi_i = (s_i-b_i)
   \begin{bmatrix}
     \delta_i \\
     \operatorname{svec}(\delta_i\delta_i^T-\Sigma)
   \end{bmatrix}.

The existing scalar, componentwise, or matrix-valued MF control-variate
coefficient is applied to this complete vector.  This preserves coupling
between the mean and covariance directions and keeps the baseline control
variate distinct from the multifidelity control variate.

Usage
-----

A diagonal prior can initialize a full-covariance variational family; the
covariance is then free to develop nonzero correlations during optimization.
For example::

   mean, std, samples, qois = romtools.workflows.run_vi(
       model=model,
       prior_parameter_space=prior,
       observations=observations,
       observations_covariance=observation_covariance,
       variational_distribution="full_covariance",
       optimizer_method="adam",
       optimizer_config=romtools.workflows.VIAdamOptimizerConfig(
           gradient_method="natural",
       ),
   )

The complete covariance/Cholesky state is written to the VI history and restart
files.  Full-covariance Newton/Hessian support is intentionally not included in
this implementation; use gradient or Adam optimization for this family.
