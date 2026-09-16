Full-covariance variational inference
=====================================

``romtools`` distinguishes the Bayesian prior from the variational family.
Both VI and MF-VI require an ``initial_variational_parameter_space``. Its type
selects the variational family and its moments provide the initial variational
state:

* ``GaussianParameterSpace`` selects diagonal/mean-field VI;
* ``MultivariateGaussianParameterSpace`` selects true full-covariance VI.

The ``prior_parameter_space`` defines only the prior. It may independently be
diagonal or multivariate, provided it uses the same parameter names in the same
order as the variational initializer. There is no separate public
``variational_distribution`` argument.

A multivariate initializer with a diagonal covariance still selects the
full-covariance family, so correlations are free to develop during
optimization. The historical fixed-correlation interpretation of a
multivariate variational initializer is not part of the public API.

Optimizer-coordinate Gaussian
-----------------------------

The full-covariance family is defined in optimizer coordinates,

.. math::

   q(x)=\mathcal N(\mu,\Sigma), \qquad \Sigma=LL^T,

where ``L`` is lower triangular with positive diagonal. If bounded-parameter
transforms are active, the model parameters are obtained from

.. math::

   \theta=T(x).

The Gaussian score is formed with ``x`` rather than the transformed physical
parameters. ``L`` is used for stable sampling, density evaluation, and restart
persistence; the optimization geometry is defined in mean and covariance
coordinates.

Natural score
-------------

For :math:`\delta=x-\mu`, the Gaussian Fisher inverse maps the ordinary score
to the per-sample natural score

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

Symmetric covariance tangents are packed with an isometric ``svec`` mapping,
which scales off-diagonal entries by :math:`\sqrt{2}` so Euclidean vector
inner products match Frobenius matrix inner products.

Entropy estimators
------------------

Both VI entropy strategies are supported. The ``joint`` strategy includes
``-log q`` in the sampled ELBO signal. With a suitable baseline it retains the
fixed-point property: when the variational distribution equals the target
posterior, the sampled gradient contribution vanishes.

For ``score_function_entropy_strategy="analytic"``, the Gaussian entropy is
handled exactly. Its covariance natural gradient is

.. math::

   \widetilde G_{\Sigma,H}=\Sigma,

with no mean contribution. In MF-VI this exact entropy contribution is added
once after the stochastic multifidelity control-variate estimator; it is not
included in the HF/LF control-variate fit.

Full-covariance natural Newton
------------------------------

Full-covariance Newton uses the ordinary score-function Hessian in
:math:`(\mu,\operatorname{svec}(\Sigma))` coordinates. Writing
:math:`P=\Sigma^{-1}` and :math:`a=P(x-\mu)`, the ordinary Gaussian scores are

.. math::

   s_\mu=a,
   \qquad
   S_\Sigma=\frac{1}{2}(aa^T-P).

For the log-joint contribution, the curvature estimator uses the same
second-order score identity as diagonal VI,

.. math::

   H_{\log p}
   =\mathbb E_q\left[
      \log p(y,x)
      \left(ss^T+\nabla^2\log q\right)
   \right].

When the joint entropy estimator is selected, ``-log q`` remains in the
*gradient* estimator. Newton curvature follows the existing romtools split
estimator: the stochastic Hessian uses the log-joint contribution above and
the Gaussian entropy Hessian is added analytically. In covariance coordinates,
for a symmetric perturbation :math:`E`,

.. math::

   D\left(\nabla_\Sigma H(q)\right)[E]
   =-\frac{1}{2}P E P.

Natural coordinates are introduced by locally whitening the exact Gaussian
Fisher metric. If :math:`A` is a symmetric tangent coordinate, the first-order
whitening map is

.. math::

   \delta\mu = L a,
   \qquad
   \delta\Sigma = \sqrt{2}\,L A L^T.

The covariance step is applied through the exponential local map

.. math::

   \Sigma(A)=L\exp(\sqrt{2}A)L^T.

Because this map is nonlinear, transforming the ordinary Hessian as
:math:`S^T H S` alone is not the Hessian of the objective in the coordinates
used by the actual covariance update. Let :math:`B_a` denote the
Frobenius-orthonormal symmetric basis associated with ``svec`` and let
:math:`G_\Sigma=\nabla_\Sigma\mathcal L`. At the local origin,

.. math::

   \frac{\partial^2\Sigma}
        {\partial z_a\partial z_b}
   =L(B_aB_b+B_bB_a)L^T.

The covariance-covariance block therefore receives the pullback correction

.. math::

   C_{ab}
   =\left\langle
      G_\Sigma,
      L(B_aB_b+B_bB_a)L^T
    \right\rangle_F.

With :math:`S` denoting the first-order Fisher-whitening map, the local Newton
system uses

.. math::

   g_N=S^Tg,
   \qquad
   H_N=S^T H S+
   \begin{bmatrix}
      0 & 0\\
      0 & C
   \end{bmatrix}.

This is the exact second-order pullback of the ELBO at the origin of the local
exponential covariance coordinates. The correction is important even for a
Gaussian target: the expected log joint is linear in :math:`\Sigma`, but it
has nonzero second derivative after composition with the exponential
covariance map. The resulting direction is then mapped back through the
first-order tangent map and applied with the SPD exponential retraction.

Full-covariance natural Newton supports both ``same_sample`` and ``lagged``
curvature strategies. The public Newton default is ``lagged`` with
``newton_hessian_averaging_factor=0.25``. For lagged curvature, the raw
ordinary-coordinate Hessian is exponentially averaged across accepted states,

.. math::

   \bar H_k = \beta \bar H_{k-1} + (1-\beta) H_k,
   \qquad \beta=0.25,

before Fisher whitening and the exponential-map pullback correction are
applied. Rejected line-search candidates do not update :math:`\bar H_k`.
Both diagonal and full projected Hessian solves can be selected through the
existing ``newton_hessian_type`` option. The standard-coordinate
full-covariance Newton route remains unsupported.

MF-VI Newton curvature
----------------------

For full-covariance MF-VI, the high- and low-fidelity second-order
score-function contributions are packed as complete matrices and passed
through the existing multifidelity control-variate estimator. The analytic
entropy Hessian is added once after the multifidelity correction. After this
ordinary-coordinate curvature is assembled, the same exponential-map
pullback correction described above is applied using the multifidelity
covariance gradient. This mirrors the treatment of the full packed gradient
and preserves the HF/LF coupling of mean and covariance directions.

SPD-preserving covariance update
--------------------------------

Given the symmetric covariance block of the Newton or gradient direction,
:math:`V_\Sigma`, define

.. math::

   A=L^{-1}V_\Sigma L^{-T}.

The covariance update uses

.. math::

   \Sigma_+ = L\exp(\alpha A)L^T,

followed by a Cholesky factorization of :math:`\Sigma_+`. This preserves
positive definiteness by construction and has first-order direction
:math:`V_\Sigma`. A configurable ``max_covariance_log_step`` uniformly limits
:math:`\max_j|\alpha\lambda_j(A)|` before exponentiation instead of clipping
covariance eigenvalues after the update.

Adam semantics
--------------

With ``gradient_method="natural"``, Adam receives the packed Gaussian natural
gradient and applies its existing componentwise moment accumulation before the
covariance block is mapped back to the SPD manifold. This is a
Fisher-preconditioned/natural-gradient Adam method, not a fully
coordinate-invariant Riemannian Adam algorithm with tangent-space moment
transport.

MF-VI control variate
---------------------

For MF-VI the stochastic quantity passed to the multifidelity control variate
for the natural gradient is the complete packed contribution

.. math::

   \Xi_i = (s_i-b_i)
   \begin{bmatrix}
     \delta_i \\
     \operatorname{svec}(\delta_i\delta_i^T-\Sigma)
   \end{bmatrix}.

The existing scalar, componentwise, or matrix-valued MF control-variate
coefficient is applied to this complete vector. This preserves coupling
between the mean and covariance directions and keeps the baseline control
variate distinct from the multifidelity control variate.

Usage
-----

A diagonal prior can be paired with a full-covariance variational family by
supplying a multivariate initializer. Natural-coordinate Newton with full,
lagged curvature is now the public Newton default::

   q0 = MultivariateGaussianParameterSpace(
       parameter_names=parameter_names,
       means=initial_mean,
       covariance=initial_covariance,
       sampler=MonteCarloSampler,
   )

   optimizer = romtools.workflows.VINewtonOptimizerConfig()

   mean, std, samples, qois = romtools.workflows.run_vi(
       model=model,
       prior_parameter_space=prior,
       initial_variational_parameter_space=q0,
       observations=observations,
       observations_covariance=observation_covariance,
       optimizer_method="newton",
       optimizer_config=optimizer,
       score_function_entropy_strategy="joint",
   )

Conversely, a ``GaussianParameterSpace`` initializer selects diagonal VI even
when the prior is multivariate.

The complete covariance/Cholesky state is written to VI history and restart
files.
