Variational inference
=====================

Variational-inference (VI) workflows approximate a posterior distribution by
optimizing a parameterized search distribution. ``romtools`` supports both
single-fidelity and multifidelity VI workflows.

ELBO scaling
------------

VI and MF-VI accept a positive numerical ``elbo_scaling_factor`` or a named
scaling convention. The default ``"diag_mean"`` (also ``"auto"``) uses the mean
observation-covariance diagonal; ``"diag_trace"`` uses its trace. To normalize by
the initial expected likelihood misfit, pass::

    romtools.workflows.run_vi(
        # model, prior, initialization, observations, etc.
        elbo_scaling_factor="initial_expected_misfit",
    )

The same option is supported by ``run_mf_vi`` and the auto-ROM wrappers, for
both diagonal and full-covariance Gaussian families. It computes

.. math::

    s = \frac{1}{\max(1,\widehat M_0)}, \qquad
    \widehat M_0 = \frac{1}{N_{\mathrm{HF}}}
      \sum_{i=1}^{N_{\mathrm{HF}}}
      \frac12 r(x^{(i)})^\top\Gamma_{\mathrm{eff}}^{-1}r(x^{(i)}),
    \qquad x^{(i)}\sim q_0,

using the initial high-fidelity samples and the same observation precision as
the likelihood (including covariance regularization, when enabled). MF-VI uses
only the initial HF likelihood misfits, not a ROM correction or independent
gain-fitting samples. This reuses the existing initialization batch and adds no
HF evaluations. The floor of one avoids amplifying objectives whose estimated
initial misfit is below one. Nonfinite or negative initial misfits are rejected.

The resolved positive multiplier scales the whole ELBO, including prior,
transformation Jacobian, entropy, gradients, and Hessians. It stays fixed during
optimization and line-search retries. Checkpoints save the numerical multiplier;
restarts with this mode reuse that value instead of estimating a new one.
Existing checkpoints must contain a finite positive ``elbo_scaling_factor``.

This option does not automatically rescale Newton damping, curvature thresholds,
fallback learning rates, or stopping tolerances. Their values remain in the
units of the scaled objective. Existing defaults and numerical scaling modes
are unchanged.

Line-search and Newton control resets
-------------------------------------

Both VI and MF-VI start every new outer iteration with
``min(initial_step_size, max_step_size)``. Mean-field Newton damping starts at
``newton_additive_regularization`` and can grow after each accepted step using
``newton_additive_regularization_growth_factor`` (default ``1.0``)::

    optimizer_config = romtools.workflows.VINewtonOptimizerConfig(
        newton_additive_regularization=0.01,
        newton_additive_regularization_growth_factor=1.05,
        newton_additive_regularization_max=0.1,
    )

After ``k`` accepted steps, the next step's base damping is
``newton_additive_regularization * growth_factor**k``, capped at the configured
maximum when set. Thus the first step uses
``0.01``, the second ``0.0105``, and the third ``0.011025``. The factor must be
finite and at least ``1.0``. An initial damping of zero remains zero.
Scheduled growth works with or without adaptive retry regularization and is
supported only for mean-field Gaussian VI/MF-VI.

Set ``newton_additive_regularization_max`` to cap additive damping, including
scheduled growth and increases during rejected retries. Its default is ``None``
(no absolute cap). A specified maximum must be finite, nonnegative, and at
least the initial ``newton_additive_regularization``. Growth levels off at the
maximum; the initial damping is not silently reduced. This option also works
without scheduled growth. Like damping growth, it is supported only for
mean-field Gaussian VI/MF-VI.

Rejected retries do not advance this schedule. When adaptive regularization is
enabled, retries increase damping up to ``newton_regularization_max_multiplier``
times the current iteration's scheduled base, also subject to the absolute
``newton_additive_regularization_max`` when set. After every acceptance, including
a permitted ELBO decrease, retry inflation is discarded and damping resets to
the next scheduled base; the step size resets to its configured initial value.
The settings ``step_size_growth_factor`` and
``newton_regularization_decrease_factor`` remain accepted configuration fields
but no longer carry outer controls forward after acceptance.

Accepted-step histories, statistics, and logs retain the controls actually used.
Checkpoints store the reset next-iteration values and damping-schedule settings.
Restart validates those settings and reconstructs the scheduled base from the
number of accepted steps, rather than restoring retry inflation. Older
checkpoints without schedule metadata imply a growth factor of ``1.0`` and
no absolute damping cap. Restart requires the same configured maximum.
Adam moments and learning rate, curvature history, ROM state, and objective
reference histories retain their existing behavior.

Adaptive multifidelity estimators
--------------------------------

MF-VI uses three settings, independently configured on ``run_mf_vi`` and
``mf_vi_with_auto_rom``:

* ``rom_retraining_strategy='same_sample'`` trains with the current HF batch
  and accepted HF history when the ROM fails ``rom_tolerance``, then uses the
  replacement immediately. ``'delayed'`` trains only after acceptance and
  activates the replacement in the next outer iteration. The default is
  ``'delayed'``.
* ``gradient_gain_estimation_strategy='same_sample'`` fits a shared gain from the
  current paired HF/LF terms. ``'loo'`` fits each held-out pair's gain from
  all the other pairs. ``'k_fold'`` shares one gain per held-out fold, fitted
  on its complementary paired samples. ``'history'`` fits one gain using accepted
  HF samples retained in the training window and predictions from the active
  ROM at those same points, importance-weighted toward the current distribution.
  ``'independent'`` fits a gain on a fresh, separate paired HF/LF batch.
  The default is ``'same_sample'``.
* ``hessian_gain_estimation_strategy='same_sample'`` always fits Hessian
  gains using the current paired HF/LF Hessian contributions. This is the
  default and the only supported Hessian strategy for now. Gradient gains
  are fitted separately and are never reused for the Hessian.

``baseline_method`` and ``mfmc_control_variate_mode`` remain shared between
gradient and Hessian estimation; the split concerns gain fitting only. The
same-sample Hessian gain can introduce bias even when delayed retraining and a
cross-fitted or historical gradient gain preserve conditional gradient
unbiasedness. Bootstrap continues to use fixed unit gains for both. Checkpoints
record both explicit strategy settings and require matching metadata on restart.

HF and LF paired evaluations always share parameter samples. Additional
LF-only samples are drawn separately. With ``n`` paired and ``m`` LF-only
samples, the correction retains its factor ``m / (n + m)``. LOO applies each
held-out gain to the difference between the LF-only mean and that pair's LF
term, then averages the corrected HF terms. No additional HF evaluations
are needed to estimate LOO or k-fold gains.

Select k-fold with ``gradient_gain_estimation_strategy='k_fold'`` and
``gain_num_folds=5`` (the default gradient fold count). Counts must be integers at least
two and are capped at the paired batch size. Balanced, contiguous folds use
sample indices, never sample values, and consume no random draws. Every held-out
pair receives its fold's gain; contributions are averaged by sample count even
when fold sizes differ. Setting the fold count to the batch size reproduces LOO.
Fewer folds require fewer gain fits but leave fewer training samples per fit.

K-fold preserves nested score-function LOO baselines: each gain-fitting sample
excludes both its outer held-out fold and itself from baseline fitting. Correction
baselines use the entire complementary training set. The paired LF contribution
and extra LF-only mean share that fold-specific LF baseline. Optimal baselines
and all coefficient guards also use only the training set; extra LF-only samples
never fit coefficients or baselines. The coefficient magnitude limit uses the
training sample count in place of ``n-1``. Fewer than two training samples give
zero fitted gain, while existing forced-unit ROM bootstrap behavior is preserved.
The fold count and membership are saved in checkpoints; k-fold restarts require
matching fold configuration.

Like LOO, k-fold gradient gain estimation requires independent MC samples. With a fixed ROM or delayed
retraining after bootstrap it is conditionally unbiased. It does not remove
same-sample ROM retraining bias. Cross-fitted standard errors remain unavailable.

``mfmc_control_variate_mode`` selects ``'scalar'``, ``'componentwise'``, or
``'matrix'`` coefficients; all are supported for every gradient gain strategy and for
same-sample Hessian estimation. LOO excludes the held-out pair from
gradient baseline fitting and coefficient guards as well as covariance calculations.
Fold covariances use stable prefix/suffix moment merges that exclude the
held-out row. Weight offsets are removed before score multiplication, and
correction means use the same stable centering. This avoids cancellation
when large log-weight offsets accompany small variations.

LOO scalar gains fit ``sum(Cov(L, H)) / sum(Var(L))`` on the other pairs;
componentwise gains use the corresponding individual covariances. Ordinary
coefficients are preserved rather than pulled toward unit gain. A fold-only
extreme-gain guard limits their magnitudes to ``1 + sqrt(n - 1) / 2``;
matrix coefficients use the same limit on their spectral norm. This is a
conservative small-fold safeguard, not an optimal-gain guarantee, and the
limit grows with sample count. Degenerate LF variance gives zero gains.
Matrix fits use a fitted scalar coefficient in unidentified covariance
null directions, preserving proportional HF/LF relationships. Numerical
nulls are excluded from matrix solves.

For outer fold ``i``, LOO gain-fitting terms use inner baselines that
exclude both ``i`` and the fitting sample ``j``. With eight paired samples,
each gain uses seven pairs and each inner baseline uses six. The held-out
HF/LF contributions use baselines fitted on the other seven pairs. The LF
baseline from that outer fold is also used for every LF-only contribution
in the same correction. No baseline or gain is fitted on LF-only samples.
Optimal HF and LF baselines are fitted separately on the outer paired fold;
``none`` uses raw terms. These cross-fitting rules apply to gradients, excluding all components of
each held-out sample together. Hessians retain the existing same-sample
score-function baseline procedure.

History gains use all accepted HF samples retained by the ROM training-window
limit, ``max_rom_training_history * fom_sample_size``. Current evaluation
samples and rejected trials are excluded. Each retained batch records the
Gaussian distribution that generated its optimizer-space samples, including
when a batch is partially retained at the window boundary.

For ``N`` retained samples and batch counts ``n_b``, the proposal density is
``q_mix(x) = sum_b (n_b / N) * q_b(x)``. Importance ratios
``q_current(x) / q_mix(x)`` are normalized in log space without clipping.
Current-distribution gradient score kernels and entropy weights are
used at the historical points. Each importance weight enters covariance
moments once: it is not applied to both HF/LF terms before multiplying them.
History LOO and optimal gain-fitting baselines use the corresponding weighted
statistics. Gains fall back to zero when effective sample size is below two,
or when LF variance is degenerate; effective sample size and normalized
weights are persisted with the history fitting data.

Cached HF outputs are reused, so history fitting requires no additional HF
evaluations. ROM predictions are reused across the retained window when the
active model is unchanged; otherwise the active ROM evaluates the historical
points again. Historical data, current target distribution, and importance
weights are determined before each fresh trial with delayed retraining.
Weights and gains may change across retries as the proposed target changes.

The resulting history gain is shared across the current paired corrections.
Current score-function baselines still exclude the sample they center, and
the same fold-specific LF baseline is used in the paired and LF-only terms.
History supports scalar, componentwise, and matrix gradient gains, including
full-covariance workflows and restarts. Restarts preserve
the accepted window, its original sampling distributions, and cached fitting
data; incomplete metadata or a changed training-window limit is rejected.
Historical HF points may also be ROM training points, making their fitted
gain optimistic about out-of-sample ROM accuracy. Reweighting improves the
distribution match but does not make the fitted gain itself unbiased.
Fresh current gradient estimates retain conditional unbiasedness with delayed
retraining because the history gain uses only information available before
those samples are drawn.

When no ROM exists, the bootstrap trains immediately on the initial HF batch
and uses fixed unit gains (an identity matrix for matrix mode). This startup
can be biased because its ROM uses its own evaluation samples. There is no
second bootstrap retraining. With no LF-only samples, the workflow uses
FOM-only estimates and zero gains of the appropriate shape.

After bootstrap, delayed retraining with LOO, k-fold, or history gradient gains
gives conditionally unbiased gradient estimates for fresh independent trials,
assuming valid underlying score estimators. Hessians use same-sample gain fitting
and do not inherit this gradient guarantee.
Same-sample retraining can introduce training bias with any gain strategy;
same-sample gain fitting can introduce bias with either retraining strategy.
LOO, k-fold, and history gradient gains require ``sampling_method='mc'``: points within a Sobol/RQMC batch are
correlated and do not satisfy the held-out independence requirement. RQMC
remains available with same-sample gains.

User-specified fixed gains
~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``use_mfmc_control_variate=False`` and ``mfmc_fixed_gain=0.8`` to use a
constant coefficient for gradient and Hessian corrections. The finite real
scalar defaults to ``1.0``. In componentwise mode it is applied uniformly; in
matrix mode it multiplies the identity. Zero removes the derivative correction;
negative values are supported. With no LF-only samples, estimates remain
FOM-only and report zero gains. This coefficient is persisted and must match
on restart when fixed gains are active, including HF archive sample reuse.

When gain fitting is enabled, ``mfmc_fixed_gain`` is ignored and the existing
fixed-unit bootstrap remains. When fixed gains are requested, the configured
coefficient is also used at bootstrap. No gain-only samples are drawn in fixed
mode. These settings apply to gradient/Hessian corrections; the ELBO estimator
and its sampling-error equation retain their current correction.

Independent gradient gain samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select ``gradient_gain_estimation_strategy='independent'`` to evaluate a new
paired FOM/ROM batch for fitting gradient coefficients on every fitted trial,
including retries. ``gain_sample_size=None`` matches the current trial's HF
evaluation count; a positive integer fixes the gain count independently of
retry sample growth. Counts are decided before sampling. Gain samples come
from the current variational distribution and use the same active ROM as the
MF estimator. HF and LF predictions within each gain pair share parameter
samples; the batch is drawn independently of estimator samples.

Gain samples never enter the current correction, ELBO, ROM training, or
historical training archive. No independent batch is evaluated when the gain
is unused: fixed-unit bootstrap, disabled gain fitting, no LF-only samples,
or auxiliary Hessian-only evaluation. After bootstrap, the extra cost is
``gain_sample_size`` HF and ROM evaluations per fitted trial (or the current
HF evaluation count when omitted).

Scalar, componentwise, and matrix gradient gains are supported for both
covariance families and entropy strategies. Gain-fitting baselines use only
the independent batch: raw terms for ``none``, within-batch LOO baselines for
``loo``, and separate HF/LF optimal baselines for ``optimal``. Covariances use
uniform sample weights and existing numerical safeguards. Fewer than two
pairs or degenerate LF variance give zero gains. Correction baselines retain
the existing held-out procedure, with the same LF baseline in paired and
LF-only terms. Gradient standard-error/SNR diagnostics remain unavailable.
Hessian gains and both strategy defaults remain ``same_sample``.

Independent gain estimation requires MC sampling. With a fixed ROM or delayed
retraining, fresh gain and evaluation samples preserve conditional gradient
unbiasedness before acceptance selection. It does not remove same-sample ROM
training or bootstrap bias. Accepted gain samples and distribution metadata
are saved for restart without reevaluation; future trials always draw a new
gain batch. Restart requires matching gain sample-count and control settings
and complete gain metadata. HF archive reuse retains its existing requirement
for same-sample gradient gains.

MF ELBO sampling uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ELBO used for acceptance retains its current correction,
``mean(H) + r * (mean(L_extra) - mean(L_paired))``, where ``r=m/(n+m)``.
Its estimated sampling variance is
``Var_sample(H-r*L_paired)/n + r**2*Var_sample(L_extra)/m``. Computing the
paired residual captures HF/LF covariance and the overlap between LF means.
The analytic entropy has no sampling variance; the standard error is multiplied
by the absolute ELBO scaling factor. With no LF-only samples, uncertainty is
FOM-only. Sample variances use ``ddof=1``; singleton groups retain the existing
zero-variance fallback, which cannot estimate that group's uncertainty.

Nonmonotone acceptance selects the largest recent accepted ELBO and uses
that entry's own standard error, not the latest state's error. Reference and
candidate variances are added before taking the square root. Both histories
are updated only after acceptance and saved in checkpoints. Stochastic
nonmonotone restarts require complete, aligned uncertainty histories.
Legacy/Adam checkpoints lacking them initialize unused reference histories
from the restored current state. Initial convergence references are preserved.

These are IID-MC plug-in uncertainty estimates. RQMC and importance-weighted
archive uncertainty remain approximate; selection of a historical maximum
and ROM training on evaluation samples are not accounted for as confidence
interval corrections. ELBO values and gradient/Hessian gains are unchanged.

The active ROM stays fixed across delayed-adaptation line-search retries.
Rejected trials do not train replacements or enter training history. Sample
counts are determined before each trial; retry growth may depend on earlier
completed trials, but each retry draws fresh samples. The guarantee applies
before line-search acceptance selection, which can bias the retained estimate.

LOO and history gradient standard errors and signal-to-noise ratios are reported as
unavailable (``NaN``); no jackknife is computed. The separate ELBO uncertainty
calculation used by line search is retained. History stores aggregate gains,
and restarts preserve per-sample gains, completed estimates, strategy
settings, and active/pending ROM training data. Incompatible or incomplete
restarts are rejected. Custom ROM builders must return a distinct replacement
without modifying the active ROM. Enabled HF sample reuse requires both
strategies to be ``'same_sample'``.

Examples
--------

.. toctree::
   :maxdepth: 1

   vi_mf_vi_demo
   h2_air_flame_vi_mf_vi_demo

Implementation organization
---------------------------

The VI implementation lives in ``romtools.workflows.inverse.vi``. Both that
package and ``romtools.workflows.inverse`` export ``run_vi``, ``run_mf_vi``,
``mf_vi_with_auto_rom``, and the VI configuration classes. Imports from the old
flat VI implementation modules must be updated to these package entry points.

Inside the package, ``api`` selects the variational family, ``optimization``
defines optimizers and their defaults, and ``common`` shares estimator utilities.
Diagonal and full-covariance drivers remain separate. ``sample_reuse`` owns
first- and second-order reuse, ``newton`` owns full-covariance Newton state, and
``gains``, ``adaptation``, and ``rom_substeps`` contain multifidelity helpers.
Evaluator selection and Newton curvature state belong to each run; importing or
running VI does not replace numerical driver functions.
