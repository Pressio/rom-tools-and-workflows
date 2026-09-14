H2-air flame VI benchmark
=========================

This example applies the romtools variational-inference workflows to the
existing :doc:`H2-air flame model <h2_air_flame>`. The benchmark uses
``H2AirFlameQoiModel`` directly; it does not duplicate the flame solver or
observation operator.

Inverse problem
---------------

The four inferred parameters are diffusivity ``kappa``, scaled activation
energy, and the two advection components ``beta_x`` and ``beta_y``. The
production parameter domain is

.. code-block:: text

   kappa                     in [0.5, 4.0]
   scaled_activation_energy  in [4.0, 12.0]
   beta_x                    in [20.0, 60.0]
   beta_y                    in [1.0, 20.0]

with synthetic truth

.. math::

   [\kappa, E/1000, \beta_x, \beta_y]
   = [2, 8, 40, 7].

The initial variational distribution is deliberately not centered on the
truth. Observations are the existing regularly strided nondimensional
temperature sensors from ``H2AirFlameQoiModel``.

A single noisy observation realization is generated once and reused for every
algorithmic method and random seed. The independent Gaussian noise standard
deviation is chosen from a 5% global RMS noise-to-signal ratio,

.. math::

   \sigma_{\mathrm{obs}}
   = 0.05\,\frac{\|y_{\mathrm{clean}}\|_2}{\sqrt{N_{\mathrm{obs}}}}.

Method comparison
-----------------

The primary YAML-driven ablation compares

#. ``BBVI (Adam)``;
#. ``ARBIS``: Adam with importance-sampling/sample reuse;
#. ``ARBIS + MF``: Adam sample reuse with automatic-GP multifidelity variance reduction;
#. ``BBVI (Newton)``;
#. ``MF-BBVI (Newton, analytic)``; and
#. ``MF-BBVI (Newton, joint)``.

Every applicable score-function estimator uses the leave-one-out baseline. The
two MF-Newton cases use identical settings except for the score-function entropy
strategy. Multifidelity cases use the romtools automatic Gaussian-process QoI
model with normalized inputs/targets and POD reduction for the vector QoI.

The production flame discretization is ``64 x 32`` with ``dt=1e-3`` and
``t_end=6e-2``, matching the existing H2-air flame EKI benchmark. A much smaller
real-flame configuration is used by CI.

Cost accounting
---------------

The primary x-axis is cumulative H2-air FOM evaluations. The postprocessor
counts the flame-specific ``solution.npz`` artifacts generated under each VI
iteration. This includes ordinary BBVI samples, sample-reuse refreshes,
line-search evaluations, and FOM evaluations used to train or rebuild the
automatic GP. GP evaluations are not counted as FOM work. Wall-clock time is
also recorded.

The configured FOM budget is a matched-work analysis cap: romtools completes a
VI iteration atomically, and the benchmark truncates comparisons to the last
completed iteration inside the common budget rather than interrupting a model
batch or line search.

Run the benchmark
-----------------

Run the production method ablation with

.. code-block:: bash

   python examples/h2_air_flame_vi_benchmark/benchmark.py

Run the sample-size sweep with

.. code-block:: bash

   python examples/h2_air_flame_vi_benchmark/benchmark.py \
     --config examples/h2_air_flame_vi_benchmark/configs/sample_size_sweep.yaml

Run the reduced CI configuration with

.. code-block:: bash

   python examples/h2_air_flame_vi_benchmark/benchmark.py --smoke

The reduced sweep machinery can be checked with

.. code-block:: bash

   python examples/h2_air_flame_vi_benchmark/benchmark.py \
     --smoke --mode sweep --methods bbvi_adam arbis_mf

Configuration and outputs
-------------------------

Scientific and algorithmic settings live in ``configs/method_ablation.yaml``
and ``configs/sample_size_sweep.yaml``. The production sweep uses FOM sample
sizes ``[4, 8, 16, 32]`` and MF ROM-extra sample sizes ``[64, 128]``. The
resolved YAML is copied into the result directory for reproducibility.

The ablation writes machine-readable histories and plots of parameter error,
ELBO, observation-space error, and per-parameter posterior means against
cumulative FOM evaluations. Repeated-seed results are summarized with medians
and interquartile bands, and the two MF-Newton entropy strategies receive a
direct matched-work comparison. The sample-size sweep additionally plots final
matched-work parameter error, ELBO, and observation error against the FOM sample
size. Final posterior moments, wall time, observation metadata, and sample-reuse
diagnostics are retained in JSON.

Implementation
--------------

.. literalinclude:: ../../../examples/h2_air_flame_vi_benchmark/benchmark.py
   :language: python
   :linenos:
