Gaussian processes
==================

``romtools`` provides lightweight Gaussian-process surrogate models for
approximating quantities of interest from parameter samples. Gaussian processes
can also be used as low-fidelity models inside higher-level workflows.

Examples
--------

The example below demonstrates fitting a Gaussian process and evaluating its
posterior mean and uncertainty at new parameter values.

.. code-block:: python

   import numpy as np

   from romtools.rom.qoi_surrogates import (
       GaussianProcessKernel,
       GaussianProcessRegressorLite,
   )

   x_train = np.array([[0.0], [1.0], [2.0]])
   y_train = np.array([0.0, 1.0, 0.0])
   x_query = np.array([[0.5], [1.5]])

   gp = GaussianProcessRegressorLite(
       kernel=GaussianProcessKernel(),
       noise_variance=1e-8,
   )
   gp.fit(x_train, y_train)
   mean, std = gp.predict_mean_and_std(x_query)

POD for vector QoIs
-------------------

For vector-valued QoIs, ``GaussianProcessQoiModel`` first mean-centers the
training QoIs and computes a proper orthogonal decomposition (POD) using an SVD.
The POD basis is truncated according to ``pod_energy_fraction`` and, optionally,
``max_pod_modes``. A separate scalar Gaussian process is then trained for each
retained POD coefficient, and predictions are reconstructed in the original
QoI space.

.. code-block:: python

   from romtools.rom.qoi_surrogates import GaussianProcessQoiModel

   surrogate = GaussianProcessQoiModel(
       parameters=training_parameters,
       qois=training_qois,
       parameter_names=["mu_1", "mu_2", "mu_3"],
       pod_energy_fraction=0.999999,
       max_pod_modes=20,
       normalize_parameters=True,
       normalize_targets=True,
   )

For scalar QoIs, no POD reduction is performed and a single Gaussian process is
trained directly on the QoI.

For examples where Gaussian-process surrogates are created automatically inside
multifidelity inverse workflows, see :doc:`Ensemble Kalman inversion
<ensemble_kalman_inversion>` and :doc:`Variational inference
<variational_inference>`.
