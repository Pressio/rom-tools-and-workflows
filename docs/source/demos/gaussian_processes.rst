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

For examples where Gaussian-process surrogates are created automatically inside
multifidelity inverse workflows, see :doc:`Ensemble Kalman inversion
<ensemble_kalman_inversion>` and :doc:`Variational inference
<variational_inference>`.
