# Analytic sine VI and MF-VI examples

This directory contains the two VI/MF-VI examples used by the documentation.
Both use the seven-parameter analytic sine-series inverse problem from the
VI/MF-VI paper benchmark and compare single-fidelity VI with automatic-GP
MF-VI.

Both production examples use the same core settings:

- Newton in natural coordinates;
- full Hessian with lagged curvature and `newton_hessian_averaging_factor=0.25`;
- joint entropy estimator with a leave-one-out baseline;
- scalar MF control variate;
- stochastic nonmonotone line search with `initial_step_size=0.25` and `max_step_size=1.0`;
- inference seed 7;
- automatic GP ROM for MF-VI with `max_rom_training_history=4`.

## 1. Diagonal posterior

`diagonal_example.py` uses 31 equispaced interior observations. Discrete sine
orthogonality makes the exact posterior covariance diagonal. The required
`initial_variational_parameter_space` is a `GaussianParameterSpace`, selecting
mean-field VI. The production configuration uses 16 FOM samples per iteration,
64 additional ROM samples for MF-VI, and `newton_regularization=5e-4`.

```bash
python examples/vi_mf_vi_demo/diagonal_example.py
```

## 2. Full-covariance posterior

`full_covariance_example.py` uses the same model, prior, noise level, truth, and
number of observations, but warps the observation locations so they are not
equispaced. The exact posterior is correlated. A
`MultivariateGaussianParameterSpace` initializer starts from the diagonal prior
covariance and selects true full-covariance VI. This example uses the same
lagged-Hessian Newton geometry as the diagonal case, including
`newton_hessian_averaging_factor=0.25`, but doubles the production sampling to
32 FOM samples per iteration and 128 additional ROM samples for MF-VI. Its
Newton regularization is reduced to `1e-4`.

```bash
python examples/vi_mf_vi_demo/full_covariance_example.py
```

Each example writes three figures: exact-KL convergence, the posterior mean,
and the posterior distribution. The full-covariance posterior figure includes
the exact correlation matrix and a two-dimensional marginal for the most
strongly correlated coefficient pair.

Run both examples together with

```bash
python examples/vi_mf_vi_demo/example.py
```

Use `--smoke` for the reduced configuration exercised by documentation CI.

Documentation figures are regenerated with

```bash
python examples/vi_mf_vi_demo/generate_docs_media.py
```
