# Analytic sine VI and MF-VI examples

This directory contains the two VI/MF-VI examples used by the documentation.
Both use the seven-parameter analytic sine-series inverse problem from the
VI/MF-VI paper benchmark and compare single-fidelity VI with automatic-GP
MF-VI.

Both production examples use the same core settings:

- Newton in natural coordinates;
- `newton_regularization=1e-4`;
- joint entropy estimator with a leave-one-out baseline;
- 16 FOM samples per iteration;
- stochastic nonmonotone line search with `max_step_size=1.0`;
- automatic GP ROM for MF-VI with `max_rom_training_history=4`.

## 1. Diagonal posterior

`diagonal_example.py` uses 31 equispaced interior observations. Discrete sine
orthogonality makes the exact posterior covariance diagonal. The required
`initial_variational_parameter_space` is a `GaussianParameterSpace`, selecting
mean-field VI.

```bash
python examples/vi_mf_vi_demo/diagonal_example.py
```

## 2. Full-covariance posterior

`full_covariance_example.py` uses the same model, prior, noise level, truth, and
number of observations, but warps the observation locations so they are not
equispaced. The exact posterior is correlated. A
`MultivariateGaussianParameterSpace` initializer starts from the diagonal prior
covariance and selects true full-covariance VI. This example uses the
full-covariance natural-coordinate Newton implementation.

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
