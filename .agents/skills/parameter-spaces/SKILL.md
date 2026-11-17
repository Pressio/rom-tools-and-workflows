---
name: parameter-spaces
description: Use when selecting distributions, bounds, sampling algorithms, or defining custom parameter-space classes.
---
# Parameter spaces

## When to use
Use when selecting distributions, bounds, sampling algorithms, or defining custom parameter-space classes.

## Read first
`romtools/workflows/parameter_spaces.py`; `romtools/workflows/parameters.py`; `romtools/workflows/sampling_methods.py`; `tests/romtools/workflows/sampling/test_sampling.py`.

## Procedure
1. Choose `UniformParameterSpace` for bounded independent uniforms; inspect its constructor and required `sampler` argument.
2. `get_names()` defines the parameter dictionary order; `generate_samples(n, seed)` returns `(n,d)`; do not transpose.
3. `MonteCarloSampler` and randomized quasi-Monte Carlo samplers differ; do not replace one silently.
4. Validate bounds and dimensionality; distinguish parameter priors from inverse posterior approximations and EKI initial ensembles.
5. Reproduce seeds with the workflow's actual sampling call rather than assuming a particular RNG implementation.

## Verification
Test shape, ordering, ranges, deterministic repeats, and edge cases. For `run_sampling`, inspect how its `random_seed` interacts with NumPy and the selected sampler.

## Reviewable execution trace
Follow the common protocol in `.agents/TRACE_PROTOCOL.md`. For this task specifically: Record parameter names/order, distributions/bounds, sampler, seed, generated sample shape and a reproducibility check; avoid storing sensitive input samples.

## Handoff
State which implementation and tests were inspected, what changed, commands actually run, numerical comparisons, and remaining uncertainty. Prefer loading only this skill plus directly relevant neighboring skills.
