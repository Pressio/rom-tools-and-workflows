# H2-air flame VI benchmark

This benchmark applies romtools variational-inference workflows to the existing
pure-Python H2-air flame model in `examples/models`. It intentionally reuses
`H2AirFlameQoiModel`; the flame solver and observation operator are not copied
into the benchmark.

The inverse problem estimates

- `kappa`,
- `scaled_activation_energy`,
- `beta_x`, and
- `beta_y`.

The synthetic truth is `[2.0, 8.0, 40.0, 7.0]`. The initial variational mean is
deliberately offset from that truth. A single noisy temperature-sensor
observation is generated with a fixed seed and a 5% global RMS noise-to-signal
ratio,

```text
sigma_obs = 0.05 * ||y_clean||_2 / sqrt(N_obs).
```

The same observation is reused by every method and algorithmic seed.

## Method ablation

The primary comparison contains six paths:

1. `BBVI (Adam)` -- single-fidelity Adam BBVI;
2. `ARBIS` -- Adam BBVI with importance-sampling/sample reuse;
3. `ARBIS + MF` -- Adam sample reuse plus automatic GP multifidelity variance reduction;
4. `BBVI (Newton)` -- single-fidelity Newton BBVI;
5. `MF-BBVI (Newton, analytic)` -- automatic-GP MF-BBVI with analytic Gaussian entropy;
6. `MF-BBVI (Newton, joint)` -- the same MF-Newton setup with sampled joint entropy.

Every score-function path uses the leave-one-out (`loo`) baseline. The two
MF-Newton cases intentionally differ only in `score_function_entropy_strategy`.
All multifidelity cases use the romtools automatic Gaussian-process QoI model,
including POD reduction for the vector-valued sensor QoI and normalized
parameters/targets.

Run the production ablation with

```bash
python examples/h2_air_flame_vi_benchmark/benchmark.py
```

The production model uses the same `64 x 32`, `dt=1e-3`, `t_end=6e-2` setup as
the existing H2-air flame EKI benchmark.

## Sample-size sweep

The sweep is also YAML-driven:

```bash
python examples/h2_air_flame_vi_benchmark/benchmark.py \
  --config examples/h2_air_flame_vi_benchmark/configs/sample_size_sweep.yaml
```

The default production sweep uses FOM sample sizes `[4, 8, 16, 32]` and
MF ROM-extra sample sizes `[64, 128]`. A sweep case changes only those sample
counts; the model, observation realization, prior, initial variational state,
optimizer settings, and GP configuration remain fixed.

## Smoke validation

A reduced configuration exercises the real flame model and all six inference
paths:

```bash
python examples/h2_air_flame_vi_benchmark/benchmark.py --smoke
```

The reduced sweep can be exercised with

```bash
python examples/h2_air_flame_vi_benchmark/benchmark.py \
  --smoke --mode sweep --methods bbvi_adam arbis_mf
```

## Configuration

The benchmark is configured by

- `configs/method_ablation.yaml`,
- `configs/sample_size_sweep.yaml`, and
- `configs/smoke.yaml`.

The YAML owns the flame mesh/time integration, sensor layout, parameter bounds,
truth, prior, variational initializer, observation noise, seeds, optimizer and
line-search settings, sample-reuse controls, GP/POD controls, entropy strategy,
and sample counts. The fully resolved configuration is copied into each result
directory.

## Cost accounting

The primary work metric is the cumulative number of H2-air flame FOM solves.
The postprocessor counts the FOM-specific `solution.npz` artifacts under each
VI iteration. This therefore includes ordinary gradient samples, line-search
FOM calls, automatic-GP training/rebuild calls, and any optimizer-specific FOM
work, while excluding GP evaluations.

`fom_budget` is a matched-work **analysis cap**. romtools VI currently completes
an iteration atomically, so the benchmark does not interrupt a line search or
model batch in the middle of an iteration. Histories and comparison plots are
truncated at the last completed iteration within the common FOM budget.
Wall-clock time is recorded separately.

## Outputs

The ablation writes machine-readable JSON plus PNG/SVG plots for

- parameter error vs cumulative FOM evaluations,
- ELBO vs cumulative FOM evaluations,
- observation-space relative MSE vs cumulative FOM evaluations,
- each posterior-mean component vs cumulative FOM evaluations with the truth marked,
- a matched-work comparison of the two MF-Newton entropy strategies, and
- median/interquartile bands across repeated algorithmic seeds.

Each run also records its final posterior mean/covariance/std, wall time,
resolved configuration, and sample-reuse diagnostics when available. The fixed
observation seed, clean/noisy observations, `sigma_obs`, realized noise ratio,
and QoI dimension are saved separately so every figure can be regenerated
without rerunning the flame truth solve.
