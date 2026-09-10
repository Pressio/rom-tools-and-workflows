# Score-function entropy benchmark

These results were generated with the default configuration:

```bash
python -m examples.vi_score_entropy_benchmark \
  --output-dir work/score_entropy
```

The run used 5,000 direct repetitions, 20 convergence seeds, 500 iterations,
sample counts of 16, 64, and 256, and a lagged-Hessian averaging factor of 0.9.

| Samples | Entropy | Mean-gradient variance | Log-std-gradient variance |
|---:|---|---:|---:|
| 16 | analytic | 1.598181e-1 | 9.194271e-1 |
| 16 | joint | 0.0 | 0.0 |
| 64 | analytic | 3.940581e-2 | 2.171781e-1 |
| 64 | joint | 0.0 | 0.0 |
| 256 | analytic | 9.779622e-3 | 5.453226e-2 |
| 256 | joint | 0.0 | 0.0 |

The analytic estimator has the expected approximate `1/N` variance scaling.
With the leave-one-out baseline, the joint estimator is zero sample by sample
at the exactly representable posterior.

| Samples | ROM | Coefficient | MF gradient variance | Mean alpha norm |
|---:|---|---|---:|---:|
| 16 | exact | fitted | 0.0 | 0.0 |
| 16 | perturbed | fitted | 0.0 | 0.0 |
| 16 | perturbed | fixed | 7.474163e-3 | 1.414214 |
| 64 | exact | fitted | 0.0 | 0.0 |
| 64 | perturbed | fitted | 0.0 | 0.0 |
| 64 | perturbed | fixed | 1.688764e-3 | 1.414214 |
| 256 | exact | fitted | 0.0 | 0.0 |
| 256 | perturbed | fitted | 0.0 | 0.0 |
| 256 | perturbed | fixed | 4.126720e-4 | 1.414214 |

The zero-HF-variance guard suppresses fitted ROM corrections. As requested,
the fixed coefficient remains one; a perturbed ROM therefore leaves a residual
whose variance also decreases approximately as `1/N`.

| Samples | Entropy | Curvature | Median final KL | Median final accepted step |
|---:|---|---|---:|---:|
| 16 | analytic | same sample | 6.775965e-14 | 0.0 |
| 16 | analytic | lagged | 1.699319e-15 | 0.0 |
| 16 | joint | same sample | 1.970288e-17 | 7.663538e-1 |
| 16 | joint | lagged | 0.0 | 1.0 |
| 64 | analytic | same sample | 1.192940e-15 | 0.0 |
| 64 | analytic | lagged | 7.222735e-16 | 0.0 |
| 64 | joint | same sample | 2.956693e-18 | 5.999217e-1 |
| 64 | joint | lagged | 0.0 | 1.0 |
| 256 | analytic | same sample | 4.774353e-16 | 0.0 |
| 256 | analytic | lagged | 6.488158e-16 | 0.0 |
| 256 | joint | same sample | 0.0 | 1.0 |
| 256 | joint | lagged | 0.0 | 1.0 |

This analytic line-search benchmark permits the analytic estimator to reach a
small KL by shrinking its accepted step to zero. The joint estimator reaches
the fixed point while retaining a finite accepted step. The generated raw CSV
histories and plots show the corresponding early-iteration behavior.
