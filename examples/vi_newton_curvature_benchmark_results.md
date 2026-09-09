# Newton curvature coupling benchmark

The analytic benchmark uses a standard-normal target and evaluates the VI
estimators at the exact standard-normal variational optimum. The exact gradient
is zero. These results were generated with:

```bash
python -m examples.vi_newton_curvature_benchmark --repetitions 5000
```

The two entries in each value are the mean and log-standard-deviation
coordinates. Parenthesized values are 95% Monte Carlo half-widths for the mean
Newton direction.

| Samples | Strategy | Mean gradient | Mean Newton direction (95% half-width) |
|---:|---|---|---|
| 16 | same sample | `6.21e-3, -1.32e-3` | `4.43e-2 (6.01e-2), 1.40e0 (8.34e-2)` |
| 16 | independent | `6.21e-3, -1.32e-3` | `1.03e-2 (1.18e-1), 1.02e-1 (2.21e-1)` |
| 16 | lagged, beta=0.9 | `6.21e-3, -1.32e-3` | `-2.12e-2 (3.32e-2), 6.11e-2 (1.40e-1)` |
| 64 | same sample | `-3.25e-3, 1.40e-3` | `-1.23e-2 (6.17e-2), 1.66e0 (1.21e-1)` |
| 64 | independent | `-3.25e-3, 1.40e-3` | `-2.55e-2 (6.33e-2), 9.43e-2 (1.35e-1)` |
| 64 | lagged, beta=0.9 | `-3.25e-3, 1.40e-3` | `-3.51e-3 (6.28e-3), 3.39e-3 (9.00e-3)` |
| 256 | same sample | `-1.88e-3, 1.54e-3` | `-4.55e-3 (2.40e-2), 6.99e-1 (8.83e-2)` |
| 256 | independent | `-1.88e-3, 1.54e-3` | `-9.15e-3 (1.32e-2), 6.98e-3 (5.66e-2)` |
| 256 | lagged, beta=0.9 | `-1.88e-3, 1.54e-3` | `-1.96e-3 (2.86e-3), 1.11e-3 (3.65e-3)` |

The gradient mean is statistically consistent with zero. The same-sample
log-standard-deviation direction remains clearly nonzero, while independent
and lagged curvature substantially reduce it. This is a stochastic diagnostic;
small changes are expected when the seed or repetition count changes.
