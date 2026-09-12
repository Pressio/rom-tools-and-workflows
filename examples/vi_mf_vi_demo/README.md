# VI and MF-VI demo

This example compares single-fidelity variational inference with multifidelity
variational inference using automatic Gaussian-process ROM construction.

The forward problem is a steady linear convection-diffusion-reaction equation
on the unit square,

\[
-\nu \nabla^2 u + \mathbf{b}\cdot\nabla u + \sigma u = 1,
\qquad \mathbf{b}=(1,1),
\]

with homogeneous Dirichlet boundary conditions. The uncertain parameters are
the diffusion coefficient `nu` and reaction coefficient `sigma`. Synthetic data
are generated at the truth

- `nu = 0.04`
- `sigma = 0.30`

from the scalar quantity of interest used by the CDR model, which approximates
the integral of the normal derivative on the right boundary. No random noise
realization is added to the synthetic observation. The likelihood assumes
Gaussian observation noise with standard deviation `5e-3`, corresponding to a
scalar observation covariance of `2.5e-5`.

The inference bounds are deliberately not centered on the truth:

- `nu` is restricted to `[0.01, 0.10]`
- `sigma` is restricted to `[0.10, 0.80]`

The Gaussian prior uses the midpoint of these intervals as its mean and one
quarter of each interval width as its standard deviation, so its mean is
`(0.055, 0.45)` rather than the true parameter vector `(0.04, 0.30)`.

Run the full example from the repository root:

```bash
python examples/vi_mf_vi_demo/example.py
```

Run the reduced CI configuration:

```bash
python examples/vi_mf_vi_demo/example.py --smoke
```

The example writes separate ELBO and posterior-parameter convergence figures.
Use `--work-dir` to place workflow data elsewhere and `--output-dir` to select
the figure destination.
