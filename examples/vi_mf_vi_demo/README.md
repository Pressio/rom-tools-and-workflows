# VI and MF-VI demo

This example compares single-fidelity variational inference with multifidelity
variational inference using automatic Gaussian-process ROM construction. Both
methods infer the diffusion coefficient and reaction rate in a steady
convection-diffusion-reaction model.

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
