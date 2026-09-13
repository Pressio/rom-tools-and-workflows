# Models

Walks through romtools model abstractions, from simple inline functions to external applications.

## Run

```bash
python example.py
```

### Steady CDR model

This folder includes a steady convection-diffusion-reaction (CDR) model
compatible with romtools workflow interfaces:

- `steady_cdr.py`: PDE operators and steady solve routine.
- `steady_cdr_model.py`: `SteadyCdrQoiModel` with `populate_run_directory`,
  `run_model`, and `compute_qoi`.

Run a quick standalone model evaluation:

```bash
python steady_cdr_model.py
```

### H2-air flame model

This folder also includes a nonlinear transient premixed H2-air flame benchmark
implemented entirely in Python with NumPy and SciPy:

- `h2_air_flame.py`: four-field advection-diffusion-reaction solver with the
  parameters `kappa`, `scaled_activation_energy`, `beta_x`, and `beta_y`.
- `h2_air_flame_model.py`: sensor-based `H2AirFlameQoiModel` wrapper for
  romtools workflows.

Run the lightweight wrapper smoke example with:

```bash
python h2_air_flame_model.py
```

See `docs/source/demos/h2_air_flame.rst` for the model description and usage.

### Solid dynamics model

A lightweight 2-D solid dynamics reference model is also provided:

- `nonlinear_solid_dynamics.py`: structured Q4 total-Lagrangian finite-element
  solver with a compressible Neo-Hookean material, explicit velocity-Verlet and
  implicit velocity-primary Newmark integration.
- `solid_dynamics.py`: common facade adding small-strain isotropic linear
  elasticity, shared model constructors, Gaussian initial conditions, and the
  analytic two-way Gaussian wave solution used for convergence checks.
- `nonlinear_solid_dynamics_model.py`: parameterized cantilever QoI wrapper for
  romtools workflows.

User-facing beam, doubly clamped Gaussian-perturbation, analytic linear-wave
convergence, and published nonlinear benchmark examples live in
`examples/nonlinear_solid_dynamics`. See
`docs/source/demos/nonlinear_solid_dynamics.rst` for the formulation and
validation notes.

## Notes

- Notebooks: `docs/source/demos/notebooks/basic_model.ipynb`, `external_model.ipynb`, `external_qoi_model.ipynb`.
- Use Jupyter to execute the model tutorials.
