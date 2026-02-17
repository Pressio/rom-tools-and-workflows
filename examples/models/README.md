# Models

Walks through romtools model abstractions, from simple inline functions to external applications.

## Run

```bash
python example.py
```

### Steady CDR model

This folder now includes a steady convection-diffusion-reaction (CDR) model
compatible with romtools workflow interfaces:

- `steady_cdr.py`: PDE operators and steady solve routine.
- `steady_cdr_model.py`: `SteadyCdrQoiModel` with `populate_run_directory`,
  `run_model`, and `compute_qoi`.

Run a quick standalone model evaluation:

```bash
python steady_cdr_model.py
```

## Notes

- Notebooks: `docs/source/demos/notebooks/basic_model.ipynb`, `external_model.ipynb`, `external_qoi_model.ipynb`.
- Use Jupyter to execute the model tutorials.
