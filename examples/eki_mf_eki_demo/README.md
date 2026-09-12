# EKI and MF-EKI Demo

Runs the EKI vs. MF-EKI demo on a 2D convection-diffusion-reaction model with a lightweight synthetic observation. The example compares single-fidelity EKI, MF-EKI with a tailored projection ROM, and MF-EKI with automatic Gaussian-process ROM construction.

## Run

```bash
python examples/eki_mf_eki_demo/example.py
```

For a reduced CI-style validation run:

```bash
python examples/eki_mf_eki_demo/example.py --smoke
```

## Documentation figure

This file is the canonical implementation used by the documentation. To regenerate the checked-in documentation figure explicitly:

```bash
python examples/eki_mf_eki_demo/example.py \
  --output docs/source/demos/notebooks/eki_mf_eki_demo.png
```

The Sphinx build does not run the benchmark implicitly. CI exercises the same code path with `--smoke` so API drift is caught without paying the full benchmark cost on every pull request.
