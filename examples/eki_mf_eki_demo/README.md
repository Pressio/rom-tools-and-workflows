# EKI and MF-EKI Demo

Runs the EKI vs. MF-EKI demo on a 2D convection-diffusion-reaction model with a lightweight synthetic observation. The example compares single-fidelity EKI, MF-EKI with a tailored projection ROM, MF-EKI with automatic Gaussian-process surrogate construction, and MF-EKI with a Lipschitz-constrained neural-network surrogate.

The neural-network case uses the standard two-hidden-layer baseline, trains with Adam for 5,000 iterations in the full benchmark, and estimates its global Lipschitz constant from the normalized training data. PyTorch support is optional and can be installed with `romtools[WithTorch]`.

## Run

```bash
python examples/eki_mf_eki_demo/example.py
```

For a reduced CI-style validation run:

```bash
python examples/eki_mf_eki_demo/example.py --smoke
```

The smoke configuration exercises the same four model paths but reduces neural-network training to 100 Adam iterations.

## Documentation figure

This file is the canonical implementation used by the documentation. To regenerate the checked-in documentation figure explicitly:

```bash
python examples/eki_mf_eki_demo/example.py \
  --output docs/source/demos/notebooks/eki_mf_eki_demo.svg
```

The Sphinx build does not run the full benchmark implicitly. CI exercises the same code path with `--smoke` so API drift is caught without paying the full benchmark cost on every pull request.
