# Monte Carlo UQ for the CDR model

This example estimates the expected integrated right-boundary flux of a steady
convection-diffusion-reaction problem and compares standard Monte Carlo with
pilot-allocated two-level multifidelity Monte Carlo.

The uncertain parameters are the advection magnitude and direction, diffusion
coefficient, and reaction coefficient. The MFMC high-fidelity model uses a
21-by-21 grid and its low-fidelity model uses a 9-by-9 grid. The allocator uses
the correlation estimated from 30 paired pilot samples.

From the repository root, run the lightweight example with

```bash
python examples/uq_cdr_demo/example.py
```

Use `--output-directory PATH` to write results somewhere other than
`examples/uq_cdr_demo/uq_cdr_output`.

The full documentation benchmark is intentionally separate from the smoke test:

```bash
python examples/uq_cdr_demo/benchmark.py
```

It evaluates MC and MFMC at several equal high-fidelity-equivalent budgets and
generates the convergence and pilot-correlation figures. The committed
`reference_stats.json` was generated from 20,000 high-fidelity samples using

```bash
python examples/uq_cdr_demo/generate_reference.py
```

Reference generation is not run as part of normal CI or documentation builds.
