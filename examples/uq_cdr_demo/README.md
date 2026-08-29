# Monte Carlo UQ for the CDR model

This example estimates the expected integrated right-boundary flux of the
steady convection-diffusion-reaction model in `examples/models`. It runs both
standard Monte Carlo and pilot-allocated two-level multifidelity Monte Carlo.
The MFMC high-fidelity model uses a 21-by-21 grid and its low-fidelity model
uses a 9-by-9 grid.

From the repository root, run:

```bash
python examples/uq_cdr_demo/example.py
```

Use `--output-directory PATH` to write results somewhere other than
`examples/uq_cdr_demo/uq_cdr_output`.
