# H2-air flame EKI/MF-EKI rejuvenation benchmark

This example benchmarks ensemble collapse and adaptive rejuvenation on the four-parameter H2-air flame inverse problem. It compares:

- single-fidelity EKI,
- GP auto-ROM MF-EKI with rejuvenation disabled, and
- GP auto-ROM MF-EKI with `rejuvenation_strategy="adaptive"`.

The two MF-EKI runs use the same seed, ensemble sizes, ROM settings, parameter bounds, and solver controls. The only intended algorithmic difference is adaptive rejuvenation. The full benchmark uses a `64 x 32` flame model, four FOM ensemble members, 32 additional ROM-only members, and up to 30 outer iterations.

The benchmark writes a JSON history plus SVG/PNG figures for the observation residual, RMS relative parameter error, and range-scaled ensemble spread.

## Run

```bash
python examples/h2_air_flame_eki_mf_eki/example.py
```

A small CI/API validation is available with:

```bash
python examples/h2_air_flame_eki_mf_eki/example.py --smoke
```

To place outputs explicitly:

```bash
python examples/h2_air_flame_eki_mf_eki/example.py \
  --work-dir /tmp/h2-air-flame-work \
  --output-dir /tmp/h2-air-flame-results
```

The documentation page uses this file as the canonical implementation; Sphinx does not execute the full benchmark during a normal documentation build.