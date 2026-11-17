---
name: sampling-workflows
description: Use when creating, modifying, or running sampling and ensemble evaluation workflows.
---
# Sampling workflows
Start with `romtools/workflows/sampling/`, relevant tests under `tests/romtools/workflows/`, and the examples in `examples/`. Verify current public entry points, model contracts, concurrency and run-directory semantics. Reuse existing models and parameter spaces when supplied. Consult `hpc-execution` only for dispatcher or remote-job work. Do not automatically run a workflow merely because the user asked to draft one.
