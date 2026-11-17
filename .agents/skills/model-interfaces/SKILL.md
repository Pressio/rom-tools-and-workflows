---
name: model-interfaces
description: Use when wrapping a solver, implementing model execution, computing QoIs, or debugging nonblocking runs.
---
# Model API adaptation

## When to use
Use when wrapping a solver, implementing model execution, computing QoIs, or debugging nonblocking runs.

## Read first
`romtools/workflows/models.py`; `romtools/workflows/model_execution.py`; `examples/models/steady_cdr_model.py`; `examples/minimal_sampling_workflow.py`; `tests/romtools/workflows/sampling/test_sampling.py`.

## Procedure
1. Read `Model`, `QoiModel`, and the workflow caller. Do not infer the contract from a similarly named legacy model.
2. Implement `populate_run_directory(run_directory, parameter_sample)` to stage inputs; `run_model(run_directory, parameter_sample)` to execute and return integer exit code for blocking execution; optionally `compute_qoi(run_directory, parameter_sample)` to read persisted results and return a numeric scalar/vector.
3. Keep parameter dictionary names consistent with the parameter space; use absolute run-directory paths, avoid depending on process cwd, and preserve solver numerics.
4. For nonblocking models, inspect `JobHandle` and executor behavior before returning a handle; do not report completion at submission time.

## Verification
Test direct method calls and a real `run_sampling` integration. Compare state and QoI against a solver reference; test failures, reruns, and saved files. For the CDR adapter benchmark, never edit the legacy solver.

## Handoff
State which implementation and tests were inspected, what changed, commands actually run, numerical comparisons, and remaining uncertainty. Prefer loading only this skill plus directly relevant neighboring skills.
