---
name: sampling-workflows
description: Use when assembling a sampling study, running parameter sweeps, computing QoI statistics, or changing sample execution.
---
# Sampling workflows

## When to use
Use when assembling a sampling study, running parameter sweeps, computing QoI statistics, or changing sample execution.

## Read first
`romtools/workflows/sampling/sampling.py`; `romtools/workflows/parameter_spaces.py`; `examples/minimal_sampling_workflow.py`; `tests/romtools/workflows/sampling/test_sampling.py`.

## Procedure
1. Start from `run_sampling` signature: model, parameter space, absolute work directory, sample count, seed, concurrency and optional dispatcher.
2. Check `get_names`, `get_dimensionality`, and `generate_samples`; parameter names must match the model's dictionary keys.
3. `populate_run_directory` prepares inputs, `run_model` executes, and `compute_qoi` extracts outputs after successful execution.
4. Expect `sample_parameters.txt`, per-sample `run_N/` directories, `passed.txt`, and `sampling_stats.npz` when QoIs are available; inspect current implementation for overwrite and dry-run behavior.
5. Use local serial execution for minimal examples. Remote jobs require separate authorization.

## Verification
Assert sample bounds/order and deterministic seeding; compare saved state/QoI to a direct solve, and test resume/failure behavior when changing execution logic. Do not assume `random_seed` reaches every custom sampler identically.

## Reviewable execution trace
Follow the common protocol in `.agents/TRACE_PROTOCOL.md`. For this task specifically: Record parameter-space configuration, seed, model API mapping, run-directory inventory, per-run statuses, QoI statistics, and failed sample IDs.

## Handoff
State which implementation and tests were inspected, what changed, commands actually run, numerical comparisons, and remaining uncertainty. Prefer loading only this skill plus directly relevant neighboring skills.
