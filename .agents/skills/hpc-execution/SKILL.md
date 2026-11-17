---
name: hpc-execution
description: Use for dispatchers, Slurm, remote execution, model concurrency, and blocking/nonblocking job semantics.
---
# HPC execution

## When to use
Use for dispatchers, Slurm, remote execution, model concurrency, and blocking/nonblocking job semantics.

## Read first
`romtools/hpc/README.md`; `romtools/hpc/dispatchers/`; `romtools/hpc/components/slurm_job_manager.py`; `romtools/workflows/model_execution.py`; `romtools/hpc/example/`.

## Procedure
1. Separate local filesystem operations, remote staging, scheduler submission, polling, and result collection.
2. Read the current dispatcher interface and distinguish blocking model exit codes from nonblocking `JobHandle` completion.
3. Bound concurrency explicitly and ensure caller cwd and temporary files are restored on errors.
4. Never submit remote jobs, access HPC allocations, or transfer large data without explicit authorization.

## Verification
Mock scheduler and dispatcher boundaries; test failed submission, pending/completed/failed states, cleanup, concurrent runs, and local fallback. A scheduler job ID is not proof of success.

## Handoff
State which implementation and tests were inspected, what changed, commands actually run, numerical comparisons, and remaining uncertainty. Prefer loading only this skill plus directly relevant neighboring skills.
