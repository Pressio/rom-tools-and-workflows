---
name: hpc-execution
description: Use when configuring dispatchers, remote execution, Slurm jobs, or concurrency of model evaluations.
---
# HPC execution
Read `romtools/hpc/README.md`, `romtools/hpc/dispatchers/`, and matching tests. Distinguish local versus remote dispatch, blocking versus nonblocking execution, and job-submission versus completion. Before initiating a remote job, explain the target and allocation implications and obtain explicit authorization. Never infer job success from submission alone.
