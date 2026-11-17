# ROM Tools agent guide

This is a short router; authoritative contracts live in source code and tests.

## Start here
1. Locate the requested subsystem, public exports, nearest tests, and a runnable example before editing.
2. Load the one most relevant skill under `.agents/skills/<topic>/SKILL.md`; load a second only for a real cross-subsystem dependency. Do not preload all skills.
3. Preserve public interfaces, numerical results, file conventions, and backward compatibility unless explicitly instructed otherwise.
4. Prefer a minimal reproducible test and numerical oracle. Report commands run and distinguish passing, failing, and unexecuted checks.
5. Never submit HPC jobs, use paid compute, or change remote state without authorization.

## Task routing
- Adapt an external solver or QoI → `model-interfaces`; then `sampling-workflows` for sampling.
- Parameter bounds/distributions → `parameter-spaces`.
- Monte Carlo or run directories → `sampling-workflows`.
- Greedy, space-filling, hybrid selection → `adaptive-sampling`.
- EKI, EGO, VI → `inverse-workflows`; MF-EKI/MF-VI also → `multifidelity-methods`.
- POD, projection, basis → `rom-construction`; DEIM/QDEIM/ECSW → `hyper-reduction`.
- Slurm, dispatcher, concurrency → `hpc-execution`.
- Sphinx or demo pages → `documentation`.
- Test failures or regression gold → `testing-and-ci`.

## Repository map
`romtools/workflows/`: models, sampling, inverse, adaptive; `romtools/rom/`, `romtools/vector_space/`, `romtools/linalg/`: ROM foundations; `romtools/hyper_reduction/`: hyper-reduction; `romtools/hpc/`: dispatch; `tests/romtools/`: tests; `examples/`: demos; `docs/source/`: documentation.

## Completion report
Summarize changed files, behavior and compatibility, numerical validation, exact tests and results, and unverified assumptions. Do not claim an agent discovered or used a skill merely because its files were present.
