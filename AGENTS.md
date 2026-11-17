# ROM Tools agent guide

This file is a navigation aid, not a substitute for the current source code or tests.

## Repository map
- `romtools/workflows/`: sampling, adaptive sampling, inverse methods, and UQ
- `romtools/rom/`, `romtools/vector_space/`, `romtools/linalg/`: reduced models, bases, and linear algebra
- `romtools/hyper_reduction/`: hyper-reduction methods
- `romtools/hpc/`: dispatchers and remote execution
- `tests/romtools/`: unit and regression tests
- `examples/`: runnable demonstrations
- `docs/source/`: user and API documentation

## Working conventions
1. Inspect the current implementation, public exports, and nearest tests before editing.
2. Preserve public interfaces and existing workflow behavior unless the task explicitly changes them.
3. Add focused tests for bug fixes; run the relevant tests before claiming success.
4. For changes affecting numerical results, check regression baselines and justify any updated gold data.
5. Do not launch remote jobs or consume HPC allocations without explicit authorization.
6. Prefer links to canonical code, tests, and demos over duplicating detailed APIs here.

## Task-specific guidance
Load only the relevant skill in `.agents/skills/`. Each `SKILL.md` declares when it applies and identifies authoritative source and test locations. Read additional skills only when the task spans subsystems.
