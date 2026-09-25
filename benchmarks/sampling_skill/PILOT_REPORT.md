# Sampling skill pilot — 2026-09-09

The skill used fewer total tokens in all three task pairs in this Codex pilot.
All six sessions passed the objective output checks. This is one repetition per
task and condition, so it is preliminary evidence, not a general savings estimate.

## Results

Provider: Codex CLI 0.153.4. Requested model: `gpt-6-astra`, medium effort.
Protocol: explicitly read the shared skill file, fresh workspace/profile/session,
fully specified local tasks with advance authorization, identical deliverables.

| Task | Without skill | With skill | Token reduction | Objective checks |
|---|---:|---:|---:|---|
| Run existing workflow | 113,309 | 98,990 | 12.6% | Both passed |
| Reuse existing model | 137,535 | 85,430 | 37.9% | Both passed |
| Generate model and workflow | 111,571 | 86,694 | 22.3% | Both passed |
| Total for this task mix | 362,415 | 271,114 | 25.2% | 6/6 passed |

Counts include input and output, with cached input counted once. Cache read/write
counts are separately preserved in the contribution. These are token reductions,
not dollar-cost reductions. The total is weighted by this particular task mix;
it is not a population estimate. Three repetitions per task/condition remain the
default for a larger exploratory run; more repetitions are needed for confidence.
