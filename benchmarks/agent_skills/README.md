# Agent skill evaluation (pilot)

This pilot measures (1) efficiency and (2) task completion across model tiers.
It uses the **Codex CLI** as a fixed harness. Model names are configurable:
the names discussed in conversation (such as "6.1 Sol" and "Tara") are
*not* assumed to be valid API/Codex model IDs. Verify accessible model IDs
and supply them through `--models`.

## Design
- Conditions: `none` (no skill files), `guide` (root AGENTS.md only), `full` (AGENTS.md and canonical skills).
- Models: user-supplied Codex-supported IDs, e.g. `--models MODEL_A MODEL_B`.
- Tasks: versioned task directories with `prompt.md`, `tests/`, and `task.json`.
- Repeats: run at least 5 independent trials per cell, randomize condition order
  during analysis, and report confidence intervals.
- Metrics: verified pass/fail, input/output/cached tokens when emitted by
  Codex JSON events, tool/runtime duration, and cost when known.
- Do not equate fewer tokens with improved performance: report success rates
  and total tokens per success alongside median tokens for successful runs.

## Usage
Install and authenticate Codex CLI separately. From a trusted machine with
the required dependencies, run:

```bash
python benchmarks/agent_skills/run.py --models MODEL_A MODEL_B --trials 5 --output /tmp/romtools-agent-eval
python benchmarks/agent_skills/summarize.py /tmp/romtools-agent-eval/results.jsonl
```

The runner makes temporary git worktrees from the current checked-out commit.
It never modifies the source checkout, but **agents execute code**; use an
isolated machine/container with no secrets, no HPC credentials, and no
production filesystem mounts. The runner uses Codex's default sandbox
restrictions and never passes a bypass-approval flag. The test verifier runs
in the isolated environment too.

## Interpreting results
A run is successful only when the independent verifier exits zero. A missing
token-usage event is recorded as null, not zero. The benchmark excludes
network access and remote HPC runs. Model availability, pricing, and Codex
CLI flags must be checked against the installed version before a full run.
Benchmark tasks are intentionally small pilots; expand to EKI and MF-EKI
workflows once deterministic gold-standard fixtures are prepared.
