# Agent skills benchmark

The pilot evaluates task success and token efficiency across three conditions:
no guidance, root AGENTS.md only, and full agent skills. Default task:
`cdr_model_adapter`, which adapts an intentionally nonconforming legacy CDR
solver to the current ROM Tools Model/QoI API and executes a sampling workflow.

## First validate the benchmark oracle

```bash
python -m pytest -q benchmarks/agent_skills/reference/tests
```

The trusted reference adapter and tests live under `reference/` and are **not
copied to the agent workspace**. The task grader under `tasks/*/tests` is also
removed from the exported workspace and copied in only after the agent exits.
This is withheld grading, not cryptographic secrecy: prompts and source code
remain public, and benchmarks should not be used for high-stakes ranking.

## Run the pilot

With an installed, authenticated Codex CLI and verified model IDs:

```bash
codex login status
codex debug models
python -m pytest -q benchmarks/agent_skills/reference/tests benchmarks/agent_skills/tests
python benchmarks/agent_skills/run.py --models MODEL_ID_A \
  --tasks cdr_model_adapter --conditions full --trials 1 \
  --keep-workspaces --output /tmp/romtools-single-pilot
```

Use a fresh output directory for each experiment. The runner exports committed
`HEAD`, so commit benchmark changes before running it. Retained workspaces contain
the agent submission before trusted grading files are restored. Review them and
the JSONL logs for actual skill reads, API inspection and verification evidence.
Credentials and sensitive data must never be placed in the agent workspace.

Only after a successful isolated pilot, run the comparison:

```bash
python benchmarks/agent_skills/run.py --models MODEL_ID_A MODEL_ID_B \
  --tasks cdr_model_adapter --trials 5 --output /tmp/romtools-eval
python benchmarks/agent_skills/summarize.py /tmp/romtools-eval/results.jsonl
```

The runner exports a Git snapshot without `.git`, randomizes run order with
a fixed seed, enforces timeouts, records the final turn token usage when emitted,
and grades independently. Null usage means missing instrumentation, not zero.
Model names and CLI compatibility must be checked against the installed CLI.
The runner explicitly requests `workspace-write` sandboxing. Nonzero agent exits
and timeouts are recorded as ungraded `agent_error` and `timeout` outcomes;
they are not numerical verification failures. Inspect stderr to distinguish
startup infrastructure errors from other agent errors. Missing token usage is
reported as unavailable by the summary command.

**Security:** agents execute code. Use a disposable container or machine
without credentials, sensitive files, or HPC access. Codex's own sandbox
settings and permissions still apply. No agent execution has been performed
by adding these files. Exporting a workspace does not itself hide the original
checkout, grading files, credentials or connected services: enforce those read
boundaries using the execution environment before conducting a benchmark.

The 2026-11-17 Work environment pilot was blocked before inference by a sandbox
network namespace restriction. See [the diagnostic](diagnostics/work_mode_pilot.md)
for the reproduction and verified checks. Use a compatible host/runtime; do not
disable sandbox protections to turn a startup failure into a benchmark result.

## Interpretation

Primary outcomes: pass rate and total tokens per successful run. Also report
median tokens conditional on success, elapsed time, and failure categories.
Use multiple trials, compare matched tasks, and report uncertainty intervals.
Do not infer skill use merely from the presence of skill files; a separate
skill-discovery trace check is needed before making that claim.
