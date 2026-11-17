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
python benchmarks/agent_skills/run.py --models MODEL_ID_A MODEL_ID_B \
  --tasks cdr_model_adapter --trials 5 --output /tmp/romtools-eval
python benchmarks/agent_skills/summarize.py /tmp/romtools-eval/results.jsonl
```

The runner exports a Git snapshot without `.git`, randomizes run order with
a fixed seed, enforces timeouts, records the final turn token usage when emitted,
and grades independently. Null usage means missing instrumentation, not zero.
Model names and CLI compatibility must be checked against the installed CLI.

**Security:** agents execute code. Use a disposable container or machine
without credentials, sensitive files, or HPC access. Codex's own sandbox
settings and permissions still apply. No agent execution has been performed
by adding these files.

## Interpretation

Primary outcomes: pass rate and total tokens per successful run. Also report
median tokens conditional on success, elapsed time, and failure categories.
Use multiple trials, compare matched tasks, and report uncertainty intervals.
Do not infer skill use merely from the presence of skill files; a separate
skill-discovery trace check is needed before making that claim.
