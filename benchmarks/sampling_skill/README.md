# Sampling workflow skill benchmark

This harness runs paired, fresh CLI sessions against small local tasks. It uses
the single skill source at `.claude/skills/workflow/SKILL.md`.

See [the completed Codex pilot](PILOT_REPORT.md) for initial measurements and
quality-review notes. It contains one pair per task, not a full repeated study.

## Quick start

Use Python 3.9+ on macOS/Linux, with the repository dependencies installed.
Install and authenticate your chosen CLI separately. Run from the repository
root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s benchmarks/sampling_skill -p 'test_*.py'

# No model calls: inspect exact prompts and manifest first.
.venv/bin/python benchmarks/sampling_skill/run.py --provider codex --model gpt-6-astra --tasks existing --repeats 1 --dry-run

# Pilot: two model sessions; consumes your provider usage allowance.
.venv/bin/python benchmarks/sampling_skill/run.py --provider codex --model gpt-6-astra --tasks existing --repeats 1

# Full exploratory study: 18 sessions (3 tasks x 2 conditions x 3 repetitions).
.venv/bin/python benchmarks/sampling_skill/run.py --provider codex --model gpt-6-astra --repeats 3

.venv/bin/python benchmarks/sampling_skill/summarize.py benchmarks/sampling_skill/results
```

Claude requires `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` in the environment
for the isolated profile (subscription users can obtain a token via `claude
setup-token`).

- `--python` selects the workflow interpreter (defaults to the runner's interpreter).
- `--executable` selects a CLI binary.
- `--effort low|medium|high` controls requested reasoning effort.
- `--skill` selects another skill version.
- `--timeout` bounds each session in seconds.

No retries are automatic. All trial outputs stay under
the ignored `results/` directory.
Every invocation creates a unique run directory.

## Experimental protocol

Tasks: execute a supplied workflow; generate a workflow reusing a supplied model;
generate both model and workflow. Edit `fixtures.py` to introduce new tasks and
extend the validator at the same time. Changes create a new harness hash.

Each trial starts a new process without resume/continue, a new temporary home,
and a new workspace containing only `romtools/` and package metadata plus its
fixture. Previous answers, repository instruction files, worked skill examples,
other skills, user history/config, and project plugins are not copied. Baseline
receives no skill file. Treatment receives the exact skill in
`workflow_skill/SKILL.md` and one additional instruction to read and follow it.
This measures **explicit skill use**, not automatic skill discovery. The file is
outside native skill discovery directories so both CLIs use the same mechanism.
The extra instruction and skill read are part of measured usage.

Both conditions receive identical deliverable requirements, including AUDIT.md.
Both explicitly preauthorize the fully specified local file writes and run. This
overrides the skill's draft-confirm pause; the experiment does **not** measure its
normal interactive approval behavior. Authorization must be described honestly.
Paired order alternates by task and repetition. Fresh sessions do not guarantee
cold provider caches; cache use is reported separately, not forcibly disabled.

## Measurements and quality

`result.json` records CLI/model/effort, hashes of source, skill, prompt and harness,
elapsed session time, process/provider completion, usage, and quality checks.
Dependency versions and repository commit are recorded; the source hash also
captures working-tree changes. Groups separate dependency versions and platforms.
Raw provider events and stderr are retained locally for inspection.

## Add another AI

Add a command builder and usage parser in `providers.py`, register its name in
`run.py`, and define its isolated authentication environment. The command must
accept the prompt on stdin, run tools in the supplied working directory, avoid
resuming sessions, and return machine-readable aggregate usage. Add parser tests
for cache accounting, errors and missing usage. Do not estimate missing tokens
from text length. Record provider-specific limits and validate a pilot before
combining results. The current adapters are Codex and Claude Code; arbitrary
providers are an extension point, not claimed live-tested support.

References: [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)
and [Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference).

---

## `run.py`

```bash
.venv/bin/python benchmarks/sampling_skill/run.py \
  --provider codex --model gpt-6-astra \
  --tasks existing --repeats 1
```

Runs paired baseline/skill benchmark trials and records token usage, artifacts,
and quality results. Execution order:
- (A1) `fixtures.py` + (A2) `providers.py`
- Provider CLI/workflow
- (B) `check_outputs.py`
- (C) `replay.py`

### (A1) `fixtures.py`

```bash
.venv/bin/python -c \
  'from benchmarks.sampling_skill.fixtures import TASKS; print(TASKS["existing"])'
```

Defines the deterministic toy model, reference workflow, and prompts used by
benchmark trials. It is a support module imported by `run.py`, not a standalone
command.

### (A2) `providers.py`

```bash
.venv/bin/python -c \
  'from benchmarks.sampling_skill.providers import command; print(command("codex", "codex", "gpt-6-astra", "medium"))'
```

Builds isolated Codex or Claude CLI or other commands and parses their output
and token usage. It also determines whether the provider completed successfully
and is imported by `run.py`.

### (B) `check_outputs.py`

```bash
.venv/bin/python benchmarks/sampling_skill/check_outputs.py \
  benchmarks/sampling_skill/results/RUN_ID/TRIAL_ID/artifacts
```

Validates sample count, parameter bounds, numerical results, run outputs, and
audit presence. Prints a JSON report and exits with status 0 only when all
objective checks pass.

### (C) `replay.py`

```bash
repo="$PWD"
(cd /path/to/workspace && PYTHONPATH="$repo" \
  "$repo/.venv/bin/python" "$repo/benchmarks/sampling_skill/replay.py" /tmp/observed.json)
```

Re-executes the workspace’s `workflow.py` while recording the actual sampling
API arguments. Writes the observed seed, sample count, concurrency, parameters,
and dispatcher to the supplied JSON file.

## `test_benchmark.py`

```bash
.venv/bin/python -m unittest benchmarks.sampling_skill.test_benchmark
```

Tests usage accounting, workspace isolation, validation, replay, and paired
runner behavior. Uses a fake provider CLI, so the test suite makes no model
requests.

## `summarize.py`

```bash
.venv/bin/python benchmarks/sampling_skill/summarize.py \
  benchmarks/sampling_skill/results
```

Finds benchmark `results.json` files and groups comparable baseline/skill
trials. Prints a Markdown summary of pass rates and paired token savings.
