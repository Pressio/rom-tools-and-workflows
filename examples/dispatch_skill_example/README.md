# Dispatch Skill Example

This directory is a worked example of the `dispatch` skill
(`.claude/skills/dispatch/SKILL.md`), which generates and/or runs
`romtools` sampling workflows — locally, or remotely against a real HPC
cluster.

## What the skill can do

- **Three ways to start it.** Point it at an existing workflow file and it
  just runs that file. Describe a sampling workflow in plain language and
  it drafts a model, a parameter space, and a workflow script from scratch.
  Point it at a model you already have — as in this example — and it
  drafts only the parameter space and workflow script, reusing your model
  as-is instead of rewriting its logic.
- **Local runs** use `LocalDispatcher` (or no dispatcher at all) and need
  no config file.
- **Remote runs** use `RemoteDispatcher` against a real HPC cluster over
  SSH, submitting an actual SLURM job. These need a config YAML you
  already have — the skill won't generate or edit one for you — and it
  asks for an explicit, separate go-ahead before submitting, since it
  consumes real cluster account allocation.
- **Confirms before writing.** Any drafted model, parameter space, or
  workflow script is shown to you before anything is written to disk.
- **Reports plainly.** A successful run's exit code and stdout are
  reported back; a failed run's exit code and stderr/stdout are surfaced
  verbatim, with no automatic retry or "fix" attempted on your behalf.
- **Leaves an audit trail.** Every run writes `AUDIT.md` into the sampling
  output directory: when it ran, local or remote (and against which
  host/config for remote), whether the model was drafted or reused,
  the parameter space and the reasoning behind any non-obvious choices,
  sample count/concurrency/seed, confirmation that drafted files and
  (for remote) the SLURM go-ahead were approved before running, and the
  exit code and outcome. It's written every time, even for a plain rerun
  of an existing workflow file.

## What it won't do

- Generate or edit the config YAML remote runs need
- Edit an existing workflow file
- Poll the status of a job submitted earlier or elsewhere
- Fetch or interpret results after a run finishes
- Retry a failed run

## Claude Code vs. other agents

`SKILL.md` is a plain markdown file with a YAML frontmatter header — it
carries no Claude Code-specific mechanics itself. Claude Code (and other
tools built on the same Agent Skills convention, like the Claude Agent
SDK) auto-discovers it under `.claude/skills/` and lets you invoke it
explicitly with `/dispatch`, or invokes it automatically when a request
matches its description. An agent without that discovery mechanism can
still use it — just point the agent at
`.claude/skills/dispatch/SKILL.md` (or paste its contents in) and ask it
to follow those instructions.

## This example

The prompt below is the exact prompt given to the skill for the
parameter-space/workflow-drafting step of this example — it presupposes
`AdrModel.py` already exists. `AdrModel.py` itself (a wrapper around a 1D
advection-diffusion solver) was created with this skill in an earlier,
unrecorded invocation, using the `adr_1d` demo notebook as a reference; its
prompt isn't preserved here. Given `AdrModel.py`, this prompt reused it
as-is and drafted `AdrParameterSpace.py` and `adr_workflow.py` around it.

> /dispatch Use the existing model Adr1dModel in
> examples/dispatch_skill_example/AdrModel.py - don't rewrite
> populate_run_directory/run_model/compute_qoi. Build a parameter space for
> its two parameters: c uniform in [0.5, 5.0], nu log-uniform in [1e-3,
> 1e-1] (it spans two orders of magnitude, so sample it in log space). Run
> locally with 16 samples, evaluation_concurrency=8, output directory
> adr_sampling.

To reproduce the parameter-space/workflow step: paste the prompt above into
a `/dispatch` invocation from a checkout where `AdrModel.py` already exists
(as it does in this repo), or run the workflow directly with
`python3 adr_workflow.py` from this directory.
