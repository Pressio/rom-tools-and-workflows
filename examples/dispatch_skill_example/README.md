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

`prompt.txt` is the exact prompt given to the skill for this example. It
reused the already-existing `AdrModel.py` (a wrapper around a 1D
advection-diffusion solver, created with this skill using the demo as an example) as-is, and drafted `AdrParameterSpace.py` and
`adr_workflow.py` around it.

To reproduce: paste `prompt.txt` into a `/dispatch` invocation, or run the
workflow directly with `python3 adr_workflow.py` from this directory.
