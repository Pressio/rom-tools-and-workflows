# Inverse Workflow Skill Example

This directory is a worked example of the `inverse-workflow` skill
([SKILL.md](../../.claude/skills/inverse-workflow/SKILL.md)), which generates and/or runs
`romtools` inverse workflows — locally, or remotely through a compatible
model and an existing HPC configuration.

## What the skill can do

- **Three ways to start it.** Point it at an existing workflow file and it
  just runs that file. Describe an inverse workflow in plain language and
  it drafts a model, a parameter space, and a workflow script from scratch.
  Point it at a model you already have — as in this example — and it
  drafts only the parameter space and workflow script, reusing your model
  as-is instead of rewriting its logic. If you haven't chosen an algorithm,
  it reads the repository's drivers and tests to select or suggest one
  for your goal, such as EGO, EKI, VI, or a multifidelity method. You can
  also ask for just a recommendation without generating or running files.
- **Local runs** use `LocalDispatcher` by default and need no config file.
  The workflow can omit `dispatcher=`; the driver creates a local
  dispatcher when none is supplied.
- **Remote runs** use an existing compatible model to dispatch evaluations
  against a real HPC cluster over SSH, submitting actual SLURM jobs. These
  need a config YAML you already have — the skill won't generate or edit
  one for you — and it asks for an explicit, separate go-ahead before
  submitting, since it consumes real cluster account allocation. Inverse
  drivers such as `run_eki` now accept an optional `dispatcher=` argument.
  The model must still implement compatible execution and QoI access;
  passing a dispatcher does not turn local NumPy code into a remote job.
  `RemoteDispatcher` requires `evaluation_concurrency=1` and paths
  relative to its configured remote root.
- **Confirms before writing.** The algorithm choice and any drafted model,
  parameter space, or workflow script are shown to you before anything
  is written to disk.
- **Reports plainly.** A successful run's exit code and stdout are
  reported back; a failed run's exit code and stderr/stdout are surfaced
  verbatim, with no automatic retry or "fix" attempted on your behalf.
  Completion and convergence are reported separately.
- **Leaves an audit trail.** Every skill-driven run writes `AUDIT.md` into
  the inverse output directory: when it ran, local or remote (and against
  which host/config for remote), whether the model was drafted or reused,
  the parameter space and observation assumptions, the algorithm and the
  reasoning behind its choice, sample count/iterations/concurrency/seed,
  confirmation that drafted files and (for remote) the SLURM go-ahead were
  approved before running, and the exit code and outcome. It's written
  every time, even for a plain rerun of an existing workflow file.

## What it won't do

- Generate or edit the config YAML remote runs need
- Add remote integration to an existing model
- Edit an existing workflow file
- Poll the status of a job submitted earlier or elsewhere
- Fetch or interpret results after a run finishes
- Retry a failed run

## Claude Code vs. other agents

`SKILL.md` is a plain markdown file with a YAML frontmatter header — it
carries no Claude Code-specific mechanics itself. The skill lives under
`.claude/skills/` and can be requested with `/inverse-workflow` in Claude
Code. An agent without that discovery mechanism can still use it — just
point the agent at `.claude/skills/inverse-workflow/SKILL.md` (or paste its
contents in) and ask it to follow those instructions.

## This example

The example follows the same approach as the sampling example on
`dispatcher-llm-skill`: start with an existing model, describe the inputs
and run settings, review the drafted files, then run a small workflow
script. The agent handles the explanation and audit record; the script
constructs inputs, calls the driver, and prints results.

The prompt below describes the parameter-space/workflow-drafting step of
this example — it presupposes `DecayModel.py` already exists. `DecayModel.py`
itself evaluates `y(t) = amplitude * exp(-rate * t)` at nine equally spaced
times in `[0, 4]`, writing parameters and a NumPy solution file for each
run. Its constructor accepts an optional dispatcher and defaults to
`LocalDispatcher`, but its methods use local `Path` and NumPy operations
directly. Supplying a dispatcher to the constructor does not change those
local operations. Given `DecayModel.py`, the prompt reuses it as-is and asks for
`DecayParameterSpace.py` and `decay_workflow.py` around it. This is a
reproduction prompt, not a transcript of an earlier invocation.

> /inverse-workflow Use the existing model DecayModel in
> examples/inverse_workflow_skill_example/DecayModel.py - don't rewrite
> populate_run_directory/run_model/compute_qoi. Build a parameter space for
> its two parameters: amplitude uniform in [0.5, 3.0], rate uniform in
> [0.1, 1.5]. Use noiseless synthetic observations at the model's nine times
> with reference amplitude=2.0 and rate=0.7, and observation covariance
> 0.02^2 I. I want best-fit calibration; the model is cheap and no surrogate
> is available. Read the inverse algorithms and tests in the repo and
> select or suggest an appropriate method, explaining the choice. Run
> locally with 16 ensemble members if using an ensemble method, at most
> 20 iterations, evaluation_concurrency=1, random_seed=1, output directory
> an absolute path ending in decay_inverse. Show the selected algorithm
> and drafted files before writing them.

The included workflow uses EKI because the goal is calibration and these
local ensemble evaluations are cheap. Its uniform parameter space supplies
an initial ensemble, and the same bounds constrain EKI updates. It is not
a Bayesian prior. The synthetic reference values and covariance are
example assumptions; the script adds no random noise. The driver call uses
an initial step size of `0.5`, up to three step-size-decrease tries, and an
error norm tolerance of `1e-3`.

To reproduce the parameter-space/workflow step: paste the prompt above into
an `/inverse-workflow` invocation from a checkout where `DecayModel.py`
already exists (as it does in this repo). The direct invocation from the
repository root is shown below, but the checked-in script currently has
the output-path limitation described immediately after it:

```bash
python -m examples.inverse_workflow_skill_example.decay_workflow
```

Use the repository's Python environment with its dependencies installed.
The workflow has no CLI options: its settings are visible in the driver
call. It currently sets `output_dir_name = "decay_inverse"`, while
`run_eki` asks `LocalDispatcher` to enforce an absolute output path.
With normal Python assertion checks enabled, this invocation therefore
stops before model evaluations with
`AssertionError: You must provide an absolute path (received: decay_inverse)`.
The reproduction prompt above requests an absolute path to satisfy the
current API; the checked-in script does not yet do so. Use a fresh output
directory for an independent run once that path is corrected.

The agent following the skill writes `AUDIT.md` after execution,
including when a run fails. A direct Python invocation does not create
that audit record.

After a completed run, the script prints the reference parameters
`[2.0, 0.7]` and the ensemble mean in `[amplitude, rate]` order. The driver
prints its stopping condition; reaching the iteration limit is distinct
from meeting a convergence tolerance. Ensemble spread is not a Bayesian
posterior estimate.

The skill file still says inverse drivers do not accept `dispatcher=`.
That instruction predates the current driver API; the dispatcher behavior
described here follows `run_eki` and the inverse dispatcher tests in this
checkout.
