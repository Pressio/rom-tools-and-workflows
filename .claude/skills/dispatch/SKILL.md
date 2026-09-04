---
name: dispatch
description: Generate and/or run a sampling workflow, locally via LocalDispatcher or remotely via RemoteDispatcher against an HPC cluster. Use this whenever the user asks to create, run, kick off, or launch a sampling workflow — whether a workflow python file already exists on disk, or the user describes one in plain language that doesn't exist yet. Remote runs require the user to already have a config YAML; this skill does not generate one.
---

# Sampling Workflow

Generates and/or runs a workflow file, locally via `LocalDispatcher` or
remotely via `RemoteDispatcher`, and reports whether the run completed
successfully.

## Preconditions

- If the user references an existing workflow file, skip straight to
  "Running a workflow" below.
- If no workflow file exists yet and the user describes what they want,
  follow "Generating a workflow" below first, then run it.
- If the user supplies an existing model instead of describing one from
  scratch, reuse it as-is — only the parameter space and workflow script
  need to be drafted. See step 2 of "Generating a workflow" below.
- **Local**: works with `LocalDispatcher`, or no dispatcher at all (which
  defaults to `LocalDispatcher` — see `romtools/hpc/README.md`'s "Updating
  your model" section). No config file needed.
- **Remote**: works with `RemoteDispatcher`, which requires the user to
  already have a config YAML (see `romtools/hpc/config/example.yaml` and
  `romtools/hpc/README.md`'s "Configuring the dispatcher" section). If they
  don't have one, point them at those references and stop — generating or
  editing that YAML is out of scope for this skill.

## Generating a workflow

1. From the user's description, work out:
   - What `populate_run_directory()` and `run_model()` should actually do —
     what files get written, what gets computed. If the user points to an
     existing model instead, use it as-is: confirm it implements
     `populate_run_directory(run_directory, parameter_sample)` and
     `run_model(run_directory, parameter_sample) -> int` (and, if the
     workflow needs it, `compute_qoi(run_directory, parameter_sample)`),
     and skip drafting model code in step 2 below.
   - The parameter names and their space (e.g. bounds for a uniform
     distribution).
   - Number of samples, evaluation concurrency, and an output directory
     name.
   - Local or remote. For remote, also get the path to the user's existing
     config YAML, and whether they have an existing SLURM script to point
     the dispatcher at (README's "SLURM" approach) or want the dispatcher
     to generate one from a command (README's "Manual Commands" approach).
     This changes the generated workflow (and model, if drafting one)
     code, so don't guess.

   If any of this is ambiguous or unstated, ask rather than guessing — same
   rule as the "ask for the path" precondition above.

2. Draft the necessary files, following the matching example set in
   `romtools/hpc/example/`. Skip the model file entirely if the user
   supplied an existing model — import it into the workflow script as-is
   instead of drafting one.

   - **Local** — parameter space and workflow script mirror
     `ExampleParameterSpace.py` and `workflow_noconn.py`; if drafting a
     model too, mirror `ExampleModelNoConn.py`: model constructor takes an
     optional dispatcher and defaults to `LocalDispatcher`;
     `populate_run_directory` and `run_model` are plain local file I/O; the
     workflow script takes no `dispatcher=` argument and does no CLI
     parsing.
   - **Remote** — parameter space and workflow script mirror
     `workflow.py`; if drafting a model too, mirror `ExampleModel.py`:
     model constructor takes a required dispatcher (no default);
     `populate_run_directory` uses `self.dispatcher.put(...)` for any
     files that need to land on the remote host; `run_model` uses
     `self.dispatcher.dispatch(...)` per whichever approach was chosen in
     step 1 (bare `dispatch()` against a pre-supplied SLURM script, or a
     `cmd` string for the dispatcher to wrap into a generated script). The
     workflow script scopes the dispatcher with
     `with RemoteDispatcher(sampling_dir) as dispatcher:`, constructs the
     model with it, and passes `dispatcher=dispatcher` into
     `run_sampling(...)`. No config-parsing code is needed in the script
     itself — it's read from `sys.argv` at runtime via `-i <config file>`.
     A user-supplied model must accept and use the dispatcher the same way
     to run remotely — flag it to the user if it doesn't.
   - **Parameter space class** (always drafted, both targets): subclasses
     `romtools.workflows.parameter_spaces.ParameterSpace` and implements
     `get_names()`, `get_dimensionality()`, and
     `generate_samples(number_of_samples)`.

3. Show the drafted files to the user and get explicit confirmation before
   writing anything to disk. The parameter space (and model logic, when
   drafting one) is a guess at the user's intent from a plain-language
   description — confirm it matches before it becomes code that runs.

4. Once confirmed, write the files, then continue to "Running a workflow"
   below.

## Running a workflow

1. Determine whether the workflow uses `LocalDispatcher` (or no dispatcher)
   or `RemoteDispatcher`. If unsure, check the model class the workflow
   constructs — `ExampleModelNoConn.py`/`workflow_noconn.py` show the
   local-only shape, `ExampleModel.py`/`workflow.py` show the remote shape.

2. **Remote runs only**: before executing anything, explicitly tell the
   user this run will open an SSH connection to the host in their config
   and submit a real SLURM job that consumes cluster account allocation,
   and get explicit go-ahead. This is separate from, and in addition to,
   the file-write confirmation in "Generating a workflow" — do not run
   automatically just because file generation was approved.

3. Run the workflow:

   ```bash
   python <workflow_file>              # local
   python <workflow_file> -i <config>  # remote
   ```

   Local execution needs no config file or CLI arguments. Remote execution
   needs `-i <path to config YAML>` at minimum, plus any CLI overrides the
   user wants.

   Remote runs block until the SLURM job reaches a terminal state (the
   dispatcher polls `squeue`/`sacct`), so they may take a while — a quiet
   terminal isn't a hang. Don't kill the process to "unblock" it:
   interrupting it cancels the live remote job.

4. Check the exit code.
   - Exit code `0` → treat as completed successfully. Report back any output
     printed to stdout.
   - Non-zero exit code → do NOT tell the user it succeeded. Surface the
     stderr/stdout contents verbatim so the failure reason is visible, and
     stop — do not retry automatically or modify the workflow to "fix" the
     error on the user's behalf.

5. Write an audit record to `<sampling directory>/AUDIT.md` (overwrite if
   one already exists from a prior run of the same directory). This is
   the durable "how and why" for the run — write it every time, whether
   or not this session did any generation. It must cover:
   - When the run happened, and whether it targeted `LocalDispatcher` or
     `RemoteDispatcher` (and if remote, the host and config file used).
   - Whether the model was drafted or reused from an existing file (and
     which file), and the parameter space's names, bounds, and
     distribution, together with the reasoning behind any non-obvious
     choice (e.g. why a bound was picked, or why a parameter was sampled
     log-uniformly instead of uniformly). If that reasoning isn't in
     hand — e.g. this session only ran a workflow file generated earlier
     or by someone else — say so rather than inventing it.
   - Sample count, evaluation concurrency, and random seed.
   - Confirmation that drafted files were shown to and confirmed by the
     user before being written (or that an existing workflow file was
     used as-is, if generation was skipped), and, for remote runs, that
     the SSH/SLURM go-ahead from step 2 above was given.
   - The exact command run, its exit code, and a one-line outcome
     summary.

6. Report back concisely: workflow file used, exit code, and (on success)
   any output the tool printed.

## Explicitly out of scope for this skill

- Generating or editing the config YAML for remote runs
- Editing existing workflow files
- Polling job status after a separate/earlier submission
- Fetching or interpreting results after the fact
- Retrying failed runs

These will be handled by follow-on work once config generation is added to
this skill.
