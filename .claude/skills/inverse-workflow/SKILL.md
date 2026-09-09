---
name: inverse-workflow
description: Generate and/or run an inverse workflow for parameter calibration or posterior inference, selecting or suggesting an algorithm from the current repository. Use this whenever the user asks to create, run, kick off, or launch an inverse workflow — whether a workflow python file already exists on disk, or the user describes one in plain language that doesn't exist yet — or asks which inverse algorithm to use. Remote runs require an existing config YAML and a compatible model; this skill does not generate the config.
---

# Inverse Workflow

Generates and/or runs a workflow file, locally or through a compatible
remote model, and reports whether the run completed successfully. Selects
or suggests an inverse algorithm by reading what exists in the repository.

## Preconditions

- If the user references an existing workflow file, skip straight to
  "Running a workflow" below.
- If no workflow file exists yet and the user describes what they want,
  follow "Generating a workflow" below first, then run it, unless they
  asked only for file generation.
- If the user supplies an existing model instead of describing one from
  scratch, reuse it as-is — only the parameter space and workflow script
  need to be drafted. See step 2 of "Generating a workflow" below.
- If the user asks only which algorithm to use, follow the input and
  algorithm-selection parts of "Generating a workflow" and report the
  recommendation without drafting files or launching evaluations.
- **Local**: works with a local `QoiModel`; no dispatcher or config file
  is needed by the inverse driver.
- **Remote**: requires an existing config YAML and a model that dispatches
  its evaluations and makes the resulting QoIs available to the inverse
  driver. See `romtools/hpc/config/example.yaml` and `romtools/hpc/README.md`.
  The inverse drivers do not accept the sampling driver's `dispatcher=`
  argument. If the config or a compatible model is missing, explain what
  is needed and stop — generating the config or adding remote integration
  is out of scope for this skill.

## Generating a workflow

1. From the user's description, work out:
   - What `populate_run_directory()` and `run_model()` should actually do —
     what files get written, what gets computed. If the user points to an
     existing model instead, use it as-is: confirm it implements
     `populate_run_directory(run_directory, parameter_sample)`,
     `run_model(run_directory, parameter_sample) -> int`, and
     `compute_qoi(run_directory, parameter_sample) -> np.ndarray`, and
     skip drafting model code in step 2 below.
   - The parameter names and their space (e.g. bounds for a uniform
     initial ensemble, or Gaussian prior moments for variational inference).
   - The observations, their ordering and units, and the observation
     covariance if required by the algorithm. For synthetic observations,
     get the reference parameters and any noise specification.
   - Whether the goal is best-fit calibration or posterior uncertainty,
     and whether the user has chosen an algorithm. Otherwise, establish
     model cost, parameter dimension, and whether a cheaper surrogate is
     available so the repository can guide the choice.
   - Ensemble or sample count, iteration limit, evaluation concurrency,
     random seed, and an output directory name. Iterations are not model
     evaluations: initial samples, trial steps, and ROM training add cost.
   - Local or remote. For remote, get the path to the user's existing
     config and inspect how their model dispatches evaluations and reads
     results, including whether it works with the requested concurrency.

   If any required input is ambiguous or unstated, ask rather than guessing.
   An unspecified algorithm is a request to select or suggest one in step 2,
   not a reason to ask the user to choose it themselves.

2. Draft the necessary files, following the matching inverse driver and
   tests. Skip the model file entirely if the user supplied an existing
   model — import it into the workflow script as-is instead of drafting one.

   - **Algorithm** — read `romtools/workflows/inverse/__init__.py`, the
     driver modules in that directory, and the matching tests in
     `tests/romtools/workflows/inverse/`. Check current signatures,
     supported inputs, optimizer configs, and return values; use examples
     and docs for additional context. Preserve a user's explicit choice
     if compatible. Otherwise select or suggest the suitable method and
     explain why, pointing to the source and test used:
     - EGO (`run_ego`, `run_batch_ego`) for best-fit observation matching
       when black-box evaluations are expensive and global exploration
       is useful. Batch EGO proposes concurrent candidates; serial EGO's
       concurrency only covers its initial samples. Check the built-in
       misfit objective; zero observations need absolute error.
     - EKI (`run_eki`) for derivative-free calibration when ensemble
       evaluations are affordable. Requires observation covariance;
       ensemble spread is not a Bayesian posterior estimate.
     - VI (`run_vi`) for approximate posterior inference with a supported
       Gaussian prior. Its diagonal or fixed-correlation Gaussian family
       cannot generally represent multiple posterior modes. Inspect
       `vi_optimization_methods.py` for gradient or Newton configs; these
       score-function methods do not require forward-model derivatives.
     - Multifidelity EKI or VI for expensive full models with a useful
       cheaper surrogate. Verify the builder interface and training cost;
       inspect the `*_with_auto_rom` wrappers for supported ROM types.
     This is a starting point, not a fixed catalog. Include other suitable
     implementations found in the checkout, but verify imports and tests
     before recommending them; a source file alone is not a public API.
   - **Local** — parameter space and workflow script mirror
     `examples/inverse_workflow_skill_example/DecayParameterSpace.py` and
     `decay_workflow.py`; if drafting a model too, mirror `DecayModel.py`:
     plain local file I/O and NumPy QoIs. The workflow script takes no
     `dispatcher=` argument and does no CLI parsing. Keep the driver call
     under `if __name__ == "__main__":`, with an absolute output path.
   - **Remote** — reuse the user's verified remote model and its existing
     config-loading pattern. Do not copy `run_sampling` dispatcher wiring
     into an inverse driver. Confirm that files and QoIs are accessible
     where the inverse algorithm runs before drafting the driver call.
   - **Parameter space class** — follow
     `romtools/workflows/parameter_spaces.py`: implement `get_names()`,
     `get_dimensionality()`, and `generate_samples(number_of_samples,
     seed=None)` for a custom space. For VI, use the supported Gaussian
     parameter-space classes and keep the prior separate from the initial
     variational state. Match observations and covariance to model QoIs,
     and use the selected driver's actual bounds and optimizer arguments.

3. Show the drafted files and algorithm choice to the user and get explicit
   confirmation before writing anything to disk. The parameter space,
   observation assumptions, and model logic (when drafting one) come from
   a plain-language description — confirm they match before they become
   code that runs.

4. Once confirmed, write the files, then continue to "Running a workflow"
   below, unless the user asked only for file generation.

## Running a workflow

1. Determine whether the workflow runs locally or through a remote model.
   If unsure, check the model class the workflow constructs. Confirm the
   existing script's driver, inputs, and output directory; do not change
   its algorithm or settings as part of a plain run request.

2. **Remote runs only**: before executing anything, explicitly tell the
   user this run will open an SSH connection to the host in their config
   and submit real SLURM jobs that consume cluster account allocation,
   and get explicit go-ahead. This is separate from, and in addition to,
   the file-write confirmation in "Generating a workflow" — do not run
   automatically just because file generation was approved.

3. Run the workflow:

   ```bash
   python <workflow_file>              # local
   python <workflow_file> -i <config>  # remote, if its model uses this CLI
   ```

   Local execution needs no config file or CLI arguments. For remote
   execution, use the existing model's verified launch command.

   Remote runs can block until submitted jobs reach a terminal state,
   so a quiet terminal isn't a hang. Don't kill the process to "unblock"
   it: interrupting a dispatcher can cancel the live remote job.

4. Check the exit code.
   - Exit code `0` → treat as completed successfully. Report back any output
     printed to stdout, including whether a tolerance or iteration limit
     ended the run. Execution completion does not establish convergence
     or posterior accuracy. The inverse helper currently does not enforce
     `run_model` return codes, so surface any reported model failures or
     non-finite outputs even when the process exits successfully.
   - Non-zero exit code → do NOT tell the user it succeeded. Surface the
     stderr/stdout contents verbatim so the failure reason is visible, and
     stop — do not retry automatically or modify the workflow to "fix" the
     error on the user's behalf. Still write the audit record below.

5. Write an audit record to `<inverse directory>/AUDIT.md` (overwrite if
   one already exists from a prior run of the same directory). This is
   the durable "how and why" for the run — write it every time, whether
   or not this session did any generation. The agent writes this record
   after execution; it does not belong in the generated workflow script.
   It must cover:
   - When the run happened, and whether it ran locally or remotely
     (and if remote, the host and config file used).
   - Whether the model was drafted or reused from an existing file (and
     which file), and the parameter space's names, bounds, and
     distribution, together with the reasoning behind non-obvious choices.
   - The observations and covariance used, and the prior if applicable;
     distinguish supplied data from synthetic data and record any reference
     parameters and noise assumptions.
   - The algorithm and why it was selected, with the source and test used.
     If reasoning isn't in hand — e.g. this session only ran an existing
     workflow — say so rather than inventing it.
   - Ensemble or sample count, iteration limit, evaluation concurrency,
     random seed, and any non-default algorithm settings.
   - Confirmation that drafted files were shown to and confirmed by the
     user before being written (or that an existing workflow file was
     used as-is), and, for remote runs, that the SSH/SLURM go-ahead from
     step 2 above was given.
   - The exact command run, its exit code, and a one-line outcome summary,
     including the stopping condition when reported by the driver.

6. Report back concisely: workflow file used, exit code, and (on success)
   any output the tool printed.

## Explicitly out of scope for this skill

- Generating or editing the config YAML for remote runs
- Adding remote integration to an existing model
- Editing existing workflow files
- Polling job status after a separate/earlier submission
- Fetching or interpreting results after the fact
- Retrying failed runs

These will be handled by follow-on work as needed.
