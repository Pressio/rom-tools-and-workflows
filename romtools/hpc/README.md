# hpc-tools

Execution layer for running Pressio-based ROM workflows on HPC clusters
from your local environment.

## Overview

Everything goes through the `Dispatcher` class, which defines public methods like:

- `put(local_path, remote_path)`: Copy a local file to the remote host
- `get(remote_path, local_path)`: Copy a remote file to the local host
- `run(cmd, run_directory)`: Executes `cmd` directly from `run_directory` on the execution host
- `submit_job(cmd, run_directory)`: Submits `cmd` to SLURM, waits for it, and collects results
- `call(target, *args, run_directory, **kwargs)`: Runs the Python callable named by `target` (as `"module:qualname"`) on the execution host and returns its result
- `path_exists(path)`: Whether `path` exists on the execution host
- `create_empty_dir(dir_name)`: Create `dir_name` (and any missing parents) on the execution host
- `list_dir(path)`: Names of the entries in `path` on the execution host
- `remove(path)`: Delete the file at `path` on the execution host
- `write_text(path, content)`: Write `content` to a text file on the execution host
- `np_savetxt(path, arr, fmt)` / `np_savez(path, **arrays)`: Write numpy data to the execution host
- `get_config(param = None)`: Returns the value of the specified param, or the whole config dict if no argument is given

You'll need to wire the Dispatcher into your model class and your workflow file. Here's how:

## Updating your model

For now, we'll assume the model only has the two fundamental methods:
- `populate_run_directory()`
- `run_model()`

For any other methods your model contains, the following ideas are applicable.

### Step 1: Add a constructor (if it doesn't have one)

You'll need to construct your model with the dispatcher as a member variable.
Add a basic `__init__` method that takes the dispatcher as input:

```py
from typing import Optional
from romtools.hpc.dispatchers import BaseDispatcher, resolve_dispatcher

class MyModel:

    def __init__(self, dispatcher: Optional[BaseDispatcher] = None):
        self.dispatcher = resolve_dispatcher(dispatcher)
```

> [!TIP]
> `resolve_dispatcher()` falls back to a `LocalDispatcher` so that the workflow
> can function with no remote capability if needed. Prefer it over building one
> yourself: the fallback it returns ignores your program's command line.
> `LocalDispatcher` overloads all public methods of the `Dispatcher` without
> actually sending any work to a remote host. For example, both `put()` and
> `get()` become a local `cp`.
> That host may itself be a cluster node, in which case `submit_job()` issues
> `sbatch` where the workflow already runs and the results need no transferring back.

### Step 2: Set up the run directory

Update `populate_run_directory()` so that, instead of copying files locally, you
use `self.dispatcher.put(local_file, remote_path)` to copy necessary files to
the remote host.

> [!NOTE]
> You can pass relative paths to `put()` and `get()`.
> The base directory on the remote host where all commands are executed from
> is called the `remote_root` and is configured at runtime.
> See [Configurating the dispatcher](#configuring-the-dispatcher) below.

If you need to execute any shell commands, e.g. to validate input decks or load
modules, use:

```py
    cmd = "load my_module && my_input_validator -i input_file.yaml"
    result = self.dispatcher.run(cmd)
    if not result.ok:
        raise RuntimeError(result.stderr)
```

> [!NOTE]
> `run()` executes the command directly, so it is the right choice for quick
> work like validation. Anything long or parallel belongs in `submit_job()`, so
> that it lands on compute nodes rather than the login node.

> [!NOTE]
> A command that exits non-zero is reported through the returned `Result`, not
> raised. Every dispatcher behaves this way, so a model can move between them
> unchanged. Check `result.ok` or `result.exit_code` and decide what a failure
> means for your model.

### Step 3: Define `run_model()`

Now we update `run_model()` to use the dispatcher as well. There are two primary
ways to do this:

1. **SLURM**

The simplest approach is to create a SLURM script locally that executes your model.
Then you just configure the dispatcher with that script (using the `--script`
command--see [Configurating the dispatcher](#configuring-the-dispatcher) below).

Then you `run_model()` method can be as simple as:

```py
    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        return self.dispatcher.submit_job().exit_code
```

This will copy your local SLURM script onto the remote host, submit it, and poll
it until it completes.

> [!NOTE]
> `submit_job()` returns a `Result` carrying the job's exit code and its
> captured output. The workflows treat a non-zero `run_model()` return as a
> failed sample, so pass the exit code through rather than returning 0.

Unless your script names its own output files, sbatch is called with `--output`
and `--error` flags so that the two streams come back separately. A script that
sets only `--output` keeps SLURM's own behavior, where stderr is merged into it.

2. **Manual Commands**

The alternative is to have the dispatcher create your SLURM script for you.
This method is implemented by the `ExampleModel` class in the `hpc/example` directory.

In short, you define the command (`cmd`) you wish to execute whenever `run_model()` is
called, and use the dispatcher to wrap it in a SLURM script and submit it:

```py
    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        cmd = "srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 my_app"
        return self.dispatcher.submit_job(cmd, run_directory).exit_code
```

The dispatcher will create a SLURM script that executes this command (configured
at runtime--see [Configurating the dispatcher](#configuring-the-dispatcher)),
submit the script, and then poll the job until it completes.

### Step 4: Update other methods

Any other methods that execute commands or read files can be similarly wired with
the core dispatcher functionality.

## Updating your workflow

After you have updated your model class to use a dispatcher, you must instantiate
the dispatcher in your workflow. The safest way is to scope it, as seen in `hpc/example/workflow.py`:

```py
    campaign_dir = "sample_00"
    with RemoteDispatcher(campaign_dir) as dispatcher:
        model = ExampleModel(dispatcher)
        ###
        romtools.workflows.run_sampling(
            model = model,'
            ###
            dispatcher = dispatcher
        )
```

Note that we pass the Dispatcher to both the model and the sampling workflow:

The following workflows accept a `dispatcher` argument:

- `run_sampling()`
- `run_eki()`, `run_mf_eki()`, `mf_eki_with_auto_rom()`
- `run_vi()`, `run_mf_vi()`, `mf_vi_with_auto_rom()`

Every one of them defaults to a `LocalDispatcher` when you pass nothing, so
existing workflows keep running unchanged.

> [!NOTE]
> In the multifidelity workflows, only the high-fidelity (FOM) evaluations are
> dispatched. Surrogate (ROM) models are fit and evaluated in-process, so their
> run directories always stay on the local machine. Because the FOM and ROM run
> directories are built from the same working directory, a multifidelity run
> with a `RemoteDispatcher` requires a *relative* working directory: FOM
> directories are resolved against the remote root, and the matching ROM
> directories against your local working directory. Passing an absolute path
> raises an error.

> [!WARNING]
> `evaluation_concurrency` greater than 1 is not supported with a
> `RemoteDispatcher`, and the workflows raise an error if you ask for it.
> Concurrent evaluation runs each sample in a separate process, which a remote
> connection is not set up to share. Use `evaluation_concurrency = 1` for remote
> runs and let SLURM provide the parallelism, or keep concurrency with a
> `LocalDispatcher`. Note that `run_mf_vi()` defaults
> `fom_evaluation_concurrency` to 10, so remote MF-VI runs have to set it to 1.

> [!NOTE]
> Concurrency does work with a `LocalDispatcher`, including one running on a
> cluster node. Each worker process holds its own copy of the dispatcher and
> each sample gets its own run directory, so the samples submit, poll, and read
> back their jobs independently. Bear in mind that `evaluation_concurrency = N`
> then puts N jobs in the queue at once, which your site's submission limits
> may cap.

> [!WARNING]
> Restart files written through a `RemoteDispatcher` land on the remote host,
> but the drivers read `restart_file` from the local filesystem. To restart a
> remote run, retrieve the restart file first (e.g. with the `collect`
> configuration option or `dispatcher.get()`).

When instantiating the dispatcher, you need to pass it a `campaign_directory`.
This will be created both locally and remotely as a subdirectory
of your local current directory and your remote `remote_root`.

This mirrored directory structure simplifies sending files back and forth
between the local and remote hosts.

## Configuring the dispatcher

The model has been updated; the workflow has been defined. All
that remains is to run the workflow and configure the dispatcher:

```sh
python my_workflow.py <config args>
```

> [!TIP]
> Refer to the `SCHEMA` in `hpc/configuration.py` to see the full
> schema of possible configuration arguments.

There are three ways to configure:

1. **YAML**

Define a YAML file that contains all configurable params, and
pass it to your workflow with `-c path/to/your/input.yaml`.

An example configuration YAML can be found in `hpc/config/example.yaml`.

2. **CLI**

You can also set these params via the command-line. For example,
set the `remote_root` (the remote directory where all commands
are executed) by passing `--remote_root /path/to/remote/root` when you
execute your workflow.

3. **Combination**

CLI arguments override any YAML parameters. Therefore, you can combine
the two approaches by using a YAML file for the bulk of configuration,
and passing CLI args to vary the configuration from run to run.

For example you might want to change which files you collect from
the remote host, but keep the rest of the configuration the same.

You could run:

```sh
python my_workflow.py -c path/to/input.yaml --collect '*.log'
```

### Core configuration arguments

As mentioned, the full configuration `SCHEMA` is defined in
`hpc/configuration.py`. To see it as a list of arguments, run:

```sh
python -m romtools.hpc
```

Every argument is available as a long option named after it, such as
`--num_nodes`. One argument also has a short alias:

| Short | Long | Meaning |
| --- | --- | --- |
| `-c` | `--config` | Path to the YAML configuration file |

> [!NOTE]
> Your workflow's own command line is what the dispatcher parses, so any
> switch the schema claims is one your workflow can no longer use for itself.
> That is why `-c` is the only single-letter switch claimed; `-h` is never
> claimed either, so your workflow keeps its own `--help`.

> [!NOTE]
> To keep the dispatcher away from your command line entirely, construct it
> with an explicit argument list: `LocalDispatcher(argv=[])` reads no switches
> at all, so it takes the schema defaults. Since `-c` is among the switches it
> no longer sees, no YAML is loaded either; pass the settings you need to the
> constructor. A workflow you call without a dispatcher at all gets exactly
> that fallback.

> [!NOTE]
> You do not need to specify every argument. Check out the
> the constructor of the `Configuration` class (in
> `hpc/configuration.py`) to see the default values for each
> parameter.

#### `ssh`

These arguments establish your connection with the remote host. They are:

- `remote`: The name of the remote host you are connecting to
- `user`: The username to use for the connection
- `port`: The port to use for the connection

In the YAML, group these all under `ssh`:

```yaml
ssh:
    remote: my_cluster
    user: my_name
    port: 22
```

#### `workflow`

These workflow arguments define file management with the dispatcher.

- `remote_root`: Directory on the remote host where commands are executed, absolute or relative to the home directory.
- `collect`: Comma-separated list of files, directories, or glob patterns to retrieve from the remote run directory. If omitted, nothing is retrieved
- `upload`: Comma-separated list of files, directories, or glob patterns to place in the run directory before work starts. A `RemoteDispatcher` sends them to the remote host; a `LocalDispatcher` copies them from the current directory. If omitted, nothing is uploaded

Two more workflow arguments describe the remote Python used by `call()`:

- `python_setup`: Shell commands that set up the remote environment before invoking Python, such as loading modules or activating a virtual environment
- `python_command`: Command that invokes the remote Python with the necessary libraries installed (default: `python3`)

In the YAML, these are grouped under `workflow`:

```yaml
workflow:
    remote_root: my_campaigns
    collect: "*.log, passed.txt"
    upload: "input.yaml, mesh/"
    python_setup: "module load python/3.11"
    python_command: python3
```

#### `slurm`

These arguments are used to schedule jobs with the dispatcher.

The primary argument simply points to an existing SLURM script:

- `script`: Path to a local SLURM script. This will be uploaded to the remote host and submitted on calls to `submit_job()`.

All other arguments are used when you use the dispatcher to
create the SLURM script for you based on some command.

- `account`: The account WCID to charge for the job
- `job_name`
- `num_nodes`
- `tasks_per_node`
- `wall_time`
- `partition`

The final SLURM argument specifies how often the dispatcher should
poll the submitted job:

- `poll_interval`: Seconds between `squeue` polls
- `timeout`: Seconds to keep retrying the `sacct` query for a finished job's exit code before giving up

In YAML, all of these arguments are grouped under `slurm`:

```yaml
slurm:
    script: /path/to/script
    poll_interval: 10
```

#### `output`

The output group only contains one argument that toggles debug logging
for the dispatcher:

- `debug`

In the YAML, it goes under the `output` group:

```yaml
output:
    debug: True
```

#### `user_defined`

You can define your own paramters in the YAML by
putting them in the `user_defined` group:

```yaml
user_defined:
    my_param: 4
```

You can then use them in your model class by calling:

```py
    self.my_param = self.dispatcher.get_config("my_param")
```

---

## Examples

Run the example workflow with:

```sh
python romtools/hpc/example/workflow.py --remote <remote-host> --user <username> --account <account/wcid>
```

See all available arguments with:

```sh
python -m romtools.hpc
```

## Dispatcher and Connection Classes

The only class you need to interact with is `Dispatcher`. It manages all remote operations for you, including establishing a persistent SSH connection to the cluster (using the `Connection` class internally). You never need to use `Connection` directly.

### How it works

- The SSH connection is established automatically using the host, user, and port you provide via command-line arguments or in your YAML config file.
- The `Dispatcher` creates a local output directory (default: `hpctools/`) and a matching directory on the remote cluster under the specified `remote_root`.
- All job submission, monitoring, and result collection is handled for you.

### Directory structure

The local output directory  mirrors the structure of the remote
directory under `remote_root`. Each run gets its own subdirectory,
so your results are organized and easy to find both locally and remotely.
