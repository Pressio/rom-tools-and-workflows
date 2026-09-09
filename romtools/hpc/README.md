# hpc-tools

Execution layer for running Pressio-based ROM workflows on HPC clusters
from your local environment.

## Overview

Everything goes through the `Dispatcher` class, which defines public methods like:

- `put(local_path, remote_path)`: Copy a local file to the remote host
- `get(remote_path, local_path)`: Copy a remote file to the local host
- `dispatch(cmd, remote_run_directory)`: Executes `cmd` from the remote host's `run_directory`
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
from romtools.hpc.dispatchers import BaseDispatcher, LocalDispatcher

class MyModel:

    def __init__(self, dispatcher: Optional[BaseDispatcher] = None):
        if dispatcher is None:
            dispatcher = LocalDispatcher()
        self.dispatcher = dispatcher
```

> [!TIP]
> We default here to the `LocalDispatcher` so that the workflow can
> function with no remote capability if needed. `LocalDispatcher` overloads
> all public methods of the `Dispatcher` without actually sending any
> work to a remote host. For example, both `put()` and `get()` become a local `cp`.

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
    self.dispatcher.dispatch(cmd, with_slurm = False)
```

> [!NOTE]
> The `with_slurm` flag determines whether or not your command will be
> submitted to the scheduler. For simple validation commands, you will
> probably want to set this to `False`. The default is `True`.

### Step 3: Define `run_model()`

Now we update `run_model()` to use the dispatcher as well. There are two primary
ways to do this:

1. **SLURM**

The simplest approach is to create a SLURM script locally that executes your model.
Then you just configure the dispatcher with that script (using the `-s`
command--see [Configurating the dispatcher](#configuring-the-dispatcher) below).

Then you `run_model()` method can be as simple as:

```py
    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        self.dispatcher.dispatch()
        return 0
```

This will copy your local SLURM script onto the remote host, submit it, and poll
it until it completes.

The SLURM script is called with `--output` and `--error` flags to split and return job output.

2. **Manual Commands**

The alternative is to have the dispatcher create your SLURM script for you.
This method is implemented by the `ExampleModel` class in the `hpc/example` directory.

In short, you define the command (`cmd`) you wish to execute whenever `run_model()` is
called, and use the dispatcher to wrap it in a SLURM script and submit it:

```py
    def run_model(self, run_directory: str, parameter_sample: dict) -> int
        cmd = "srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 my_app"
        self.dispatcher.dispatch( cmd, run_directory )
        return 0
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
    sampling_dir = "sample_00"
    with RemoteDispatcher(sampling_dir) as dispatcher:
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

> [!WARNING]
> Restart files written through a `RemoteDispatcher` land on the remote host,
> but the drivers read `restart_file` from the local filesystem. To restart a
> remote run, retrieve the restart file first (e.g. with the `collect`
> configuration option or `dispatcher.get()`).

When instantiating the dispatcher, you need to pass it a `sampling_directory`.
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
pass it to your workflow with `-i path/to/your/input.yaml`.

An example configuration YAML can be found in `hpc/config/example.yaml`.

2. **CLI**

You can also set these params via the command-line. For example,
set the `remote_root` (the remote directory where all commands
are executed) by passing `-R /path/to/remote/root` when you
execute your workflow.

3. **Combination**

CLI arguments override any YAML parameters. Therefore, you can combine
the two approaches by using a YAML file for the bulk of configuration,
and passing CLI args to vary the configuration from run to run.

For example you might want to change which files you collect from
the remote host, but keep the rest of the configuration the same.

You could run:

```sh
python my_workflow.py -i path/to/input.yaml -o '*.log'
```

### Core configuration arguments

As mentioned, the full configuration `SCHEMA` is defined in
`hpc/configuration.py`. You can also pass `-h` to your
workflow to see the full schema.

> [!NOTE]
> You do not need to specify every argument. Check out the
> the constructor of the `Configuration` class (in
> `hpc/configuration.py`) to see the default values for each
> parameter.

#### `ssh`

These arguments establish your connection with the remote host. They are:

- `remote` (`-r`): The name of the remote host you are connecting to
- `user` (`-u`): The username to use for the connection
- `port` (`-p`): The port to use for the connection

In the YAML, group these all under `ssh`:

```yaml
ssh:
    remote: my_cluster
    user: my_name
    port: 22
```

#### `workflow`

These workflow arguments define file management with the dispatcher.

- `remote_root` (`-R`): Directory on the remote host where commands are executed, absolute or relative to the home directory.
- `collect` (`-o`): Comma-separated list of files, directories, or glob patterns to retrieve from the remote run directory. If omitted, nothing is retrieved
- `upload` (`-U`): Comma-separated list of files, directories, or glob patterns to upload to the remote run directory. If omitted, nothing is uploaded

Two more workflow arguments describe the remote Python used by `call()`:

- `python_setup` (`-e`): Shell commands that set up the remote environment before invoking Python, such as loading modules or activating a virtual environment
- `python_command` (`-c`): Command that invokes the remote Python with the necessary libraries installed (default: `python3`)

In the YAML, these are grouped under `workflow`:

```yaml
workflow:
    remote_root: my_sampling_directory
    collect: "*.log, passed.txt"
    upload: "input.yaml, mesh/"
    python_setup: "module load python/3.11"
    python_command: python3
```

#### `slurm`

These arguments are used to schedule jobs with the dispatcher.

The primary argument simply points to an existing SLURM script:

- `script` (`-s`): Path to a local SLURM script. This will be uploaded to the remote host and submitted on calls to `dispatch()`.

All other arguments are used when you use the dispatcher to
create the SLURM script for you based on some command.

- `account` (`-a`): The account WCID to charge for the job
- `job_name` (`-j`)
- `num_nodes` (`-n`)
- `tasks_per_node` (`-t`)
- `wall_time` (`-w`)
- `partition` (`-q`)

The final SLURM argument specifies how often the dispatcher should
poll the submitted job:

- `poll_interval` (`-I`): Seconds between `squeue` polls
- `timeout` (`-T`): Seconds to keep retrying the `sacct` query for a finished job's exit code before giving up

In YAML, all of these arguments are grouped under `slurm`:

```yaml
slurm:
    script: /path/to/script
    poll_interval: 10
```

#### `output`

The output group only contains one argument that toggles debug logging
for the dispatcher:

- `debug` (`-d`)

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
python romtools/hpc/example/workflow.py -r <remote-host> -u <username> -a <account/wcid>
```

See all available arguments with:

```sh
python romtools/hpc/example/workflow.py -h
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
