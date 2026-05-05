# hpc-tools

Execution layer for running Pressio-based ROM workflows on HPC clusters
from your local environment.

## Getting started

### Installation

Install the romtools dependencies with

```bash
pip install -e .
```

in the project root.

## Running

Run the example workflow with:

```sh
python romtools/hpc/example/workflow.py -r <remote-host> -u <username> -a <account/wcid>
```

See all available arguments with:

```sh
python romtools/hpc/example/workflow.py -h
```

### Using a YAML Input File

Instead of supplying every argument on the command line, you can use the `-i` (or `--input`) arg to specify a YAML 
file with your config. There is an `example.yaml` file to reference how it can be used. Arguments can either be
flat-mapped or set in a hierarchical structure (refer to the example for this hierarchy).

You do not need to supply every possible argument, if you omit any out of the yaml and cli it will default to the app's
default (in `DispatcherConfig` ctor).

If you want to override any arguments specified in your yaml file, you do not have to edit the file, you can simply supply those arguments
on the command line (these take precedence).


## Dispatcher and Connection Classes

The only class you need to interact with is `Dispatcher`. It manages all remote operations for you, including establishing a persistent SSH connection to the cluster (using the `Connection` class internally). You never need to use `Connection` directly.

### How it works

- The SSH connection is established automatically using the host, user, and port you provide via command-line arguments or in your YAML config file.
- The `Dispatcher` creates a local output directory (default: `hpctools/`) and a matching directory on the remote cluster under the specified `remote_root`.
- All job submission, monitoring, and result collection is handled for you.

### Using the Dispatcher

**Constructor arguments:**

The `Dispatcher` takes in these arguments:

- `logger`: An instance of the Logger class for logging
- `sampling_directory`: An optional string for your local + remote output directory (defaults to `hpctools/`)

**dispatch(cmd, run_directory=None):**

- `cmd`: The command to run as a SLURM job (e.g., the application or script you want to execute)
- `run_directory`: (optional) Subdirectory name for this run (created under both local and remote roots)

The method submits your command as a SLURM job, waits for it to finish, and automatically collects the results back to your local output directory.

### Directory structure

The local output directory (default: `hpctools/`, specified in `Dispatcher` ctor) mirrors the structure of the remote directory under `remote_root`. Each run gets its own subdirectory, so your results are organized and easy to find both locally and remotely.

---
For most users, you only need to configure your connection and job parameters, then call `dispatch()` to run your workflow. All SSH and SLURM details are handled for you.
