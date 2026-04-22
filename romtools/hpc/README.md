# hpc-tools

Execution layer for running Pressio-based ROM workflows on HPC clusters
from your local environment.

## Getting started

### Installation

Install the hpc dependencies with

```bash
pip install .[hpc]
```

in the project root.

## Running

Run the basic workflow with:

```sh
python romtools/hpc/basic/workflow.py -r <remote-host> -u <username> -a <account/wcid>
```

See all available arguments with:

```sh
python romtools/hpc/basic/workflow.py -h
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

This package provides two core building blocks for running workflows on an HPC system via SSH and SLURM:

- `Connection`: a lightweight SSH/SCP wrapper that executes remote commands and transfers files using a multiplexed SSH session.
- `Dispatcher`: a higher-level orchestration layer that uses Connection to create remote working directories, generate and submit SLURM jobs, monitor execution, and retrieve results back to the local machine.

### `Connection`: SSH + SCP with connection multiplexing

`Connection` is a minimal abstraction over ssh and scp implemented with subprocess. It starts an SSH ControlMaster session (connection multiplexing) and reuses it for subsequent remote commands and file transfers-avoiding repeated authentication handshakes.

#### Key capabilities

- `run(command)`: execute a command on the remote host and return a Result object (stdout, stderr, exited, ok).
- `put(local, remote=None)`: upload a local file to the remote host via scp.
- `get(remote, local=None)`: download a remote file via scp (creates local directories as needed).
- `local(cmd)`: run a local shell command and return a Result.
- `close()`: terminates the multiplexed SSH session (ssh -O exit).

#### Notes

- Uses `StrictHostKeyChecking=accept-new` by default.
- The control socket is stored in a temporary directory to avoid path-length issues.

### `Dispatcher`: remote workflow orchestration via SLURM

`Dispatcher` is the “main” HPC automation class. It initializes configuration, establishes an SSH connection, prepares local/remote directories, submits SLURM jobs, monitors them, and pulls results back to the local machine.

#### What it does

- Connects to the remote host using `Connection(host, user, port)`.
- Creates a local sampling/output directory (default: hpctools) and a matching directory under `remote_root` on the cluster.
- Generates or uploads a SLURM script:
    - If `config.script` is provided, it uploads that script unchanged.
    - Otherwise it calls `create_slurm_script(...)` to render a script from configuration (job name, nodes, tasks, walltime, partition/account, etc.) and writes it on the remote host.
- Submits the job via sbatch, parses the SLURM job ID, and tracks it in `current_jobs`.
- Monitors the job by polling squeue until the job leaves the queue.
Collects results by tarring the remote run directory, copying the archive locally, unpacking it into the local sampling directory, and cleaning up the archives.

#### Primary public methods

- `dispatch(cmd, run_directory=None)`: submit a command as a SLURM job, wait for completion, then collect results.
- `create_remote_directory(remote_dir, base_dir=False)`: create a directory on the remote host (optionally treated as an absolute/base path).
- `cancel_job(job_id)`: cancel a submitted SLURM job (scancel).
- Context manager support: with `Dispatcher(...) as d: ...` ensures `close()` is called.

__Connection safety__: Most methods are decorated with `@require_connection`, ensuring an active SSH session is available before executing remote operations.
