Remote Execution
================

The remote execution layer runs Pressio-based ROM workflows on HPC clusters
from your local environment. Everything goes through the ``Dispatcher`` class,
which manages the SSH connection, file transfers, job submission, monitoring,
and result collection on your behalf.

The Dispatcher
--------------

The ``Dispatcher`` exposes a small public interface:

- ``put(local_path, remote_path)``: Copy a local file to the remote host.
- ``get(remote_path, local_path)``: Copy a remote file to the local host.
- ``run(cmd, run_directory)``: Execute ``cmd`` directly from ``run_directory``
  on the execution host.
- ``submit_job(cmd, run_directory)``: Submit ``cmd`` to SLURM, wait for it, and
  collect the results.
- ``call(target, *args, run_directory, **kwargs)``: Run the Python callable
  named by ``target`` (as ``"module:qualname"``) on the execution host and
  return its result.
- ``path_exists(path)``: Whether ``path`` exists on the execution host.
- ``create_empty_dir(dir_name)``: Create ``dir_name``, and any missing parents,
  on the execution host.
- ``list_dir(path)``: Names of the entries in ``path`` on the execution host.
- ``remove(path)``: Delete the file at ``path`` on the execution host.
- ``write_text(path, content)``: Write ``content`` to a text file on the
  execution host.
- ``np_savetxt(path, arr, fmt)`` and ``np_savez(path, **arrays)``: Write numpy
  data to the execution host.
- ``get_config(param=None)``: Return the value of the specified param, or the
  whole config dict if no argument is given.

You wire the dispatcher into your model class and your workflow file. The
``Dispatcher`` is the only class you need to interact with; it uses an internal
``Connection`` class to maintain a persistent SSH connection that you never
touch directly.

.. note::
   A ``LocalDispatcher`` overloads all public methods of the ``Dispatcher``
   without sending any work to a remote host. This lets a workflow run with no
   remote capability when needed; for example, both ``put()`` and ``get()``
   become a local ``cp``.

   That host may itself be a cluster node, in which case ``submit_job()``
   issues ``sbatch`` where the workflow already runs and the results need no
   transferring back. Both dispatchers read the same configuration, so the
   SLURM settings below describe the job either way.

Updating your model
-------------------

Assume your model has the two fundamental methods ``populate_run_directory()``
and ``run_model()``. The same ideas apply to any other methods your model
contains.

Step 1: Add a constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct your model with the dispatcher as a member variable:

.. code-block:: python

   from typing import Optional
   from romtools.hpc.dispatchers import BaseDispatcher, resolve_dispatcher

   class MyModel:

      def __init__(self, dispatcher: Optional[BaseDispatcher] = None):
         self.dispatcher = resolve_dispatcher(dispatcher)

.. tip::
   ``resolve_dispatcher()`` falls back to a ``LocalDispatcher``, so the model
   works with no remote capability if needed. Prefer it over building one
   yourself: the fallback it returns ignores your program's command line.

Step 2: Set up the run directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update ``populate_run_directory()`` so that, instead of copying files locally,
you use ``self.dispatcher.put(local_file, remote_path)`` to copy the necessary
files to the remote host.

.. note::
   You can pass relative paths to ``put()`` and ``get()``. The base directory
   on the remote host where all commands are executed is called the
   ``remote_root`` and is configured at runtime (see
   `Configuring the dispatcher`_).

To run shell commands, e.g. to validate input decks or load modules:

.. code-block:: python

   cmd = "load my_module && my_input_validator -i input_file.yaml"
   result = self.dispatcher.run(cmd)
   if not result.ok:
       raise RuntimeError(result.stderr)

.. note::
   ``run()`` executes the command directly, so it is the right choice for quick
   work such as validation. Anything long or parallel belongs in ``submit_job()``,
   so that it lands on compute nodes rather than the login node.

.. note::
   A command that exits non-zero is reported through the returned ``Result``,
   not raised. Every dispatcher behaves this way, so a model can move between
   them unchanged. Check ``result.ok`` or ``result.exit_code`` and decide what
   a failure means for your model.

Step 3: Define ``run_model()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two primary ways to run the model through the dispatcher.

**SLURM script.** Create a SLURM script locally that executes your model and
configure the dispatcher with that script (using ``--script``, see
`Configuring the dispatcher`_). Then ``run_model()`` can be as simple as:

.. code-block:: python

   def run_model(self, run_directory: str, parameter_sample: dict) -> int:
       return self.dispatcher.submit_job().exit_code

This copies your local SLURM script onto the remote host, submits it, and polls
it until it completes.

.. note::
   ``submit_job()`` returns a ``Result`` carrying the job's exit code and its
   captured output. The workflows treat a non-zero ``run_model()`` return as a
   failed sample, so pass the exit code through rather than returning 0.

**Manual commands.** Alternatively, have the dispatcher create the SLURM script
for you. Define the command you wish to execute and let the dispatcher wrap it
in a SLURM script and submit it:

.. code-block:: python

   def run_model(self, run_directory: str, parameter_sample: dict) -> int:
       cmd = "srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 my_app"
       return self.dispatcher.submit_job(cmd, run_directory).exit_code

The dispatcher creates a SLURM script that executes this command (configured at
runtime), submits it, and polls the job until it completes.

Step 4: Update other methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any other methods that execute commands or read files can be wired with the
same core dispatcher functionality.

Updating your workflow
----------------------

After updating your model class, instantiate the dispatcher in your workflow.
The safest approach is to scope it with a context manager:

.. code-block:: python

   campaign_dir = "sample_00"
   with RemoteDispatcher(campaign_dir) as dispatcher:
       model = ExampleModel(dispatcher)
       romtools.workflows.run_sampling(
           model=model,
           dispatcher=dispatcher,
       )

Note that the dispatcher is passed to both the model and the sampling workflow.
The ``campaign_directory`` you pass to the dispatcher is created both locally
and remotely, as a subdirectory of your local current directory and your remote
``remote_root``. This mirrored directory structure simplifies sending files
back and forth between hosts, and gives each run its own subdirectory.

Supported workflows
-------------------

Each of these workflows accepts a ``dispatcher`` argument:

- ``run_sampling()``
- ``run_eki()``, ``run_mf_eki()``, ``mf_eki_with_auto_rom()``
- ``run_vi()``, ``run_mf_vi()``, ``mf_vi_with_auto_rom()``

Every one of them falls back to a ``LocalDispatcher`` when you pass nothing, so
existing workflows and models keep running unchanged. That fallback is built
with ``argv=[]``: a workflow you never handed a dispatcher keeps its own
command line, and the dispatcher takes the schema defaults.

Inverse workflows
~~~~~~~~~~~~~~~~~

Inverse workflows evaluate the model many times per iteration, under
``<work_directory>/iteration_<k>/run_*``. Pass the dispatcher to the model and
to the driver just as you would for sampling:

.. code-block:: python

   with RemoteDispatcher("eki_00") as dispatcher:
       model = MyModel(dispatcher)
       parameter_samples, qois = romtools.workflows.run_eki(
           model=model,
           parameter_space=my_parameter_space,
           observations=obs,
           observations_covariance=obs_cov,
           absolute_eki_directory="eki_00",
           dispatcher=dispatcher,
       )

The driver uses the dispatcher to create each run directory and to write its
restart, history, and per-iteration statistics files. Your model still decides
how a single evaluation runs, through the ``self.dispatcher`` calls you added
above.

.. note::
   The working directory argument (``absolute_eki_directory`` or
   ``absolute_vi_directory``) must be an absolute path for local runs. When you
   pass a ``RemoteDispatcher``, a relative path is also accepted and is
   resolved against ``remote_root``.

.. note::
   In the multifidelity workflows (``run_mf_eki()``, ``run_mf_vi()``, and their
   ``*_with_auto_rom()`` wrappers), only the high-fidelity (FOM) evaluations
   are dispatched. Surrogate (ROM) models are fit and evaluated in-process, so
   their run directories always stay on the local machine.

.. warning::
   ``evaluation_concurrency`` greater than 1 is not supported with a
   ``RemoteDispatcher``. Concurrent evaluation runs each sample in a separate
   process, which a remote connection is not set up to share. Use
   ``evaluation_concurrency=1`` for remote runs and let SLURM provide the
   parallelism, or keep concurrency with a ``LocalDispatcher``.

.. note::
   Concurrency does work with a ``LocalDispatcher``, including one running on
   a cluster node. Each worker process holds its own copy of the dispatcher
   and each sample gets its own run directory, so the samples submit, poll,
   and read back their jobs independently. Bear in mind that
   ``evaluation_concurrency=N`` then puts N jobs in the queue at once, which
   your site's submission limits may cap.

.. warning::
   Restart files written through a ``RemoteDispatcher`` land on the remote host,
   but the drivers read ``restart_file`` from the local filesystem. To restart a
   remote run, retrieve the restart file first, either with the ``collect``
   configuration option or with ``dispatcher.get()``.

Configuring the dispatcher
--------------------------

Once the model is updated and the workflow is defined, run the workflow and
configure the dispatcher:

.. code-block:: bash

   python my_workflow.py <config args>

There are three ways to configure:

1. **YAML.** Define a YAML file with all configurable params and pass it with
   ``-c path/to/your/config.yaml``.
2. **CLI.** Set params on the command line. For example, set the
   ``remote_root`` by passing ``--remote_root /path/to/remote/root``.
3. **Combination.** CLI arguments override YAML parameters, so you can use a
   YAML file for the bulk of configuration and CLI args to vary settings from
   run to run:

   .. code-block:: bash

      python my_workflow.py -c path/to/config.yaml --collect '*.log'

.. tip::
   Run ``python -m romtools.hpc``, or refer to the ``SCHEMA`` in
   ``hpc/configuration.py``, to see the full schema of configuration arguments.
   You do not need to specify every argument; the ``Configuration`` class
   defines a default value for each parameter.

Core configuration arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every argument is available as a long option named after it, such as
``--num_nodes``. One argument also has a short alias:

.. list-table::
   :header-rows: 1

   * - Short
     - Long
     - Meaning
   * - ``-c``
     - ``--config``
     - Path to the YAML configuration file

.. note::
   Your workflow's own command line is what the dispatcher parses, so any
   switch the schema claims is one your workflow can no longer use for itself.
   That is why ``-c`` is the only single-letter switch claimed; ``-h`` is never
   claimed either, so your workflow keeps its own ``--help``.

.. note::
   To keep the dispatcher away from your command line entirely, construct it
   with an explicit argument list: ``LocalDispatcher(argv=[])`` reads no
   switches at all, so it takes the schema defaults. Since ``-c`` is among the
   switches it no longer sees, no YAML is loaded either; pass the settings you
   need to the constructor. A workflow you call without a dispatcher at all
   gets exactly that fallback.

**ssh** — establish the connection with the remote host:

- ``remote``: Name of the remote host.
- ``user``: Username for the connection.
- ``port``: Port for the connection.

.. code-block:: yaml

   ssh:
       remote: my_cluster
       user: my_name
       port: 22

**workflow** — file management with the dispatcher:

- ``remote_root``: Directory on the remote host where commands are
  executed, absolute or relative to the home directory.
- ``collect``: Comma-separated list of files, directories, or glob
  patterns to retrieve from the remote run directory. If omitted, nothing is
  retrieved.
- ``upload``: Comma-separated list of files, directories, or glob
  patterns to upload to the remote run directory. If omitted, nothing is
  uploaded.
- ``python_setup``: Shell commands that set up the remote environment
  before invoking Python, such as loading modules or activating a virtual
  environment. Used by ``call()``.
- ``python_command``: Command that invokes the remote Python with the
  necessary libraries installed (default: ``python3``). Used by ``call()``.

.. code-block:: yaml

   workflow:
       remote_root: my_campaigns
       collect: "*.log, passed.txt"
       upload: "input.yaml, mesh/"
       python_setup: "module load python/3.11"
       python_command: python3

**slurm** — schedule jobs with the dispatcher:

- ``script``: Path to a local SLURM script, uploaded to the remote
  host and submitted on calls to ``submit_job()``.
- ``account``: Account WCID to charge for the job.
- ``job_name``
- ``num_nodes``
- ``tasks_per_node``
- ``wall_time``
- ``partition``
- ``poll_interval``: Seconds between ``squeue`` polls.
- ``timeout``: Seconds to keep retrying the ``sacct`` query for a
  finished job's exit code before giving up.

All arguments other than ``script`` are used when the dispatcher creates the
SLURM script for you from a command.

.. code-block:: yaml

   slurm:
       script: /path/to/script
       poll_interval: 10

**output** — toggle debug logging for the dispatcher:

- ``debug``

.. code-block:: yaml

   output:
       debug: True

**user_defined** — define your own parameters:

.. code-block:: yaml

   user_defined:
       my_param: 4

Access them in your model class with:

.. code-block:: python

   self.my_param = self.dispatcher.get_config("my_param")

Example
-------

Run the example workflow with:

.. code-block:: bash

   python romtools/hpc/example/workflow.py --remote <remote-host> --user <username> --account <account/wcid>

See all available arguments with:

.. code-block:: bash

   python -m romtools.hpc
