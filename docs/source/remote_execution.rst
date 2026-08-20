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
- ``dispatch(cmd, remote_run_directory)``: Execute ``cmd`` from the remote
  host's ``run_directory``.
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
   from romtools.hpc.dispatchers import BaseDispatcher, LocalDispatcher

   class MyModel:

      def __init__(self, dispatcher: Optional[BaseDispatcher] = None):
         if dispatcher is None:
            dispatcher = LocalDispatcher()
         self.dispatcher = dispatcher

.. tip::
   Defaulting to ``LocalDispatcher`` lets the workflow function with no
   remote capability if needed.

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
   self.dispatcher.dispatch(cmd, with_slurm=False)

.. note::
   The ``with_slurm`` flag determines whether your command is submitted to the
   scheduler. For simple validation commands you will probably want ``False``.
   The default is ``True``.

Step 3: Define ``run_model()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two primary ways to run the model through the dispatcher.

**SLURM script.** Create a SLURM script locally that executes your model and
configure the dispatcher with that script (using ``-s``, see
`Configuring the dispatcher`_). Then ``run_model()`` can be as simple as:

.. code-block:: python

   def run_model(self, run_directory: str, parameter_sample: dict) -> int:
       self.dispatcher.dispatch()
       return 0

This copies your local SLURM script onto the remote host, submits it, and polls
it until it completes.

**Manual commands.** Alternatively, have the dispatcher create the SLURM script
for you. Define the command you wish to execute and let the dispatcher wrap it
in a SLURM script and submit it:

.. code-block:: python

   def run_model(self, run_directory: str, parameter_sample: dict) -> int:
       cmd = "srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 my_app"
       self.dispatcher.dispatch(cmd, run_directory)
       return 0

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

   sampling_dir = "sample_00"
   with RemoteDispatcher(sampling_dir) as dispatcher:
       model = ExampleModel(dispatcher)
       romtools.workflows.run_sampling(
           model=model,
           dispatcher=dispatcher,
       )

Note that the dispatcher is passed to both the model and the sampling workflow.
The ``sampling_directory`` you pass to the dispatcher is created both locally
and remotely, as a subdirectory of your local current directory and your remote
``remote_root``. This mirrored directory structure simplifies sending files
back and forth between hosts, and gives each run its own subdirectory.

.. warning::
   For now, only the ``run_sampling()`` workflow supports the dispatcher.
   Support in other workflows is coming soon.

Configuring the dispatcher
--------------------------

Once the model is updated and the workflow is defined, run the workflow and
configure the dispatcher:

.. code-block:: bash

   python my_workflow.py <config args>

There are three ways to configure:

1. **YAML.** Define a YAML file with all configurable params and pass it with
   ``-i path/to/your/input.yaml``.
2. **CLI.** Set params on the command line. For example, set the
   ``remote_root`` by passing ``-R /path/to/remote/root``.
3. **Combination.** CLI arguments override YAML parameters, so you can use a
   YAML file for the bulk of configuration and CLI args to vary settings from
   run to run:

   .. code-block:: bash

      python my_workflow.py -i path/to/input.yaml -c *.log

.. tip::
   Pass ``-h`` to your workflow, or refer to the ``SCHEMA`` in
   ``hpc/configuration.py``, to see the full schema of configuration arguments.
   You do not need to specify every argument; the ``Configuration`` class
   defines a default value for each parameter.

Core configuration arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**ssh** — establish the connection with the remote host:

- ``remote`` (``-r``): Name of the remote host.
- ``user`` (``-u``): Username for the connection.
- ``port`` (``-p``): Port for the connection.

.. code-block:: yaml

   ssh:
       remote: my_cluster
       user: my_name
       port: 22

**workflow** — file management with the dispatcher:

- ``remote_root`` (``-R``): Directory on the remote host where commands are
  executed, absolute or relative to the home directory.
- ``collect`` (``-c``): Comma-separated list of files, directories, or glob
  patterns to retrieve from the remote run directory. If omitted, nothing is
  retrieved.

.. code-block:: yaml

   workflow:
       remote_root: my_sampling_directory
       collect: *.log, passed.txt

**slurm** — schedule jobs with the dispatcher:

- ``script`` (``-s``): Path to a local SLURM script, uploaded to the remote
  host and submitted on calls to ``dispatch()``.
- ``account`` (``-a``): Account WCID to charge for the job.
- ``job_name`` (``-j``)
- ``num_nodes`` (``-n``)
- ``tasks_per_node`` (``-t``)
- ``wall_time`` (``-w``)
- ``partition`` (``-q``)
- ``poll_interval`` (``-P``): Seconds between ``squeue`` polls.

All arguments other than ``script`` are used when the dispatcher creates the
SLURM script for you from a command.

.. code-block:: yaml

   slurm:
       script: /path/to/script
       poll_interval: 10

**output** — toggle debug logging for the dispatcher:

- ``debug`` (``-d``)

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

   python romtools/hpc/example/workflow.py -r <remote-host> -u <username> -a <account/wcid>

See all available arguments with:

.. code-block:: bash

   python romtools/hpc/example/workflow.py -h
