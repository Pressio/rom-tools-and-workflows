import os
import posixpath as ppath
import shlex
from typing import Optional

from romtools.hpc.logger import Logger
from romtools.hpc.connection import Connection, Result
from romtools.hpc.dispatchers.base_dispatcher import BaseDispatcher
from romtools.hpc.components.caller import RemoteCaller
from romtools.hpc.components.file_manager import RemoteFileManager
from romtools.hpc.components.slurm_job_manager import SlurmJobManager
from romtools.hpc.components.transfer_manager import TransferManager


class RemoteDispatcher(BaseDispatcher):
    """
    Main class of ROM's HPC tools. Establishes SSH connection to remote host, dispatches
    desired workflows, and transfers results back to the local machine.

    Coordinates composed helpers rather than doing the work itself: files go through
    RemoteFileManager, Python calls through RemoteCaller, batch jobs through
    SlurmJobManager, and archives through TransferManager. Paths are taken relative
    to the configured remote root.

    Arguments:
        logger: An instance of the Logger class for logging
        campaign_directory: An optional string naming the directory this campaign
            runs in. It is mirrored locally and under the remote root.
        argv: Argument list to configure from instead of the real process argv.
            Pass [] to ignore the surrounding program's command line.

    The basic command is therefore:
        ssh user@remote -p port
    """
    def __init__(self, campaign_directory: str = "hpctools", logger: Logger = None,
                 connection: Optional[Connection] = None, argv: list = None):
        # Initialize the base Dispatcher class (sets up config and logger)
        super().__init__(campaign_directory, logger, argv=argv)

        # Core members
        self.conn : Optional[Connection] = None
        self.campaign_directory = os.path.basename(campaign_directory)

        # Confirm that connection is possible
        if not self.config.get("remote") or not self.config.get("user"):
            raise ValueError("Remote host and user must be specified in the configuration to use RemoteDispatcher.")

        # Establish connection, or use the one provided (e.g. by tests)
        if connection is not None:
            self.conn = connection
            self.logger.set_hostname(self.conn.host)
        else:
            self._connect_to_remote()

        # Composed sub-classes
        self.files = RemoteFileManager(
            connection=self.conn,
            config=self.config,
            logger=self.logger)

        self.caller = RemoteCaller(
            connection=self.conn,
            config=self.config,
            logger=self.logger,
            files=self.files)

        self.slurm = SlurmJobManager(
            run_cmd=self.conn.run,
            config=self.config,
            logger=self.logger,
            campaign_directory=self.campaign_directory,
            files=self.files)

        self.transfer = TransferManager(
            connection=self.conn,
            config=self.config,
            logger=self.logger,
            campaign_directory=self.campaign_directory,
            files=self.files)

    # ------------------------------------------------------------------
    # Initialization and setup
    # ------------------------------------------------------------------

    def _connect_to_remote(self) -> None:
        """
        Attempts to establish an SSH connection to the remote host using the provided configuration.
        Exits if the connection fails.
        """
        try:
            self.conn = Connection(host=self.config.get("remote"), user=self.config.get("user"), port=self.config.get("port"))
            self.logger.set_hostname(self.conn.host)
            self.logger.log(f"Connection established with {self.conn.host}.", local=True)
            return
        except Exception as e:
            raise RuntimeError(f"Failed to establish SSH connection: {e}")

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self):
        self.conn.close()
        self.logger.log("Connection closed.", local=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def require_relative_path(self, path: str) -> None:
        if ppath.isabs(path):
            raise ValueError(
                f"You must provide a path relative to the remote root (received: {path}). "
                "This workflow also creates the same directory on the local machine."
            )

    def require_supported_concurrency(self, concurrency: int) -> None:
        if concurrency != 1:
            raise ValueError(
                f"Concurrency > 1 is not supported with a RemoteDispatcher (received: {concurrency}). "
                "Use a concurrency of 1 and let SLURM provide the parallelism."
            )

    def upload(self, run_directory) -> None:
        self.transfer.upload(run_directory)

    def collect_results(self) -> None:
        """Bring the finished job's output files back from the remote host."""
        self.transfer.collect_results()

    def run(self, cmd: str, run_directory: str = None) -> Result:
        """
        Run a command directly on the remote host, without SLURM.

        This executes wherever the SSH session lands, which on a cluster is the
        login node. Use it for quick work such as staging or preprocessing;
        anything long or parallel belongs in submit_job(), so that it runs on
        compute nodes instead.

        Args:
            cmd: The command to run.
            run_directory: The directory to run it from, relative to the remote
                root. Defaults to the remote root itself.

        A failing command is reported through the Result, not raised, so that a
        model works the same way whichever dispatcher it is handed.

        Returns a Result object (with stdout, stderr, exit_code, ok)
        """
        remote_cmd = f"cd {shlex.quote(self.files.resolve_path(run_directory))} && {cmd}"
        res = self.conn.run(remote_cmd)
        if res.ok:
            self.logger.debug(f"Executed command on remote host: {cmd}")
        else:
            self.logger.log(f"Command failed ({cmd}), exit code {res.exit_code}: {res.stderr}")
        return res
