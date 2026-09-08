import os

import numpy as np
import posixpath as ppath
from typing import Optional

from romtools.hpc.logger import Logger
from romtools.hpc.connection import Connection, Result
from romtools.hpc.dispatchers.base_dispatcher import BaseDispatcher
from romtools.hpc.archive import validate_file_patterns
from romtools.hpc.components.caller import RemoteCaller
from romtools.hpc.components.command_runner import RemoteCommandRunner
from romtools.hpc.components.file_manager import RemoteFileManager
from romtools.hpc.components.slurm_job_manager import SlurmJobManager
from romtools.hpc.components.transfer_manager import TransferManager


class RemoteDispatcher(BaseDispatcher):
    """
    Main class of ROM's HPC tools. Establishes SSH connection to remote host, dispatches
    desired workflows, and transfers results back to the local machine.

    Coordinates composed helpers rather than doing the work itself: files go through
    RemoteFileManager, plain commands through RemoteCommandRunner, batch jobs through
    SlurmJobManager, and archives through TransferManager.

    Arguments:
        logger: An instance of the Logger class for logging
        campaign_directory: An optional string naming the directory this campaign
            runs in. It is mirrored locally and under the remote root.

    The basic command is therefore:
        ssh user@remote -p port
    """
    def __init__(self, campaign_directory: str = "hpctools", logger: Logger = None, connection: Optional[Connection] = None):
        # Initialize the base Dispatcher class (sets up config and logger)
        super().__init__(campaign_directory, logger)

        # Core members
        self.conn : Optional[Connection] = None
        self.campaign_directory = os.path.basename(campaign_directory)

        if not self.config.get("remote") or not self.config.get("user"):
            raise ValueError("Remote host and user must be specified in the configuration to use RemoteDispatcher.")

        # Establish connection, or use the one provided (e.g. by tests)
        if connection is not None:
            self.conn = connection
            self.logger.set_hostname(self.conn.host)
        else:
            self.__connect_to_remote()

        # validate collect and upload patterns
        self.collect_patterns = validate_file_patterns(self.config.get("collect"))
        self.upload_patterns = validate_file_patterns(self.config.get("upload"))

        self.caller = RemoteCaller(connection=self.conn, config=self.config, logger=self.logger)
        self.files = RemoteFileManager(connection=self.conn, config=self.config, logger=self.logger)
        self.runner = RemoteCommandRunner(connection=self.conn, config=self.config, logger=self.logger)
        self.slurm = SlurmJobManager(
            connection=self.conn,
            config=self.config,
            logger=self.logger,
            campaign_directory=self.campaign_directory,
        )
        self.transfer = TransferManager(
            connection=self.conn,
            config=self.config,
            logger=self.logger,
            campaign_directory=self.campaign_directory,
            collect_patterns=self.collect_patterns,
            upload_patterns=self.upload_patterns,
        )

    # ------------------------------------------------------------------
    # Initialization and setup
    # ------------------------------------------------------------------

    def __connect_to_remote(self) -> None:
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

    def upload(self, run_directory) -> None:
        self.transfer.upload(run_directory)

    def put(self, local_path: str, remote_path: str) -> None:
        self.files.put(local_path, remote_path)

    def get(self, remote_path: str, local_path: str) -> None:
        self.files.get(remote_path, local_path)

    def path_exists(self, path: str) -> bool:
        return self.files.path_exists(path)

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

    def create_empty_dir(self, dir_name: str):
        self.files.create_empty_dir(dir_name)

    def list_dir(self, path: str) -> list:
        return self.files.list_dir(path)

    def remove(self, path: str) -> None:
        self.files.remove(path)

    def write_text(self, path: str, content: str) -> None:
        self.files.write_text(path, content)

    def dispatch(self, cmd: str = None, run_directory: str = None, with_slurm : bool = True) -> Result:
        """
        Main method of the Dispatcher. Dispatches provided work to the
        remote host, polls the job, and collects results.

        Args:
            cmd: The command to run in the SLURM job.
                 Should be executable from self.config.get("remote_root") or the specified run_directory.
                 If not provided, dispatcher must be configured with
                 a SLURM script that includes the command to run.
            run_directory: The directory in which to execute the command or SLURM job
                 on the remote host. If not provided, defaults to self.config.get("remote_root").
            with_slurm: If True, the command will be run as a SLURM job.
                 If False, the command will be executed directly without SLURM.

        Returns a Result object (with stdout, stderr, exitcode, ok)
        """
        if not with_slurm:
            return self.runner.run(cmd, run_directory=run_directory)
        job_id = self.slurm.submit(cmd, run_directory)
        status = self.slurm.wait(job_id)
        self.transfer.collect_results()
        job_stdout, job_stderr = self.slurm.get_output(job_id, run_directory)
        return Result(job_stdout, job_stderr, status)

    def np_savetxt(self, path: str, arr: np.ndarray, fmt: str) -> None:
        self.files.np_savetxt(path, arr, fmt)

    def np_savez(self, path: str, **arrays) -> None:
        self.files.np_savez(path, **arrays)
