import io
import os
import re
import time
import shlex
import tempfile

import numpy as np
import posixpath as ppath
from typing import Optional

from romtools.hpc.util.logger import Logger
from romtools.hpc.collector import Collector
from romtools.hpc.connection import Connection
from romtools.hpc.configuration import Configuration

from romtools.hpc.util.slurm import create_slurm_script
from romtools.hpc.util.decorators import require_connection

## ----------------------------------------------------------------------------

class RemoteDispatcher(DispatcherBase):
    """
    Main class of ROM's HPC tools. Establishes SSH connection to remote host, dispatches
    desired workflows, and transfers results back to the local machine.

    Arguments:
        logger: An instance of the Logger class for logging
        sampling_directory: An optional string for your local output directory

    The basic command is therefore:
        ssh user@remote -p port
    """
    def __init__(self, sampling_directory: str = "hpctools", logger: Logger = None):
        # Core members
        super().__init__(logger, sampling_directory)
        self.conn : Optional[Connection] = None
        self.collector : Optional[Collector] = None
        self.sampling_directory = os.path.basename(sampling_directory)

        # Parse configuration
        self.config = Configuration()
        self.logger = logger if logger is not None else Logger(self.config.debug)

        if not self.config.remote or not self.config.user:
            raise ValueError("Remote host and user must be specified in the configuration to use RemoteDispatcher.")

        # Establish connection
        self.__connect_to_remote()
        self.collector = Collector(
            self.conn,
            self.config,
            sampling_directory=self.sampling_directory,
            logger=self.logger,
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
            self.conn = Connection(host=self.config.remote, user=self.config.user, port=self.config.port)
            self.logger.set_hostname(self.conn.host)
            self.logger.log(f"Connection established with {self.conn.host}.", local=True)
            return
        except Exception as e:
            raise RuntimeError(f"Failed to establish SSH connection: {e}")

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def __resolve_remote_path(self, remote_path: str, preserve_relative: bool = False) -> str:
        if ppath.isabs(remote_path) or preserve_relative:
            return remote_path
        return ppath.join(self.config.remote_root, remote_path)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self):
        self.conn.close()
        self.logger.log("Connection closed.", local=True)

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    def __generate_slurm_script(self, base_command: str = None, run_directory: str = None) -> str:
        """
        Render a SLURM job script, write it to a local temp file, upload it to the
        remote host, and return the remote path.

        Args:
            base_command: The command to run in the SLURM job (executed from self.config.remote_root).

        If a local SLURM script is provided (via configuration), it will be uploaded directly without modification
        and base_command will be ignored.

        Returns:
            The remote path of the uploaded SLURM script.
        """
        if base_command is None and self.config.script is None:
            raise ValueError("Either a base command or a SLURM script must be provided to the Dispatcher.")

        if self.config.script:
            script_name = os.path.basename(self.config.script)
            remote_script_path = f"{self.config.remote_root}/{self.sampling_directory}/{script_name}"
            self.conn.put(self.config.script, remote_script_path)
            self.logger.log(f"Uploaded local script {self.config.script} to {self.conn.host}:{remote_script_path}")
            return script_name

        script_content = create_slurm_script(
            job_name=self.config.job_name,
            num_nodes=self.config.num_nodes,
            tasks_per_node=self.config.tasks_per_node,
            wall_time=self.config.wall_time,
            wcid=self.config.account,
            partition=self.config.partition,
            command=base_command)
        self.logger.debug(f"Generated SLURM script:\n{script_content}", local=True)

        remote_script_name = f"{self.config.job_name}_slurm.sh"
        script_base = (
            ppath.join(self.config.remote_root, run_directory)
            if run_directory
            else ppath.join(self.config.remote_root, self.sampling_directory)
        )
        remote_script_path = ppath.join(script_base, remote_script_name)

        # Write the SLURM script content to the remote file using a heredoc
        outer = "__HPCTOOLS_SLURM_EOF__"
        cmd = f"cat > {shlex.quote(remote_script_path)} << '{outer}'\n{script_content}\n{outer}\n"
        res = self.conn.run(cmd)

        if not res.ok:
            raise RuntimeError(f"Failed to write SLURM script: {res.stderr}")

        self.logger.log(f"Wrote SLURM script to {self.conn.host}:{remote_script_path}")

        return remote_script_name

    def __submit_slurm_job(self,  cmd: str = None, run_directory: str = None) -> str:
        """
        Generate, upload, and submit a SLURM script.

        Args:
            cmd:  The command to run in the SLURM job.
            run_directory: The directory in which to execute the command on the remote host.

        Returns:
            The SLURM job ID as a string
        """
        slurm_script_name = self.__generate_slurm_script(cmd, run_directory=run_directory)
        if run_directory:
            full_run_dir = ppath.join(self.config.remote_root, run_directory)
            result = self.conn.run(
                f"cd {shlex.quote(full_run_dir)} && sbatch {shlex.quote(slurm_script_name)}"
            )
        else:
            full_run_dir = ppath.join(self.config.remote_root, self.sampling_directory)
            result = self.conn.run(
                f"cd {shlex.quote(full_run_dir)} && sbatch {shlex.quote(slurm_script_name)}"
            )

        if not result.ok:
            raise RuntimeError(f"sbatch failed:\n{result.stderr}")

        # sbatch output: "Submitted batch job <id>"
        match = re.search(r"(\d+)", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout!r}")
        job_id = match.group(1)

        self.logger.log(f"Submitted SLURM job {job_id}")
        return job_id

    # ------------------------------------------------------------------
    # Job monitoring
    # ------------------------------------------------------------------

    def __wait_for_job(self, job_id: str) -> None:
        """
        Block until the SLURM job is no longer in the queue (RUNNING or PENDING).

        Args:
            job_id:        The SLURM job ID to monitor.
            poll_interval: Seconds between squeue polls (default: 30).
        """
        self.logger.log(f"Polling SLURM job {job_id} every {self.config.poll_interval}s (Ctrl+C to cancel job)...")
        try:
            while True:
                result = self.conn.run(f"squeue -j {job_id} -h")
                if not result.stdout.strip():
                    # Job no longer appears in the queue — it has finished.
                    break
                self.logger.debug(f"Job {job_id} still running...")
                time.sleep(self.config.poll_interval)
            self.logger.log(f"Job {job_id} completed.")
        except KeyboardInterrupt:
            self.__cancel_job(job_id)
            raise

    # ------------------------------------------------------------------
    # I/O methods
    # ------------------------------------------------------------------

    def __write_text(self, remote_path: str, content: str) -> None:
        """
        Write text content to a file on the remote host.
        """
        remote_path = self.__resolve_remote_path(remote_path)
        outer = "__HPCTOOLS_FILE_EOF__"
        cmd = f"cat > {shlex.quote(remote_path)} << '{outer}'\n{content}\n{outer}\n"
        res = self.conn.run(cmd)
        if not res.ok:
            raise RuntimeError(f"Failed to write remote file {remote_path}: {res.stderr}")
        self.logger.log(f"Wrote remote file: {remote_path}")

    @require_connection
    def __create_remote_directory(self, remote_dir: str, base_dir = False) -> None:
        remote_dir = self.__resolve_remote_path(remote_dir, preserve_relative=base_dir)
        result = self.conn.run(f"mkdir -p {shlex.quote(remote_dir)}")
        if not result.ok:
            raise RuntimeError(f"Failed to create remote directory {remote_dir}: {result.stderr}")
        self.logger.log(f"Created remote directory: {remote_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(self, local_path: str, remote_path: str) -> None:
        remote_path = self.__resolve_remote_path(remote_path)
        self.conn.put(local_path, remote_path)
        self.logger.log(f"Uploaded local file {local_path} to {self.conn.host}:{remote_path}")

    def get(self, remote_path: str, local_path: str) -> None:
        remote_path = self.__resolve_remote_path(remote_path)
        self.conn.get(remote_path, local_path)
        self.logger.log(f"Downloaded remote file {self.conn.host}:{remote_path} to local path {local_path}")

    def path_exists(self, path: str) -> bool:
        remote_path = self.__resolve_remote_path(path)
        result = self.conn.run(f"test -e {shlex.quote(remote_path)}")
        return result.ok

    def create_empty_dir(self, dir_name: str):
        self.__create_remote_directory(dir_name)

    def dispatch(self, cmd: str, run_directory: str = None) -> str:
        job_id = self.__submit_slurm_job(cmd, run_directory)
        self.__wait_for_job(job_id)
        self.collector.collect_results()
        return job_id

    def np_savetxt(self, path: str, arr: np.ndarray, fmt: str) -> None:
        buffer = io.StringIO()
        np.savetxt(buffer, arr, fmt=fmt)
        self.__write_text(path, buffer.getvalue())
        self.logger.log(f"Saved array to path {path}", local=False)

    def np_savez(self, path: str, **arrays) -> None:
        """
        Write multiple arrays to a .npz file.
            - If a connection exists, the .npz file is first written to a local temp directory and then uploaded to the remote host.
            - If no connection exists, the .npz file is written directly to the specified path.
        """
        remote_path = ppath.normpath(path)
        if not remote_path.endswith(".npz"):
            remote_path += ".npz"

        remote_dir = ppath.dirname(remote_path) or "."
        assert self.path_exists(remote_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, ppath.basename(remote_path))
            np.savez(local_path, **arrays)
            self.put(local_path, remote_path)

        final_path = remote_path

        self.logger.log(f"Saved arrays to path {final_path}", local=(not self.conn))
