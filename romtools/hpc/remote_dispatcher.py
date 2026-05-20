import io
import os
import re
import time
import shlex
import tempfile
import posixpath

import numpy as np

from romtools.hpc.dispatcher_base import DispatcherBase
from romtools.hpc.logger import Logger
from romtools.hpc.connection import Connection
from romtools.hpc.scheduler import create_slurm_script

## ----------------------------------------------------------------------------

class RemoteDispatcher(DispatcherBase):
    """
    Main class of ROM's HPC tools. Establishes SSH connection to remote host, dispatches
    desired workflows, and transfers results back to the local machine.

    Arguments:
        logger: An instance of the Logger class for logging
        sampling_directory: An optional string for your local output directory
        local_only: If True, do not attempt to establish an SSH connection or run any remote commands.

    The basic command is therefore:
        ssh user@remote -p port

    Ensure that this command works without a password prompt (e.g., by setting up SSH keys)
    before using this tool.
    """
    def __init__(self, logger: Logger = None, sampling_directory: str = "hpctools"):
        if not self.config.remote or not self.config.user:
            raise ValueError("Remote host and user must be specified in the configuration to use RemoteDispatcher.")
        # Core members
        super().__init__(logger, sampling_directory)

        # Establish connection
        self.conn : Connection = self.__connect_to_remote()

        # Then create the remote output directory
        self.__create_remote_directory(
            os.path.join(self.config.remote_root, self.sampling_directory),
            base_dir=True
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
        if os.path.isabs(remote_path) or preserve_relative:
            return remote_path
        return os.path.join(self.config.remote_root, remote_path)

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
        script_base = f"{shlex.quote(self.config.remote_root)}/{run_directory}" if run_directory else f"{shlex.quote(self.config.remote_root)}/{self.sampling_directory}"
        remote_script_path = f"{script_base}/{self.config.job_name}_slurm.sh"

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
            full_run_dir = f"{shlex.quote(self.config.remote_root)}/{run_directory}"
            result = self.conn.run(f"cd {full_run_dir} && sbatch {shlex.quote(slurm_script_name)}")
        else:
            result = self.conn.run(f"cd {shlex.quote(self.sampling_directory)} && sbatch {shlex.quote(slurm_script_name)}")

        if not result.ok:
            raise RuntimeError(f"sbatch failed:\n{result.stderr}")

        # sbatch output: "Submitted batch job <id>"
        match = re.search(r"(\d+)", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout!r}")
        job_id = match.group(1)

        self.current_jobs.append(job_id)

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
        self.logger.log(f"Waiting for SLURM job {job_id} to complete (polling every {self.config.poll_interval}s)...")
        while True:
            result = self.conn.run(f"squeue -j {job_id} -h")
            if not result.stdout.strip():
                # Job no longer appears in the queue — it has finished.
                break
            self.logger.debug(f"Job {job_id} still running...")
            time.sleep(self.config.poll_interval)
        self.logger.log(f"Job {job_id} completed.")

    # ------------------------------------------------------------------
    # Result collection
    # ------------------------------------------------------------------

    def __collect_results(self) -> None:
        """
        Transfer the self.sampling_directory from remote to local.
        """
        remote_sampling_dir = f"{self.config.remote_root}/{self.sampling_directory}"
        self.logger.log(f"Transferring results from {self.conn.host}:{remote_sampling_dir} -> {self.sampling_directory}", local=True)

        # Pack all results into an archive
        archive_name = f"{self.config.job_name}.tar.gz"
        remote_archive_path = f"{self.config.remote_root}/{archive_name}"
        pack_cmd = (
            f"tar -czf {shlex.quote(remote_archive_path)} "
            f"-C {shlex.quote(self.config.remote_root)}/{self.sampling_directory} ."
        )
        pack_result = self.conn.run(pack_cmd)
        if not pack_result.ok:
            raise RuntimeError(f"Remote result archive failed: {pack_result.stderr}")

        self.logger.log(f"Packed remote results into archive: {remote_archive_path}")

        # Copy remote archive to local
        self.conn.get(remote_archive_path, archive_name)

        self.logger.log(f"Copied remote archive to local: {archive_name}")

        # Unzip local archive into self.sampling_directory
        os.makedirs(self.sampling_directory, exist_ok=True)
        unpack_cmd = f"tar -xzf {shlex.quote(archive_name)} -C {shlex.quote(self.sampling_directory)}"
        result = os.system(unpack_cmd)
        if result != 0:
            raise RuntimeError(f"Failed to unpack local archive: {archive_name}")

        self.logger.log(f"Results collected in {self.sampling_directory}", local=True)

        # Clean up both archives
        os.remove(archive_name)
        self.conn.run(f"rm {shlex.quote(remote_archive_path)}")
        self.logger.debug(f"Cleaned up archives.")

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

    def __create_remote_directory(self, remote_dir: str, base_dir = False) -> None:
        remote_dir = self.__resolve_remote_path(remote_dir, preserve_relative=base_dir)
        self.conn.run(f"mkdir -p {shlex.quote(remote_dir)}")
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
        self.__collect_results()

    def np_savetxt(self, path: str, arr: np.ndarray, fmt: str) -> None:
        buffer = io.StringIO()
        np.savetxt(buffer, arr, fmt=fmt)
        self.__write_text(path, buffer.getvalue())
        self.logger.log(f"Saved array to path {path}", local=False)

    def np_savez(self, path: str, **arrays) -> None:
        """
        Write multiple arrays to a .npz file.
        The .npz file is first written to a local temp directory and then uploaded to the remote host.
        """
        remote_path = posixpath.normpath(path)
        if not remote_path.endswith(".npz"):
            remote_path += ".npz"

        remote_dir = posixpath.dirname(remote_path) or "."
        assert self.path_exists(remote_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, posixpath.basename(remote_path))
            np.savez(local_path, **arrays)
            self.put(local_path, remote_path)

        final_path = remote_path
        self.logger.log(f"Saved arrays to path {final_path}", local=(not self.conn))
