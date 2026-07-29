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
from romtools.hpc.dispatcher_base import DispatcherBase

from romtools.hpc.util.slurm import create_slurm_script

## ----------------------------------------------------------------------------

SLURM_TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}

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
    def __init__(self, sampling_directory: str = "hpctools", logger: Logger = None, connection: Optional[Connection] = None):
        # Initialize the base Dispatcher class (sets up config and logger)
        super().__init__(sampling_directory, logger)

        # Core members
        self.conn : Optional[Connection] = None
        self.collector : Optional[Collector] = None
        self.sampling_directory = os.path.basename(sampling_directory)

        if not self.config.get("remote") or not self.config.get("user"):
            raise ValueError("Remote host and user must be specified in the configuration to use RemoteDispatcher.")

        # Establish connection, or use the one provided (e.g. by tests)
        if connection is not None:
            self.conn = connection
            self.logger.set_hostname(self.conn.host)
        else:
            self.__connect_to_remote()

        # Initialize and validate the Collector
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
            self.conn = Connection(host=self.config.get("remote"), user=self.config.get("user"), port=self.config.get("port"))
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
        return ppath.join(self.config.get("remote_root"), remote_path)

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
            base_command: The command to run in the SLURM job (executed from self.config.get("remote_root")).

        If a local SLURM script is provided (via configuration), it will be uploaded directly without modification
        and base_command will be ignored.

        Returns:
            The remote path of the uploaded SLURM script.
        """
        script = self.config.get("script")
        remote_root = self.config.get("remote_root")

        if base_command is None and script is None:
            raise ValueError("Either a base command or a SLURM script must be provided to the Dispatcher.")

        if script:
            script_name = os.path.basename(script)
            remote_script_path = f"{remote_root}/{self.sampling_directory}/{script_name}"
            self.conn.put(script, remote_script_path)
            self.logger.log(f"Uploaded local script {script} to {self.conn.host}:{remote_script_path}")
            return script_name

        script_content = create_slurm_script(
            job_name       = self.config.get("job_name"),
            num_nodes      = self.config.get("num_nodes"),
            tasks_per_node = self.config.get("tasks_per_node"),
            wall_time      = self.config.get("wall_time"),
            wcid           = self.config.get("account"),
            partition      = self.config.get("partition"),
            command        = base_command)

        self.logger.debug(f"Generated SLURM script:\n{script_content}", local=True)

        remote_script_name = f"{self.config.get('job_name')}_slurm.sh"
        script_base = (
            ppath.join(remote_root, run_directory)
            if run_directory
            else ppath.join(remote_root, self.sampling_directory)
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
            full_run_dir = ppath.join(self.config.get("remote_root"), run_directory)
            result = self.conn.run(
                f"cd {shlex.quote(full_run_dir)} && sbatch {shlex.quote(slurm_script_name)}"
            )
        else:
            full_run_dir = ppath.join(self.config.get("remote_root"), self.sampling_directory)
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

    def __cancel_job(self, job_id: str) -> None:
        """
        Cancel the specified SLURM job.

        Args: job_id (the SLURM job ID)
        """
        try:
            res = self.conn.run(f"scancel {job_id}")
            if res.ok:
                self.logger.log(f"Cancelled job {job_id}.")
            else:
                self.logger.log(f"Could not cancel job {job_id}: {res.stderr}")
        except Exception as e:
            self.logger.log(f"Failed to cancel job {job_id}: {e}")

    def __get_sacct_status(self, job_id: str, sacct_start_time: str = None):
        """
        Return the SLURM accounting state and exit code for a completed/disappeared job.

        Returns:
            tuple[str, str] | tuple[None, None]:
                (state, exit_code), or (None, None) if sacct does not have the record yet.
        """
        jid = shlex.quote(str(job_id))

        cmd = (
            f"sacct -j {jid} -X -n -P "
            "--format=JobIDRaw,State%30,ExitCode"
        )

        if sacct_start_time is not None:
            cmd += f" --starttime {shlex.quote(sacct_start_time)}"

        result = self.conn.run(cmd)

        if not result.ok:
            self.logger.debug(f"sacct failed for job {job_id}: {result.stderr}")
            return None, None

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 3:
                continue

            sacct_job_id = parts[0].strip()
            state = parts[1].strip().split()[0].upper()
            exit_code = parts[2].strip()

            if sacct_job_id == str(job_id):
                return state, exit_code

        return None, None

    def __wait_for_job(self, job_id: str, sacct_start_time=None) -> str:
        """
        Block until the SLURM job is no longer in the queue (RUNNING or PENDING).

        Args:
            job_id:        The SLURM job ID to monitor.
            poll_interval: Seconds between squeue polls (default: 30).

        Returns:
            -1 if unable to retrieve sacct status

            job exit code + linux signal number otherwise (the result from sacct)
            '0:0', etc
        """
        poll_interval = self.config.get("poll_interval")
        accounting_timeout = self.config.get("accounting_timeout")
        self.logger.log(f"Polling SLURM job {job_id} every {poll_interval}s (Ctrl+C to cancel job)...")
        try:
            while True:
                result = self.conn.run(f"squeue -j {job_id} -h")
                if not result.stdout.strip():
                    # Job no longer appears in the queue — it has finished.
                    break
                self.logger.debug(f"Job {job_id} still running...")
                time.sleep(poll_interval)

            self.logger.log(f"Job {job_id} completed. Retrieving sacct status...")
            start_wait = time.time()
            while True:
                state, exit_code = self.__get_sacct_status(job_id, sacct_start_time)
                elapsed = time.time() - start_wait
                if state is None:
                    if elapsed > accounting_timeout:
                        return -1

                    self.logger.debug(
                        f"Job {job_id} is no longer in squeue, but sacct has no "
                        f"record yet. Waiting..."
                    )
                    time.sleep(min(5, poll_interval))
                    continue

                self.logger.debug(
                    f"sacct reports job {job_id}: state={state}, exit_code={exit_code}"
                )

                if state not in SLURM_TERMINAL_STATES:
                    if elapsed > accounting_timeout:
                        return -1
                    time.sleep(min(5, poll_interval))
                    continue

                if not (state == "COMPLETED" and exit_code == "0:0"):
                    self.logger.log(
                        f"Job {job_id} finished unsuccessfully: state={state}, exit_code={exit_code}"
                    )
                return exit_code

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

    def __run(self, cmd: str, run_directory: str = None) -> None:
        resolved_run_dir = ppath.join(self.config.get("remote_root"), run_directory) if run_directory else self.config.get("remote_root")
        remote_cmd = f"cd {shlex.quote(resolved_run_dir)} && {cmd}"
        res = self.conn.run(remote_cmd)
        if not res.ok:
            raise RuntimeError(f"Command failed ({cmd}): {res.stderr}")
        self.logger.debug(f"Executed command on remote host: {cmd}")

    def __get_remote_sacct_start_time(self, margin_seconds: int = 60) -> str:
        """
        Return a Slurm-compatible timestamp from the remote machine's timezone.
        """
        margin_seconds = int(margin_seconds)

        result = self.conn.run(
            f"date -d '{margin_seconds} seconds ago' '+%Y-%m-%dT%H:%M:%S'"
        )

        if not result.ok:
            raise RuntimeError(
                f"Failed to get remote time for sacct start time: {result.stderr}"
            )

        return result.stdout.strip()

    def dispatch(self, cmd: str = None, run_directory: str = None, with_slurm : bool = True) -> str:
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

        Returns the SLURM job exit code in sacct format of 0:0, or "No SLURM JOB submitted.".
        """
        if not with_slurm:
            self.__run(cmd, run_directory=run_directory)
            return "No SLURM job submitted."
        sacct_start_time = self.__get_remote_sacct_start_time()
        job_id = self.__submit_slurm_job(cmd, run_directory)
        status = self.__wait_for_job(job_id, sacct_start_time)
        self.collector.collect_results()
        return status

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
