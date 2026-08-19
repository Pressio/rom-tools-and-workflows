import io
import os
import re
import time
import shlex
import tempfile

import numpy as np
import posixpath as ppath
from typing import Optional, Tuple

from romtools.hpc.util.logger import Logger
from romtools.hpc.util.slurm import SLURM_TERMINAL_STATES, DEFAULT_SLURM_ERRFILE, DEFAULT_SLURM_OUTFILE, slurm_exitcode_to_python_style, parse_sbatch_out_args
from romtools.hpc.connection import Connection, Result
from romtools.hpc.dispatchers import BaseDispatcher
from romtools.hpc.util.file_transfer import pack_results, safe_extract_tar, local_cmd, validate

from romtools.hpc.util.slurm import create_slurm_script

## ----------------------------------------------------------------------------

class RemoteDispatcher(BaseDispatcher):
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
        self.sampling_directory = os.path.basename(sampling_directory)

        # If slurm script specifies out and/or error file, use those instead of our default
        self.slurm_specified_out = None
        self.slurm_specified_err = None

        if not self.config.get("remote") or not self.config.get("user"):
            raise ValueError("Remote host and user must be specified in the configuration to use RemoteDispatcher.")

        # Establish connection, or use the one provided (e.g. by tests)
        if connection is not None:
            self.conn = connection
            self.logger.set_hostname(self.conn.host)
        else:
            self.__connect_to_remote()

        # validate collect and upload patterns
        self.collect_patterns = self.config.get("collect")
        if self.collect_patterns:
            self.collect_patterns = validate(self.collect_patterns)

        self.upload_patterns = self.config.get("upload")
        if self.upload_patterns:
            self.upload_patterns = validate(self.upload_patterns)

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

    def upload(self, run_directory) -> None:
        if not self.upload_patterns:
            return

        remote_root = self.config.get("remote_root")
        tar_name = f"dispatcher-upload-{self.config.get('job_name')}.tar.gz"
        files_packed = pack_results(lambda msg: self.logger.log(msg, local=True), local_cmd, ".", tar_name, self.upload_patterns)
        if not files_packed:
            return
        tar_path = f"{ppath.join(remote_root, run_directory)}/{tar_name}"
        try:
            self.conn.put(tar_name, tar_path)
            self.logger.log(f"Uploaded local file {tar_name} to {self.conn.host}:{tar_path}")
        except Exception as e:
            raise RuntimeError(f"File transfer failed on upload: {e}")
        finally:
            os.remove(tar_name)
        res = safe_extract_tar(lambda cmd: self.conn.run(cmd), tar_path, f"{ppath.join(remote_root, run_directory)}")
        if not res.ok:
            raise RuntimeError(f"Extraction failed on remote! {res.stderr}")

    def __collect_results(self) -> None:
        """
        Collect results from remote HPC runs.
        """
        remote_sampling_dir = ppath.join(self.config.get("remote_root"), self.sampling_directory)
        self.logger.log(
            f"Transferring results from {self.conn.host}:{remote_sampling_dir} -> {self.sampling_directory}",
            local=True,
        )

        archive_name = f"dispatcher-collect-{self.config.get('job_name')}.tar.gz"
        remote_archive_path = ppath.join(self.config.get("remote_root"), archive_name)

        do_collection = pack_results(lambda msg: self.logger.log(msg), lambda cmd: self.conn.run(cmd), remote_sampling_dir, remote_archive_path, self.collect_patterns)

        if not do_collection:
            return

        # Copy remote archive to local
        self.conn.get(remote_archive_path, archive_name)
        self.logger.log(f"Copied remote archive to local: {archive_name}")

        # Clean up remote archive
        self.conn.run(f"rm -f {shlex.quote(remote_archive_path)}")
        self.logger.debug("Cleaned up remote archive.")

        # Unpack local archive into local sampling directory
        os.makedirs(self.sampling_directory, exist_ok=True)
        safe_extract_tar(local_cmd, archive_name, os.path.abspath(self.sampling_directory))
        self.logger.log(f"Results collected in {self.sampling_directory}", local=True)

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
            self.slurm_specified_out, self.slurm_specified_err = parse_sbatch_out_args(script)

            script_name = os.path.basename(script)
            script_base = (
                ppath.join(remote_root, run_directory)
                if run_directory
                else ppath.join(remote_root, self.sampling_directory)
            )
            remote_script_path = ppath.join(script_base, script_name)
            self.conn.put(script, remote_script_path)
            self.logger.log(f"Uploaded local script {script} to {self.conn.host}:{remote_script_path}")
            return remote_script_path

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

        return remote_script_path

    def __submit_slurm_job(self,  cmd: str = None, run_directory: str = None) -> str:
        """
        Generate, upload, and submit a SLURM script.

        Args:
            cmd:  The command to run in the SLURM job.
            run_directory: The directory in which to execute the command on the remote host.

        Returns:
            The SLURM job ID as a string
        """
        remote_script_path = self.__generate_slurm_script(cmd, run_directory=run_directory)

        output_args = []
        if self.slurm_specified_out is None:
            self.slurm_specified_out = DEFAULT_SLURM_OUTFILE
            output_args.append(f"--output={self.slurm_specified_out}")
            # If user only specified out file, they probably expect stderr to go there
            if self.slurm_specified_err is None:
                self.slurm_specified_err = DEFAULT_SLURM_ERRFILE
                output_args.append(f"--error={self.slurm_specified_err}")

        output_cmd = " ".join(output_args)
        if output_cmd:
            output_cmd += " "

        if run_directory:
            full_run_dir = ppath.join(self.config.get("remote_root"), run_directory)
        else:
            full_run_dir = ppath.join(self.config.get("remote_root"), self.sampling_directory)

        script_name = ppath.basename(remote_script_path)
        result = self.conn.run(
            f"cd {shlex.quote(full_run_dir)} && sbatch {output_cmd}{shlex.quote(script_name)}"
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

    def __get_sacct_status(self, job_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Return the SLURM accounting state and exit code for a completed/disappeared job.

        Returns:
            tuple[str, str] | tuple[None, None]:
                (state, exit_code), or (None, None) if sacct does not have the record yet.
        """
        jid = shlex.quote(str(job_id))

        # ExitCode reflects batch script's exit code
        # DerivedExitCode can reflect failures from job steps, even if the
        # main script exits successfully
        cmd = (
            f"sacct -j {jid} -X -n -P "
            "--format=JobIDRaw,State%30,ExitCode,DerivedExitCode"
        )

        result = self.conn.run(cmd)

        if not result.ok:
            self.logger.log(f"sacct failed for job {job_id}: {result.stderr}")
            return None, None

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 4:
                continue

            sacct_job_id = parts[0].strip()
            state = parts[1].strip().split()[0].upper()
            exit_code = parts[2].strip()
            derived_exit_code = parts[3].strip()

            if sacct_job_id == str(job_id):
                # Default to exit_code, return derived_exit_code if exit_code is 0 and derived is not.
                if exit_code == "0:0" and derived_exit_code != "0:0":
                    return state, derived_exit_code
                else:
                    return state, exit_code

        self.logger.debug(f"sacct did not find job {job_id}")
        return None, None

    def __wait_for_status(self, job_id: str):
        timeout = self.config.get("timeout")
        start_wait = time.time()
        sacct_poll_interval = 5

        while True:
            state, exit_code = self.__get_sacct_status(job_id)

            elapsed = time.time() - start_wait
            if state is None:
                if elapsed > timeout:
                    return None

                self.logger.debug(
                    f"Job {job_id} is no longer in squeue, but sacct has no "
                    f"record yet. Waiting..."
                )
                time.sleep(sacct_poll_interval)
                continue

            self.logger.debug(
                f"sacct reports job {job_id}: state={state}, exit_code={exit_code}"
            )

            if state not in SLURM_TERMINAL_STATES:
                if elapsed > timeout:
                    return None
                time.sleep(sacct_poll_interval)
                continue

            if not (state == "COMPLETED" and exit_code == "0:0"):
                self.logger.log(
                    f"Job {job_id} failed: state={state}, exit_code={exit_code}"
                )
            return exit_code

    def __wait_for_job(self, job_id: str) -> str:
        """
        Block until the SLURM job is no longer in the queue (RUNNING or PENDING).

        Args:
            job_id:        The SLURM job ID to monitor.

        Returns:
            Job exit code + linux signal number (the result from sacct)
            Example: '0:0'

            'None' otherwise
        """
        poll_interval = self.config.get("poll_interval")
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
            return slurm_exitcode_to_python_style(self.__wait_for_status(job_id))

        except KeyboardInterrupt:
            self.__cancel_job(job_id)
            raise

    def __get_job_output(self, job_id: int, run_directory:str=None) -> Tuple[str, str]:
        def get_file_contents(filepath: str) -> str:
            result = self.conn.run(f"cat {filepath}")
            if not result.ok:
                self.logger.log(f"Could not read file {filepath}: {result.stderr}")
                return ""

            return result.stdout

        self.logger.log("Retrieving job output...")
        jid = shlex.quote(str(job_id))

        out_dir = os.path.join(self.config.get("remote_root"), self.sampling_directory if run_directory is None else run_directory)

        stdout_filepath = os.path.join(out_dir, self.slurm_specified_out.replace("%j", jid))
        stderr_filepath = os.path.join(out_dir, self.slurm_specified_err.replace("%j", jid))

        return get_file_contents(stdout_filepath), get_file_contents(stderr_filepath)

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

        Returns the SLURM job exit code in sacct format of 0:0, or "No SLURM JOB submitted.".
        """
        if not with_slurm:
            self.__run(cmd, run_directory=run_directory)
            return "No SLURM job submitted."
        job_id = self.__submit_slurm_job(cmd, run_directory)
        status = self.__wait_for_job(job_id)
        self.__collect_results()
        job_stdout, job_stderr = self.__get_job_output(job_id, run_directory)
        return Result(job_stdout, job_stderr, status)

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
