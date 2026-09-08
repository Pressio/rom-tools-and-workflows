"""SLURM job submission and monitoring for the remote dispatcher."""

import os
import re
import time
import shlex

import posixpath as ppath
from typing import Optional, Tuple

from romtools.hpc.connection import Connection
from romtools.hpc.logger import Logger
from romtools.hpc.slurm import SLURM_TERMINAL_STATES, DEFAULT_SLURM_ERRFILE, DEFAULT_SLURM_OUTFILE, slurm_exitcode_to_python_style, parse_sbatch_out_args
from romtools.hpc.slurm import create_slurm_script


class SlurmJobManager:
    """
    Generates and submits SLURM scripts, then polls the job until it finishes.

    Remote-only; LocalDispatcher has no use for it.

    Arguments:
        connection: An established Connection to the remote host
        config: The dispatcher's configuration dictionary
        logger: An instance of the Logger class for logging
        campaign_directory: The directory jobs run in when no run_directory is given
    """

    def __init__(self, connection: Connection, config: dict = None, logger: Logger = None,
                 campaign_directory: str = None):
        self.conn = connection
        self.config = config if config is not None else {}
        self.logger = logger
        self.campaign_directory = campaign_directory

        # If slurm script specifies out and/or error file, use those instead of our default
        self.slurm_specified_out = None
        self.slurm_specified_err = None

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
                else ppath.join(remote_root, self.campaign_directory)
            )
            remote_script_path = ppath.join(script_base, script_name)
            self.conn.put(script, remote_script_path)
            self.logger.debug(f"Uploaded local script {script} to {self.conn.host}:{remote_script_path}")
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
            else ppath.join(remote_root, self.campaign_directory)
        )
        remote_script_path = ppath.join(script_base, remote_script_name)

        # Write the SLURM script content to the remote file using a heredoc
        outer = "__HPCTOOLS_SLURM_EOF__"
        cmd = f"cat > {shlex.quote(remote_script_path)} << '{outer}'\n{script_content}\n{outer}\n"
        res = self.conn.run(cmd)

        if not res.ok:
            raise RuntimeError(f"Failed to write SLURM script: {res.stderr}")

        self.logger.debug(f"Wrote SLURM script to {self.conn.host}:{remote_script_path}")

        return remote_script_path

    def submit(self,  cmd: str = None, run_directory: str = None) -> str:
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
            full_run_dir = ppath.join(self.config.get("remote_root"), self.campaign_directory)

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

        poll_interval = self.config.get("poll_interval")
        self.logger.log(
            f"Submitted SLURM job {job_id}, polling every {poll_interval}s (Ctrl+C to cancel job)."
        )
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
            self.logger.debug(f"sacct failed for job {job_id}: {result.stderr}")
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
                    self.logger.log(f"Gave up retrieving the sacct status of job {job_id} after {timeout}s.")
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
                    self.logger.log(f"Job {job_id} did not reach a terminal state within {timeout}s (state={state}).")
                    return None
                time.sleep(sacct_poll_interval)
                continue

            if not (state == "COMPLETED" and exit_code == "0:0"):
                self.logger.log(
                    f"Job {job_id} failed: state={state}, exit_code={exit_code}"
                )
            return exit_code

    def wait(self, job_id: str) -> str:
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
        try:
            while True:
                result = self.conn.run(f"squeue -j {job_id} -h")
                if not result.stdout.strip():
                    # Job no longer appears in the queue — it has finished.
                    break
                self.logger.debug(f"Job {job_id} still running...")
                time.sleep(poll_interval)

            self.logger.log(f"Job {job_id} finished.")
            self.logger.debug(f"Retrieving sacct status for job {job_id}...")
            return slurm_exitcode_to_python_style(self.__wait_for_status(job_id))

        except KeyboardInterrupt:
            self.__cancel_job(job_id)
            raise

    def get_output(self, job_id: int, run_directory:str=None) -> Tuple[str, str]:
        def get_file_contents(filepath: str) -> str:
            result = self.conn.run(f"cat {filepath}")
            if not result.ok:
                self.logger.log(f"Could not read file {filepath}: {result.stderr}")
                return ""

            return result.stdout

        jid = shlex.quote(str(job_id))

        out_dir = os.path.join(self.config.get("remote_root"), self.campaign_directory if run_directory is None else run_directory)

        stdout_filepath = os.path.join(out_dir, self.slurm_specified_out.replace("%j", jid))
        stderr_filepath = os.path.join(out_dir, self.slurm_specified_err.replace("%j", jid))

        stdout = get_file_contents(stdout_filepath)
        stderr = get_file_contents(stderr_filepath)

        self.logger.debug("Retrieved job output.")

        return stdout, stderr
