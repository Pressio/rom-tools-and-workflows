import os
import re
import time
import shlex
from typing import Optional

from romtools.hpc.dispatcher_config import DispatcherConfig
from romtools.hpc.logger import Logger
from romtools.hpc.connection import Connection
from romtools.hpc.scheduler import create_slurm_script
from romtools.hpc.decorators import require_connection

## ----------------------------------------------------------------------------

class Dispatcher:
    """
    Main class of ROM's HPC tools. Establishes SSH connection to remote host, dispatches
    desired workflows, and transfers results back to the local machine.

    Arguments:
        logger: An instance of the Logger class for logging
        sampling_directory: An optional string for your local output directory

    The basic command is therefore:
        ssh user@remote -p port

    Ensure that this command works without a password prompt (e.g., by setting up SSH keys)
    before using this tool.
    """
    def __init__(self, logger: Logger, sampling_directory: str = "hpctools"):
        # Core members
        self.logger = logger
        self.conn : Optional[Connection] = None
        self.sampling_directory = os.path.basename(sampling_directory)
        self.config = DispatcherConfig(self.logger)

        # Maintain list of current jobs
        self.current_jobs = []

        # Initialization
        self.__connect_to_remote()
        self.__set_up_directories()

    # ------------------------------------------------------------------
    # Initialization and setup
    # ------------------------------------------------------------------

    def __connect_to_remote(self) -> None:
        assert self.config.remote is not None, "Remote host must be specified (use --remote argument on command line)."
        assert self.config.user is not None, "Username must be specified (use --user argument on command line)."

        try:
            self.conn = Connection(host=self.config.remote, user=self.config.user, port=self.config.port)
            self.logger.set_hostname(self.conn.host)
            self.logger.log(f"Connection established with {self.conn.host}.", local=True)
            return
        except Exception as e:
            raise RuntimeError(f"Failed to establish SSH connection: {e}")

    def __set_up_directories(self) -> None:
        # First, create the local output directory
        os.makedirs(self.sampling_directory, exist_ok=True)
        self.logger.log(f"Local sampling directory: {self.sampling_directory}", local=True)

        # Then create the remote output directory
        if self.conn:
            self.create_remote_directory(
                os.path.join(self.config.remote_root, self.sampling_directory),
                base_dir=True
            )

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self):
        if self.conn:
            self.conn.close()
            self.logger.log("Connection closed.", local=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    @require_connection
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

    @require_connection
    def __submit_slurm_job(self,  cmd: str = None, run_directory: str = None) -> str:
        """
        Generate, upload, and submit a SLURM script.

        Args:
            cmd:  The command to run in the SLURM job.
            run_directory: The directory in which to execute the command on the remote host.

        Returns:
            The SLURM job ID as a string.s
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

    @require_connection
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

    @require_connection
    def __collect_results(self) -> None:
        """
        Transfer the self.sampling_directory from remote to local.
        """
        remote_sampling_dir = f"{self.config.remote_root}/{self.sampling_directory}"
        self.logger.log(f"Transfering results from {self.conn.host}:{remote_sampling_dir} -> {self.sampling_directory}", local=True)

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
    # Public API
    # ------------------------------------------------------------------

    @require_connection
    def cancel_job(self, job_id: str) -> str:
        output = self.conn.run(f"scancel {job_id}")
        self.logger.log(f"Cancelled SLURM job {job_id}.")
        return output

    @require_connection
    def create_remote_directory(self, remote_dir: str, base_dir = False) -> None:
        # TODO: Need better method here
        if not os.path.isabs(remote_dir) and not base_dir:
            remote_dir = f"{shlex.quote(self.config.remote_root)}/{remote_dir}"

        self.conn.run(f"mkdir -p {shlex.quote(remote_dir)}")
        self.logger.log(f"Created remote directory: {remote_dir}")

    @require_connection
    def dispatch(self, cmd: str, run_directory: str = None) -> str:
        job_id = self.__submit_slurm_job(cmd, run_directory)
        self.__wait_for_job(job_id)
        self.__collect_results()
