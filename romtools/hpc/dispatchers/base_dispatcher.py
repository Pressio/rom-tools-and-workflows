
import numpy as np

from romtools.hpc.components.caller import BaseCaller
from romtools.hpc.components.file_manager import BaseFileManager
from romtools.hpc.components.slurm_job_manager import SlurmJobManager
from romtools.hpc.logger import Logger
from romtools.hpc.configuration import Configuration
from romtools.hpc.connection import Result


class BaseDispatcher:
    """
    Shared configuration, logging, and public API for the dispatchers.

    The public methods here are a flat facade over composed helpers, so that
    workflows and models depend on one stable interface rather than on how the
    work is split up internally. Subclasses install the local or remote
    implementation of each helper, and only override a method when the
    behavior genuinely differs.
    """

    def __init__(self, campaign_directory: str = "hpctools", logger: Logger = None):
        self.config = Configuration().to_dict()
        self.logger = logger if logger is not None else Logger(self.config["debug"])
        self.campaign_directory = campaign_directory

        # Subclasses replace these with local or remote implementations
        self.caller: BaseCaller = BaseCaller(config=self.config, logger=self.logger)
        self.files: BaseFileManager = BaseFileManager(config=self.config, logger=self.logger)

        # Installed by subclasses that can reach a batch scheduler
        self.slurm: SlurmJobManager = None

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------
    # Running work
    # ------------------------------------------------------------------

    def call(self, target: str, *args, run_directory: str = None, **kwargs):
        """
        Run a Python callable, given as "module:qualname", and return its result.

        Args:
            target: The callable to run, e.g. "my_model:evaluate".
            run_directory: The directory to run from. Modules staged there are
                importable. Defaults to the dispatcher's own default directory.

        Positional and keyword arguments are forwarded to the callable.
        """
        return self.caller.call(target, *args, run_directory=run_directory, **kwargs)

    def run(self, cmd: str, run_directory: str = None) -> Result:
        """
        Run a command directly on the execution host.

        Returns a Result object (with stdout, stderr, exit_code, ok)
        """
        raise NotImplementedError

    def submit_job(self, cmd: str = None, run_directory: str = None) -> Result:
        """
        Submit work to SLURM, wait for it to finish, and read back its output.

        Args:
            cmd: The command to run in the SLURM job, executable from
                run_directory. If omitted, the dispatcher must be configured
                with a SLURM script that includes the command to run.
            run_directory: The directory to run the job in on the execution
                host. Defaults to the campaign directory.

        Returns a Result object (with stdout, stderr, exit_code, ok)
        """
        if self.slurm is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no batch scheduler available to submit to."
            )

        job_id = self.slurm.submit(cmd, run_directory)
        status = self.slurm.wait(job_id)
        self.collect_results()
        job_stdout, job_stderr = self.slurm.get_output(job_id, run_directory)
        return Result(job_stdout, job_stderr, status)

    def collect_results(self) -> None:
        """
        Bring a finished job's output back to the local machine.

        Work that runs on this machine leaves its results in place, so only
        dispatchers that run work elsewhere have anything to do here.
        """
        pass

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def put(self, local_path: str, remote_path: str) -> None:
        self.files.put(local_path, remote_path)

    def get(self, remote_path: str, local_path: str) -> None:
        self.files.get(remote_path, local_path)

    def path_exists(self, path: str) -> bool:
        return self.files.path_exists(path)

    def create_empty_dir(self, dir_name: str):
        self.files.create_empty_dir(dir_name)

    def list_dir(self, path: str) -> list:
        return self.files.list_dir(path)

    def remove(self, path: str) -> None:
        self.files.remove(path)

    def remove_dir(self, path: str) -> None:
        self.files.remove_dir(path)

    def write_text(self, path: str, content: str) -> None:
        self.files.write_text(path, content)

    def read_text(self, path: str) -> str:
        return self.files.read_text(path)

    def np_savetxt(self, path: str, arr: np.ndarray, fmt: str) -> None:
        self.files.np_savetxt(path, arr, fmt)

    def np_savez(self, path: str, **arrays) -> None:
        self.files.np_savez(path, **arrays)

    def upload(self, run_directory) -> None:
        """Send the configured upload patterns to the run directory."""
        pass

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def require_absolute_path(self, path: str) -> None:
        # Overridden by dispatchers that can only address absolute paths
        pass

    def require_relative_path(self, path: str) -> None:
        # Overridden by dispatchers that resolve paths against a root
        pass

    def require_supported_concurrency(self, concurrency: int) -> None:
        # Overridden by dispatchers that cannot run concurrent model evaluations
        pass

    def get_config(self, param: str = None) -> dict:
        if param is None:
            return self.config
        return self.config.get(param, None)
