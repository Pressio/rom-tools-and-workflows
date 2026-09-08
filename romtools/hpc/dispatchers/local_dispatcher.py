

import os

import numpy as np

from romtools.hpc.logger import Logger
from romtools.hpc.dispatchers.base_dispatcher import BaseDispatcher
from romtools.hpc.connection import Result
from romtools.hpc.components.caller import LocalCaller
from romtools.hpc.components.command_runner import LocalCommandRunner
from romtools.hpc.components.file_manager import LocalFileManager


class LocalDispatcher(BaseDispatcher):
    """
    LocalDispatcher is a subclass of BaseDispatcher that implements the core functionality
    for dispatching ROM workflows on the local machine. It composes local implementations
    of the file, command, and call helpers, making it suitable for local execution.
    """
    def __init__(self, campaign_directory: str = "hpctools", logger: Logger = None):
        # Local execution has no use for remote/SLURM CLI flags, and reading
        # the real process argv here would pick up whatever CLI args the
        # embedding process was started with (e.g. pytest's own flags).
        super().__init__(campaign_directory=campaign_directory, logger=logger, argv=[])

        self.caller = LocalCaller(config=self.config, logger=self.logger)
        self.files = LocalFileManager(config=self.config, logger=self.logger)
        self.runner = LocalCommandRunner(config=self.config, logger=self.logger)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, remote_path: str, local_path: str) -> None:
        self.files.get(remote_path, local_path)

    def put(self, local_path: str, remote_path: str) -> None:
        self.files.put(local_path, remote_path)

    def path_exists(self, path: str) -> bool:
        return self.files.path_exists(path)

    def require_absolute_path(self, path: str) -> None:
        # Only LocalDispatcher needs absolute paths (for now)
        assert os.path.isabs(path), f"You must provide an absolute path (received: {path})"

    def create_empty_dir(self, dir_name: str):
        self.files.create_empty_dir(dir_name)

    def list_dir(self, path: str) -> list:
        return self.files.list_dir(path)

    def remove(self, path: str) -> None:
        self.files.remove(path)

    def write_text(self, path: str, content: str) -> None:
        self.files.write_text(path, content)

    def dispatch(self, cmd: str, run_directory: str = None) -> Result:
        """
        Returns:
            sacct format string of job exit code + linux signal number (always 0 in local)
            example: '0:0'
        """
        return self.runner.run(cmd, run_directory=run_directory)

    def np_savetxt(self, path: str, arr: np.ndarray, fmt: str) -> None:
        self.files.np_savetxt(path, arr, fmt)

    def np_savez(self, path: str, **arrays) -> None:
        self.files.np_savez(path, **arrays)
