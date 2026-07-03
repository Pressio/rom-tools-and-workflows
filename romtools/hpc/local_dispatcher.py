

import os
import subprocess
import numpy as np

from romtools.hpc.connection import Result
from romtools.hpc.util.logger import Logger
from romtools.hpc.dispatcher_base import DispatcherBase

class LocalDispatcher(DispatcherBase):
    """
    LocalDispatcher is a subclass of DispatcherBase that implements the core functionality
    for dispatching ROM workflows on the local machine. It overrides methods to set up
    directories and execute commands without SSH, making it suitable for local execution.
    """
    def __init__(self, sampling_directory: str = "hpctools", logger: Logger = None):
        super().__init__(sampling_directory=sampling_directory, logger=logger)

    def __copy(self, src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            os.system(f"cp -r {src} {dst}")
        else:
            os.system(f"cp {src} {dst}")
        self.logger.log(f"Copied {src} to {dst}", local=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, remote_path: str, local_path: str) -> None:
        """Local 'get' is just a copy from remote_path to local_path."""
        self.__copy(remote_path, local_path)

    def put(self, local_path: str, remote_path: str) -> None:
        """Local 'put' is just a copy from local_path to remote_path."""
        self.__copy(local_path, remote_path)

    def path_exists(self, path: str) -> bool:
        return os.path.exists(path)

    def create_empty_dir(self, dir_name: str):
        os.makedirs(dir_name, exist_ok=True)

    def dispatch(self, cmd: str, run_directory: str = None) -> None:
        result = subprocess.run(
            f"cd {run_directory} && {cmd}",
            capture_output=True,
            text=True
        )
        return Result(result.stdout, result.stderr, result.returncode)

    def np_savetxt(self, path: str, arr: np.ndarray, fmt: str) -> None:
        np.savetxt(path, arr, fmt=fmt)
        self.logger.log(f"Saved array to path {path}", local=True)

    def np_savez(self, path: str, **arrays) -> None:
        """
        Write multiple arrays to a .npz file.
        The .npz file is written directly to the specified path.
        """
        local_path = os.path.normpath(path)
        if not local_path.endswith(".npz"):
            local_path += ".npz"

        np.savez(local_path, **arrays)
        final_path = local_path

        self.logger.log(f"Saved arrays to path {final_path}", local=True)
