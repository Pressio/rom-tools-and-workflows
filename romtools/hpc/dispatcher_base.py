
import numpy as np
import os

from romtools.hpc.dispatcher_config import DispatcherConfig
from romtools.hpc.logger import Logger


class DispatcherBase:
    def __init__(self, logger: Logger = None, sampling_directory: str = "hpctools"):
        self.logger = logger if logger is not None else Logger()
        self.sampling_directory = sampling_directory
        self.config = DispatcherConfig(self.logger)

        # Create the local output directory
        os.makedirs(self.sampling_directory, exist_ok=True)
        self.logger.log(f"Local sampling directory: {self.sampling_directory}", local=True)

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
    # Public API
    # ------------------------------------------------------------------

    def put(self, local_path: str, remote_path: str) -> None:
        pass

    def get(self, remote_path: str, local_path: str) -> None:
        pass

    def path_exists(self, path: str) -> bool:
        pass

    def create_empty_dir(self, dir_name: str):
        pass

    def dispatch(self, cmd: str, run_directory: str = None) -> str:
        pass

    def np_savetxt(self, path: str, arr: np.ndarray, fmt: str) -> None:
        pass

    def np_savez(self, path: str, **arrays) -> None:
        pass
