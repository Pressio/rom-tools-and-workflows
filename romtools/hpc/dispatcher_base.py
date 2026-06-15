
import numpy as np
import os

from romtools.hpc.util.logger import Logger
from romtools.hpc.configuration import Configuration


class DispatcherBase:
    def __init__(self, sampling_directory: str = "hpctools", logger: Logger = None):
        self.config = Configuration().to_dict()
        self.logger = logger if logger is not None else Logger(self.config["debug"])
        self.sampling_directory = sampling_directory

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

    def get_config(self, param: str = None) -> dict:
        if param is None:
            return self.config
        return self.config.get(param, None)
