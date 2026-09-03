
import numpy as np
import os

from romtools.hpc.util.logger import Logger
from romtools.hpc.configuration import Configuration
from romtools.hpc.connection import Result


class BaseDispatcher:
    def __init__(self, sampling_directory: str = "hpctools", logger: Logger = None, argv: list = None):
        self.config = Configuration(argv=argv).to_dict()
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

    def upload(self, run_directory) -> None:
        pass

    def put(self, local_path: str, remote_path: str) -> None:
        pass

    def get(self, remote_path: str, local_path: str) -> None:
        pass

    def path_exists(self, path: str) -> bool:
        pass

    def create_empty_dir(self, dir_name: str):
        pass

    def list_dir(self, path: str) -> list:
        pass

    def remove(self, path: str) -> None:
        pass

    def write_text(self, path: str, content: str) -> None:
        pass

    def dispatch(self, cmd: str, run_directory: str = None) -> Result:
        pass

    def np_savetxt(self, path: str, arr: np.ndarray, fmt: str) -> None:
        pass

    def np_savez(self, path: str, **arrays) -> None:
        pass

    def require_absolute_path(self, path: str) -> None:
        pass

    def require_relative_path(self, path: str) -> None:
        pass

    def require_supported_concurrency(self, concurrency: int) -> None:
        # Overridden by dispatchers that cannot run concurrent model evaluations
        pass

    def get_config(self, param: str = None) -> dict:
        if param is None:
            return self.config
        return self.config.get(param, None)
