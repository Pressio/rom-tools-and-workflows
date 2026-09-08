"""File operations for the dispatchers, in local and remote flavors."""

import io
import os
import shlex
import shutil
import tempfile

import numpy as np
import posixpath as ppath

from romtools.hpc.connection import Connection
from romtools.hpc.logger import Logger


class BaseFileManager:
    """
    Reads and writes files on whichever machine a dispatcher runs work on.

    Arguments:
        config: The dispatcher's configuration dictionary
        logger: An instance of the Logger class for logging
    """

    def __init__(self, config: dict = None, logger: Logger = None):
        self.config = config if config is not None else {}
        self.logger = logger

    def put(self, local_path: str, remote_path: str) -> None:
        raise NotImplementedError

    def get(self, remote_path: str, local_path: str) -> None:
        raise NotImplementedError

    def path_exists(self, path: str) -> bool:
        raise NotImplementedError

    def create_empty_dir(self, dir_name: str):
        raise NotImplementedError

    def list_dir(self, path: str) -> list:
        raise NotImplementedError

    def remove(self, path: str) -> None:
        raise NotImplementedError

    def write_text(self, path: str, content: str) -> None:
        raise NotImplementedError

    def np_savetxt(self, path: str, arr: np.ndarray, fmt: str) -> None:
        raise NotImplementedError

    def np_savez(self, path: str, **arrays) -> None:
        raise NotImplementedError


class LocalFileManager(BaseFileManager):
    """Operates directly on the local filesystem."""

    def __copy(self, src, dst):
        dst_dir = os.path.dirname(dst)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)

        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

        self.logger.debug(f"Copied {src} to {dst}", local=True)

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

    def list_dir(self, path: str) -> list:
        if not os.path.isdir(path):
            return []
        return os.listdir(path)

    def remove(self, path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
            self.logger.debug(f"Removed {path}", local=True)

    def write_text(self, path: str, content: str) -> None:
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as text_file:
            text_file.write(content)
        self.logger.debug(f"Wrote file {path}", local=True)

    def np_savetxt(self, path: str, arr: np.ndarray, fmt: str) -> None:
        np.savetxt(path, arr, fmt=fmt)
        self.logger.debug(f"Saved array to path {path}", local=True)

    def np_savez(self, path: str, **arrays) -> None:
        """
        Write multiple arrays to a .npz file.
        The .npz file is written directly to the specified path.
        """
        local_path = os.path.normpath(path)
        if not local_path.endswith(".npz"):
            local_path += ".npz"

        np.savez(local_path, **arrays)
        self.logger.debug(f"Saved arrays to path {local_path}", local=True)


class RemoteFileManager(BaseFileManager):
    """
    Operates on the remote filesystem over an SSH connection.

    Relative paths are resolved against the configured remote root.

    Arguments:
        connection: An established Connection to the remote host
        config: The dispatcher's configuration dictionary
        logger: An instance of the Logger class for logging
    """

    def __init__(self, connection: Connection, config: dict = None, logger: Logger = None):
        super().__init__(config=config, logger=logger)
        self.conn = connection

    def __resolve_remote_path(self, remote_path: str, preserve_relative: bool = False) -> str:
        if ppath.isabs(remote_path) or preserve_relative:
            return remote_path
        return ppath.join(self.config.get("remote_root"), remote_path)

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
        self.logger.debug(f"Wrote remote file: {remote_path}")

    def __create_remote_directory(self, remote_dir: str, base_dir = False) -> None:
        remote_dir = self.__resolve_remote_path(remote_dir, preserve_relative=base_dir)
        result = self.conn.run(f"mkdir -p {shlex.quote(remote_dir)}")
        if not result.ok:
            raise RuntimeError(f"Failed to create remote directory {remote_dir}: {result.stderr}")
        self.logger.debug(f"Created remote directory: {remote_dir}")

    def put(self, local_path: str, remote_path: str) -> None:
        remote_path = self.__resolve_remote_path(remote_path)
        self.conn.put(local_path, remote_path)
        self.logger.debug(f"Uploaded local file {local_path} to {self.conn.host}:{remote_path}")

    def get(self, remote_path: str, local_path: str) -> None:
        remote_path = self.__resolve_remote_path(remote_path)
        self.conn.get(remote_path, local_path)
        self.logger.debug(f"Downloaded remote file {self.conn.host}:{remote_path} to local path {local_path}")

    def path_exists(self, path: str) -> bool:
        remote_path = self.__resolve_remote_path(path)
        result = self.conn.run(f"test -e {shlex.quote(remote_path)}")
        return result.ok

    def create_empty_dir(self, dir_name: str):
        self.__create_remote_directory(dir_name)

    def list_dir(self, path: str) -> list:
        remote_path = self.__resolve_remote_path(path)
        res = self.conn.run(f"ls -1 {shlex.quote(remote_path)}")
        if not res.ok:
            return []
        return [entry for entry in res.stdout.splitlines() if entry]

    def remove(self, path: str) -> None:
        remote_path = self.__resolve_remote_path(path)
        res = self.conn.run(f"rm -f {shlex.quote(remote_path)}")
        if not res.ok:
            raise RuntimeError(f"Failed to remove remote file {remote_path}: {res.stderr}")
        self.logger.debug(f"Removed remote file: {remote_path}")

    def write_text(self, path: str, content: str) -> None:
        self.__write_text(path, content)

    def np_savetxt(self, path: str, arr: np.ndarray, fmt: str) -> None:
        buffer = io.StringIO()
        np.savetxt(buffer, arr, fmt=fmt)
        self.__write_text(path, buffer.getvalue())
        self.logger.debug(f"Saved array to path {path}", local=False)

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

        self.logger.debug(f"Saved arrays to path {remote_path}", local=(not self.conn))
