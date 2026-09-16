"""File operations for the dispatchers, in local and remote flavors."""

from abc import ABC, abstractmethod
import base64
import io
import os
import posixpath as ppath
import shlex
import shutil
import tempfile

import numpy as np

from romtools.hpc.components.component import Component
from romtools.hpc.connection import Connection
from romtools.hpc.logger import Logger

# base64 inflates the ssh command 4/3 and one argv element is capped near 128 KiB
REMOTE_INLINE_WRITE_LIMIT = 32 * 1024


class BaseFileManager(Component, ABC):
    """Reads and writes files on whichever machine a dispatcher runs work on."""

    @abstractmethod
    def resolve_path(self, path: str = None) -> str:
        ...

    @abstractmethod
    def put(self, local_path: str, remote_path: str) -> None:
        ...

    @abstractmethod
    def get(self, remote_path: str, local_path: str) -> None:
        ...

    @abstractmethod
    def path_exists(self, path: str) -> bool:
        ...

    @abstractmethod
    def create_empty_dir(self, dir_name: str):
        ...

    @abstractmethod
    def list_dir(self, path: str) -> list:
        ...

    @abstractmethod
    def remove(self, path: str) -> None:
        ...

    @abstractmethod
    def remove_dir(self, path: str) -> None:
        """Remove a directory and its contents. A missing directory is not an error."""

    @abstractmethod
    def write_text(self, path: str, content: str) -> None:
        ...

    @abstractmethod
    def read_text(self, path: str) -> str:
        ...

    @abstractmethod
    def np_savetxt(self, path: str, arr: np.ndarray, fmt: str) -> None:
        ...

    @abstractmethod
    def np_savez(self, path: str, **arrays) -> None:
        ...


class LocalFileManager(BaseFileManager):
    """Operates directly on the local filesystem."""

    def resolve_path(self, path: str = None) -> str:
        """Local paths are used as given; relative ones follow the process cwd."""
        return path if path else os.curdir

    def _copy(self, src, dst):
        if os.path.exists(src) and os.path.exists(dst) and os.path.samefile(src, dst):
            self.logger.debug(f"{src} is already in place; skipping copy", local=True)
            return

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
        self._copy(remote_path, local_path)

    def put(self, local_path: str, remote_path: str) -> None:
        """Local 'put' is just a copy from local_path to remote_path."""
        self._copy(local_path, remote_path)

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

    def remove_dir(self, path: str) -> None:
        """Remove a directory and everything under it, if it exists."""
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return
        except OSError as e:
            raise RuntimeError(f"Failed to remove directory {path}: {e}") from e
        self.logger.debug(f"Removed directory {path}", local=True)

    def _make_parent_dir(self, path: str) -> None:
        """Every writer creates the directory it writes into, on both hosts."""
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    def write_text(self, path: str, content: str) -> None:
        self._make_parent_dir(path)
        with open(path, "w", encoding="utf-8") as text_file:
            text_file.write(content)
        self.logger.debug(f"Wrote file {path}", local=True)

    def read_text(self, path: str) -> str:
        """Return the contents of a text file."""
        with open(path, "r", encoding="utf-8") as text_file:
            return text_file.read()

    def np_savetxt(self, path: str, arr: np.ndarray, fmt: str) -> None:
        self._make_parent_dir(path)
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

        self._make_parent_dir(local_path)
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

    def __init__(self, connection: Connection, *, config: dict = None, logger: Logger = None):
        super().__init__(config=config, logger=logger)
        self.conn = connection

    def _make_parent_dir_cmd(self, resolved_path: str) -> str:
        """Every writer creates the directory it writes into, on both hosts."""
        parent_dir = ppath.dirname(resolved_path)
        return f"mkdir -p {shlex.quote(parent_dir)} && " if parent_dir else ""

    def resolve_path(self, path: str = None) -> str:
        """
        Resolve a remote path: absolute paths pass through, relative ones are
        taken under the remote root, and no path at all means the root itself.
        """
        remote_root = self.config.get("remote_root")
        if not path:
            return remote_root
        if ppath.isabs(path):
            return path
        return ppath.join(remote_root, path)

    def _write_staged(self, remote_path: str, data: bytes) -> None:
        """Content too large for one ssh command goes up as a staged file."""
        parent_dir = ppath.dirname(remote_path)
        if parent_dir:
            res = self.conn.run(f"mkdir -p {shlex.quote(parent_dir)}")
            if not res.ok:
                raise RuntimeError(f"Failed to create remote directory {parent_dir}: {res.stderr}")

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, ppath.basename(remote_path))
            with open(local_path, "wb") as staged:
                staged.write(data)
            self.conn.put(local_path, remote_path)

    def write_text(self, remote_path: str, content: str) -> None:
        remote_path = self.resolve_path(remote_path)
        data = content.encode("utf-8")

        if len(data) > REMOTE_INLINE_WRITE_LIMIT:
            self._write_staged(remote_path, data)
        else:
            # base64 so the content survives verbatim, whatever it contains
            payload = base64.b64encode(data).decode("ascii")
            cmd = (
                f"{self._make_parent_dir_cmd(remote_path)}"
                f"printf %s {shlex.quote(payload)} | base64 -d > {shlex.quote(remote_path)}"
            )
            res = self.conn.run(cmd)
            if not res.ok:
                raise RuntimeError(f"Failed to write remote file {remote_path}: {res.stderr}")

        self.logger.debug(f"Wrote remote file: {remote_path}")

    def put(self, local_path: str, remote_path: str) -> None:
        remote_path = self.resolve_path(remote_path)
        self.conn.put(local_path, remote_path)
        self.logger.debug(f"Uploaded local file {local_path} to {self.conn.host}:{remote_path}")

    def get(self, remote_path: str, local_path: str) -> None:
        remote_path = self.resolve_path(remote_path)
        self.conn.get(remote_path, local_path)
        self.logger.debug(f"Downloaded remote file {self.conn.host}:{remote_path} to local path {local_path}")

    def path_exists(self, path: str) -> bool:
        remote_path = self.resolve_path(path)
        result = self.conn.run(f"test -e {shlex.quote(remote_path)}")
        return result.ok

    def create_empty_dir(self, dir_name: str):
        remote_dir = self.resolve_path(dir_name)
        result = self.conn.run(f"mkdir -p {shlex.quote(remote_dir)}")
        if not result.ok:
            raise RuntimeError(f"Failed to create remote directory {remote_dir}: {result.stderr}")
        self.logger.debug(f"Created remote directory: {remote_dir}")

    def list_dir(self, path: str) -> list:
        remote_path = self.resolve_path(path)
        res = self.conn.run(f"ls -1 {shlex.quote(remote_path)}")
        if not res.ok:
            return []
        return [entry for entry in res.stdout.splitlines() if entry]

    def remove(self, path: str) -> None:
        remote_path = self.resolve_path(path)
        res = self.conn.run(f"rm -f {shlex.quote(remote_path)}")
        if not res.ok:
            raise RuntimeError(f"Failed to remove remote file {remote_path}: {res.stderr}")
        self.logger.debug(f"Removed remote file: {remote_path}")

    def remove_dir(self, path: str) -> None:
        """Remove a remote directory and everything under it."""
        remote_path = self.resolve_path(path)
        res = self.conn.run(f"rm -rf {shlex.quote(remote_path)}")
        if not res.ok:
            raise RuntimeError(f"Failed to remove remote directory {remote_path}: {res.stderr}")
        self.logger.debug(f"Removed remote directory: {remote_path}")

    def read_text(self, path: str) -> str:
        """Return the contents of a remote text file."""
        remote_path = self.resolve_path(path)
        res = self.conn.run(f"cat {shlex.quote(remote_path)}")
        if not res.ok:
            raise RuntimeError(f"Failed to read remote file {remote_path}: {res.stderr}")
        return res.stdout

    def np_savetxt(self, path: str, arr: np.ndarray, fmt: str) -> None:
        buffer = io.StringIO()
        np.savetxt(buffer, arr, fmt=fmt)
        self.write_text(path, buffer.getvalue())
        self.logger.debug(f"Saved array to path {path}", local=False)

    def np_savez(self, path: str, **arrays) -> None:
        """Write multiple arrays to a .npz file, staged locally then uploaded."""
        remote_path = ppath.normpath(path)
        if not remote_path.endswith(".npz"):
            remote_path += ".npz"

        remote_dir = ppath.dirname(remote_path)
        if remote_dir:
            self.create_empty_dir(remote_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, ppath.basename(remote_path))
            np.savez(local_path, **arrays)
            self.put(local_path, remote_path)

        self.logger.debug(f"Saved arrays to path {remote_path}")
