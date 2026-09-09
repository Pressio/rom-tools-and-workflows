import os
import json
import uuid
import shlex
import tempfile

import numpy as np
import posixpath as ppath

from .call_runner import (
    CALL_INPUT_JSON,
    CALL_INPUT_NPZ,
    CALL_OUTPUT_JSON,
    CALL_OUTPUT_NPZ,
    CALL_RUNNER,
    build_call_runner,
    pack,
    resolve_target,
    unpack,
    working_directory,
)
from romtools.hpc.connection import Connection
from romtools.hpc.logger import Logger


def build_call_command(python_setup: str, python_command: str, call_id: str, target: str) -> str:
    """
    Build the shell command that runs a staged call runner.

    Runner paths are relative to the run directory the command executes from,
    since the remote root itself may be relative. The body is a brace group so
    that a failed `cd` skips it entirely instead of running the payload in the
    login directory without errexit.
    """
    runner = shlex.quote(ppath.join(call_id, CALL_RUNNER))
    runner_args = " ".join(
        shlex.quote(ppath.join(call_id, name))
        for name in (CALL_INPUT_JSON, CALL_INPUT_NPZ, CALL_OUTPUT_JSON, CALL_OUTPUT_NPZ)
    )

    lines = ["{", "set -e"]
    if python_setup:
        lines.append(python_setup)
    lines.append(f"{python_command} {runner} {shlex.quote(target)} {runner_args}")
    lines.append("}")

    return "\n".join(lines)


class BaseCaller:
    """
    Executes a Python callable named by a "module:qualname" target string.

    Arguments and return values may contain numpy arrays, tuples, lists, and
    dictionaries thereof.

    Arguments:
        config: The dispatcher's configuration dictionary
        logger: An instance of the Logger class for logging
    """

    def __init__(self, config: dict = None, logger: Logger = None):
        self.config = config if config is not None else {}
        self.logger = logger

    def call(self, target: str, *args, run_directory: str = None, **kwargs):
        raise NotImplementedError


class LocalCaller(BaseCaller):
    """Imports and runs the target in the current process."""

    def call(self, target: str, *args, run_directory: str = None, **kwargs):
        if run_directory is None:
            return resolve_target(target)(*args, **kwargs)

        with working_directory(run_directory):
            return resolve_target(target)(*args, **kwargs)


class RemoteCaller(BaseCaller):
    """
    Runs the target on a remote host over an SSH connection.

    Arguments:
        connection: An established Connection to the remote host
        config: The dispatcher's configuration dictionary
        logger: An instance of the Logger class for logging
    """

    def __init__(self, connection: Connection, config: dict = None, logger: Logger = None, files=None):
        super().__init__(config=config, logger=logger)
        self.conn = connection
        self.files = files
        self.python_setup = self.config.get("python_setup")
        self.python_command = self.config.get("python_command") or "python3"

    def call(self, target: str, *args, run_directory: str = None, **kwargs):
        call_id = f".dispatcher_call_{uuid.uuid4().hex}"
        call_dir = ppath.join(run_directory, call_id) if run_directory else call_id

        self._create_call_directory(call_dir)
        try:
            with tempfile.TemporaryDirectory() as staging_dir:
                self._upload_inputs(staging_dir, call_dir, args, kwargs)
                self._run_target(run_directory, call_id, target)
                result = self._download_result(staging_dir, call_dir)
        finally:
            self._remove_call_directory(call_dir)

        self.logger.log(f"Executed {target} on remote host.")
        return result

    def _create_call_directory(self, call_dir: str) -> None:
        self.files.create_empty_dir(call_dir)

    def _remove_call_directory(self, call_dir: str) -> None:
        try:
            self.files.remove_dir(call_dir)
        except RuntimeError as e:
            self.logger.log(f"Failed to clean up remote call directory {call_dir}: {e}")

    def _upload_inputs(self, staging_dir: str, call_dir: str, args: tuple, kwargs: dict) -> None:
        arrays = {}
        payload = {"args": pack(args, arrays), "kwargs": pack(kwargs, arrays)}

        runner_path = os.path.join(staging_dir, CALL_RUNNER)
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(build_call_runner())

        input_json_path = os.path.join(staging_dir, CALL_INPUT_JSON)
        with open(input_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        input_npz_path = os.path.join(staging_dir, CALL_INPUT_NPZ)
        np.savez(input_npz_path, **arrays)

        for local_path, name in ((runner_path, CALL_RUNNER),
                                 (input_json_path, CALL_INPUT_JSON),
                                 (input_npz_path, CALL_INPUT_NPZ)):
            self.files.put(local_path, ppath.join(call_dir, name))

        self.logger.debug(f"Staged call inputs in {self.conn.host}:{call_dir}")

    def _run_target(self, run_directory: str, call_id: str, target: str) -> None:
        cmd = build_call_command(self.python_setup, self.python_command, call_id, target)
        res = self.conn.run(f"cd {shlex.quote(self.files.resolve_path(run_directory))} && {cmd}")
        if not res.ok:
            raise RuntimeError(
                f"Remote call of {target} failed (exit code {res.exit_code}).\n"
                f"STDOUT:\n{res.stdout}\n"
                f"STDERR:\n{res.stderr}"
            )

    def _download_result(self, staging_dir: str, call_dir: str):
        output_json_path = os.path.join(staging_dir, CALL_OUTPUT_JSON)
        output_npz_path = os.path.join(staging_dir, CALL_OUTPUT_NPZ)

        self.files.get(ppath.join(call_dir, CALL_OUTPUT_JSON), output_json_path)
        self.files.get(ppath.join(call_dir, CALL_OUTPUT_NPZ), output_npz_path)

        with open(output_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        with np.load(output_npz_path, allow_pickle=False) as arrays:
            return unpack(payload["result"], arrays)
