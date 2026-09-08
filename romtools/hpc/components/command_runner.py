"""Direct shell command execution, in local and remote flavors.

SLURM submission lives in slurm_job_manager.py; this module only runs commands.
"""

import shlex
import subprocess

import posixpath as ppath

from romtools.hpc.connection import Connection, Result
from romtools.hpc.logger import Logger


def run_local_bash(cmd: str) -> Result:
    res = subprocess.run(
        ["bash", "-c", cmd],
        cwd=".",
        capture_output=True,
        text=True
    )
    return Result(res.stdout, res.stderr, res.returncode)


class BaseCommandRunner:
    """
    Runs a shell command and reports the outcome as a Result.

    Arguments:
        config: The dispatcher's configuration dictionary
        logger: An instance of the Logger class for logging
    """

    def __init__(self, config: dict = None, logger: Logger = None):
        self.config = config if config is not None else {}
        self.logger = logger

    def run(self, cmd: str, run_directory: str = None) -> Result:
        raise NotImplementedError


class LocalCommandRunner(BaseCommandRunner):
    """Runs the command in a subshell on the local machine."""

    def run(self, cmd: str, run_directory: str = None) -> Result:
        full_cmd = f"cd {shlex.quote(run_directory)} && {cmd}" if run_directory else cmd
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True
        )

        return Result(result.stdout, result.stderr, result.returncode)


class RemoteCommandRunner(BaseCommandRunner):
    """
    Runs the command on the remote host, from the remote root or a directory under it.

    Arguments:
        connection: An established Connection to the remote host
        config: The dispatcher's configuration dictionary
        logger: An instance of the Logger class for logging
    """

    def __init__(self, connection: Connection, config: dict = None, logger: Logger = None):
        super().__init__(config=config, logger=logger)
        self.conn = connection

    def run(self, cmd: str, run_directory: str = None) -> Result:
        resolved_run_dir = ppath.join(self.config.get("remote_root"), run_directory) if run_directory else self.config.get("remote_root")
        remote_cmd = f"cd {shlex.quote(resolved_run_dir)} && {cmd}"
        res = self.conn.run(remote_cmd)
        if not res.ok:
            raise RuntimeError(f"Command failed ({cmd}): {res.stderr}")
        self.logger.debug(f"Executed command on remote host: {cmd}")
        return res
