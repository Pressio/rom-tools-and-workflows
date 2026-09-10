

import os
import shlex

from romtools.hpc.logger import Logger
from romtools.hpc.dispatchers.base_dispatcher import BaseDispatcher
from romtools.hpc.connection import Result, run_local_bash
from romtools.hpc.components.caller import LocalCaller
from romtools.hpc.components.file_manager import LocalFileManager
from romtools.hpc.components.slurm_job_manager import SlurmJobManager


class LocalDispatcher(BaseDispatcher):
    """
    Runs ROM workflows on the machine this process runs on.

    Paths address the local filesystem and commands run in a local bash shell
    rather than over SSH. That machine may itself be a cluster node, in which
    case submit_job() reaches the scheduler directly and results need no
    transferring.

    Arguments:
        campaign_directory: The directory jobs run in when given no run_directory
        logger: An instance of the Logger class for logging
        argv: Argument list to configure from instead of the real process argv.
            Pass [] to ignore the surrounding program's command line.
    """
    def __init__(self, campaign_directory: str = "hpctools", logger: Logger = None,
                 argv: list = None):
        super().__init__(campaign_directory=campaign_directory, logger=logger, argv=argv)

        self.caller = LocalCaller(config=self.config, logger=self.logger)
        self.files = LocalFileManager(config=self.config, logger=self.logger)

        self.slurm = SlurmJobManager(
            run_cmd=run_local_bash,
            config=self.config,
            logger=self.logger,
            campaign_directory=self.campaign_directory,
            files=self.files)

    def require_absolute_path(self, path: str) -> None:
        # Local run directories are addressed as given, and concurrent
        # evaluations run in worker processes that may change directory.
        if not os.path.isabs(path):
            raise ValueError(f"You must provide an absolute path (received: {path})")

    def run(self, cmd: str, run_directory: str = None) -> Result:
        """
        Run a command on the local machine, from run_directory if given and the
        current working directory otherwise.

        Returns a Result object (with stdout, stderr, exit_code, ok)
        """
        full_cmd = f"cd {shlex.quote(run_directory)} && {cmd}" if run_directory else cmd
        return run_local_bash(full_cmd)
