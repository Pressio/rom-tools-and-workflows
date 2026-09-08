"""Bundling, transferring, and unpacking of run files between local and remote."""

import os
import shlex

import posixpath as ppath

from romtools.hpc.archive import create_tarball, safe_extract_tar
from romtools.hpc.connection import Connection
from romtools.hpc.logger import Logger
from romtools.hpc.components.command_runner import run_local_bash


class TransferManager:
    """
    Moves the files a campaign needs onto the remote host, and its results back.

    Both directions go through a single tar.gz archive, built from the
    configured upload and collect patterns.

    Arguments:
        connection: An established Connection to the remote host
        config: The dispatcher's configuration dictionary
        logger: An instance of the Logger class for logging
        campaign_directory: The campaign directory, mirrored locally and remotely
        collect_patterns: Validated patterns to retrieve from the remote run directory
        upload_patterns: Validated patterns to send to the remote run directory
    """

    def __init__(self, connection: Connection, config: dict = None, logger: Logger = None,
                 campaign_directory: str = None, collect_patterns: list = None,
                 upload_patterns: list = None):
        self.conn = connection
        self.config = config if config is not None else {}
        self.logger = logger
        self.campaign_directory = campaign_directory
        self.collect_patterns = collect_patterns
        self.upload_patterns = upload_patterns

    def __archive_name(self) -> str:
        return f"dispatcher-transfer-{self.config.get('job_name')}.tar.gz"

    def collect_results(self) -> None:
        """
        Collect results from remote HPC runs.
        """
        remote_campaign_dir = ppath.join(self.config.get("remote_root"), self.campaign_directory)
        self.logger.debug(
            f"Transferring results from {self.conn.host}:{remote_campaign_dir} -> {self.campaign_directory}",
            local=True,
        )

        archive_name = self.__archive_name()
        remote_archive_path = ppath.join(self.config.get("remote_root"), archive_name)

        try:
            create_tarball(lambda msg: self.logger.log(msg), lambda cmd: self.conn.run(cmd), remote_campaign_dir, remote_archive_path, self.collect_patterns)
        except Exception as e:
            self.logger.log(f"Failed to create tarball on {self.conn.host}: {e}")
            return

        # Copy remote archive to local
        self.conn.get(remote_archive_path, archive_name)
        self.logger.debug(f"Copied remote archive to local: {archive_name}")

        # Clean up remote archive
        self.conn.run(f"rm -f {shlex.quote(remote_archive_path)}")

        # Unpack local archive into local campaign directory
        os.makedirs(self.campaign_directory, exist_ok=True)
        res = safe_extract_tar(run_local_bash, archive_name, os.path.abspath(self.campaign_directory))
        if not res.ok:
            self.logger.log("Results failed to extract.", local=True)
        else:
            self.logger.log(f"Results collected in {self.campaign_directory}", local=True)

    def upload(self, run_directory) -> None:
        if not self.upload_patterns:
            return

        remote_root = self.config.get("remote_root")
        tar_name = self.__archive_name()
        create_tarball(lambda msg: self.logger.log(msg, local=True), run_local_bash, ".", tar_name, self.upload_patterns)
        tar_path = f"{ppath.join(remote_root, run_directory)}/{tar_name}"
        try:
            self.conn.put(tar_name, tar_path)
            self.logger.debug(f"Uploaded local file {tar_name} to {self.conn.host}:{tar_path}")
        except Exception as e:
            raise RuntimeError(f"File transfer failed on upload: {e}")
        finally:
            os.remove(tar_name)
        res = safe_extract_tar(lambda cmd: self.conn.run(cmd), tar_path, f"{ppath.join(remote_root, run_directory)}")
        if not res.ok:
            raise RuntimeError(f"Extraction failed on remote! {res.stderr}")
