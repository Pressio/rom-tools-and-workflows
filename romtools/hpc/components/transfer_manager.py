"""Bundling, transferring, and unpacking of run files between local and remote."""

import os

import posixpath as ppath

from romtools.hpc.components.archive import create_tarball, safe_extract_tar, validate_file_patterns
from romtools.hpc.connection import Connection, run_local_bash
from romtools.hpc.logger import Logger
from romtools.hpc.components.file_manager import RemoteFileManager


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
        files: The remote file manager, whose resolve_path decides what paths mean

    Raises:
        ValueError: if the configured collect or upload patterns are invalid.
    """

    def __init__(self, connection: Connection, *, config: dict = None, logger: Logger = None,
                 campaign_directory: str = None, files: RemoteFileManager = None):
        self.conn = connection
        self.config = config if config is not None else {}
        self.logger = logger
        self.campaign_directory = campaign_directory
        self.files = files

        # Patterns are this component's concern, so it validates its own
        self.collect_patterns = validate_file_patterns(self.config.get("collect"))
        self.upload_patterns = validate_file_patterns(self.config.get("upload"))

    def _archive_name(self) -> str:
        return f"dispatcher-transfer-{self.config.get('job_name')}.tar.gz"

    def collect_results(self) -> None:
        """
        Collect results from remote HPC runs.
        """
        remote_campaign_dir = self.files.resolve_path(self.campaign_directory)
        self.logger.debug(
            f"Transferring results from {self.conn.host}:{remote_campaign_dir} -> {self.campaign_directory}",
            local=True,
        )

        archive_name = self._archive_name()
        remote_archive_path = self.files.resolve_path(archive_name)

        try:
            create_tarball(lambda msg: self.logger.log(msg), lambda cmd: self.conn.run(cmd), remote_campaign_dir, remote_archive_path, self.collect_patterns)
        except Exception as e:
            self.logger.log(f"Failed to create tarball on {self.conn.host}: {e}")
            return

        # Copy remote archive to local
        self.files.get(archive_name, archive_name)
        self.logger.debug(f"Copied remote archive to local: {archive_name}")

        # Clean up remote archive
        self.files.remove(archive_name)

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

        tar_name = self._archive_name()
        create_tarball(lambda msg: self.logger.log(msg, local=True), run_local_bash, ".", tar_name, self.upload_patterns)
        remote_tar_path = ppath.join(run_directory, tar_name) if run_directory else tar_name
        try:
            self.files.put(tar_name, remote_tar_path)
            self.logger.debug(f"Uploaded local file {tar_name} to {self.conn.host}:{remote_tar_path}")
        except Exception as e:
            raise RuntimeError(f"File transfer failed on upload: {e}")
        finally:
            os.remove(tar_name)
        res = safe_extract_tar(lambda cmd: self.conn.run(cmd),
                               self.files.resolve_path(remote_tar_path),
                               self.files.resolve_path(run_directory))
        if not res.ok:
            raise RuntimeError(f"Extraction failed on remote! {res.stderr}")
