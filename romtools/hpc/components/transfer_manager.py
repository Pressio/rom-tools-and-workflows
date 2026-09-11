"""Moving a campaign's input files to where it runs, and its results back."""

import glob
import os

import posixpath as ppath

from romtools.hpc.components.archive import (
    create_tarball,
    matches_everything,
    safe_extract_tar,
    validate_file_patterns,
)
from romtools.hpc.components.component import Component
from romtools.hpc.components.file_manager import BaseFileManager, RemoteFileManager
from romtools.hpc.connection import Connection, run_local_bash
from romtools.hpc.logger import Logger


class BaseTransferManager(Component):
    """
    Stages the files a campaign needs into its run directory, and brings its
    results back afterwards.

    Nothing moves by default: a dispatcher whose work already runs against the
    files in place needs neither half. Subclasses override whichever half
    genuinely has work to do.

    Arguments:
        files: The file manager whose resolve_path decides what paths mean
        campaign_directory: The campaign directory results are gathered into
        config: The dispatcher's configuration dictionary
        logger: An instance of the Logger class for logging

    Raises:
        ValueError: if the configured collect or upload patterns are invalid.
    """

    def __init__(self, *, files: BaseFileManager, config: dict = None,
                 logger: Logger = None, campaign_directory: str = None):
        super().__init__(config=config, logger=logger)
        self.files = files
        self.campaign_directory = campaign_directory

        # Patterns are this component's concern, so it validates its own
        self.collect_patterns = validate_file_patterns(self.config.get("collect"))
        self.upload_patterns = validate_file_patterns(self.config.get("upload"))

    def _archive_name(self) -> str:
        return f"dispatcher-transfer-{self.config.get('job_name')}.tar.gz"

    def upload(self, run_directory) -> None:
        """Put the configured upload patterns into the run directory."""

    def collect_results(self) -> None:
        """Gather a finished job's output into the local campaign directory."""


class LocalTransferManager(BaseTransferManager):
    """
    Copies the configured upload patterns from the current directory into the
    run directory.

    The work runs on this machine against these same files, so there is nothing
    to pack up and nothing to bring back: only upload does anything.
    """

    def upload(self, run_directory) -> None:
        if not self.upload_patterns:
            return

        destination = self.files.resolve_path(run_directory)
        sources = [s for s in self._expand_patterns() if not self._contains(s, destination)]
        if not sources:
            return

        for source in sources:
            self.files.put(source, os.path.join(destination, source))
        self.logger.debug(f"Copied {len(sources)} upload path(s) into {destination}", local=True)

    def _expand_patterns(self) -> list:
        """Expand the upload patterns against the current directory, in order and without repeats."""
        if matches_everything(self.upload_patterns):
            return sorted(os.listdir(os.curdir))

        sources = []
        for pattern in self.upload_patterns:
            matches = sorted(glob.glob(pattern))
            if not matches:
                self.logger.log(f"Warning: no files matched upload pattern {pattern!r}", local=True)
            sources.extend(match for match in matches if match not in sources)
        return sources

    @staticmethod
    def _contains(source: str, destination: str) -> bool:
        """True if copying source would recurse into the destination directory."""
        source = os.path.abspath(source)
        destination = os.path.abspath(destination)
        return os.path.commonpath([source, destination]) == source


class RemoteTransferManager(BaseTransferManager):
    """
    Moves the files a campaign needs onto the remote host, and its results back,
    in both directions through a single tar.gz archive.

    Arguments:
        connection: An established Connection to the remote host
        campaign_directory: The campaign directory, mirrored locally and remotely
        files: The remote file manager, whose resolve_path decides what paths mean
    """

    def __init__(self, connection: Connection, *, files: RemoteFileManager, config: dict = None,
                 logger: Logger = None, campaign_directory: str = None):
        super().__init__(files=files, config=config, logger=logger,
                         campaign_directory=campaign_directory)
        self.conn = connection

    def collect_results(self) -> None:
        """Collect results from remote HPC runs."""
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

        # Clean up remote archive. A stale archive is untidy, not fatal, so a
        # failure here must not stop us from unpacking what we already have.
        try:
            self.files.remove(archive_name)
        except RuntimeError as e:
            self.logger.log(f"Could not remove remote archive {archive_name}: {e}")

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
