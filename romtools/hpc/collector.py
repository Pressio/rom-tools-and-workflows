import os
import re
import shlex

import posixpath as ppath
from typing import List, Optional

from romtools.hpc.util.logger import Logger
from romtools.hpc.connection import Connection
from romtools.hpc.util.file_wrangler import pack_results, safe_extract_tar, local_cmd

class Collector:
    """
    Handles collecting results from remote HPC runs.

    The collect specification can be:
      - None: do not retrieve anything
      - "all", "*", "any": retrieves everything
      - A comma-separated string of files/directories/globs to retrieve
      - A list of strings specifying files/directories/globs to retrieve
    """

    def __init__(
        self,
        conn: Connection,
        config: dict,
        sampling_directory: str,
        logger: Logger = None,
    ):
        # Initialize member variables
        self.conn = conn
        self.config = config
        self.sampling_directory = os.path.basename(sampling_directory)
        self.logger = logger or Logger()

        # Validate the collect patterns
        self.patterns = self.__validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __validate(self) -> Optional[List[str]]:
        """
        Validate and normalize config.collect.

        Returns:
            - None if collect is not specified
            - list[str] of cleaned patterns otherwise

        Raises:
            ValueError: if any pattern is invalid or if collect is specified but empty.
        """
        collect_patterns = self.config.get("collect")
        if collect_patterns is None:
            return None

        cleaned_patterns = []
        forbidden_chars = {"\n", "\r"}

        for pattern in collect_patterns:
            if not pattern or not pattern.strip():
                continue

            p = pattern.strip()

            if p.startswith("-"):
                raise ValueError(
                    f"Invalid collect pattern {p!r}: patterns may not begin with '-'."
                )

            if any(ch in p for ch in forbidden_chars):
                raise ValueError(
                    f"Invalid collect pattern {p!r}: contains forbidden characters."
                )

            # Restrict patterns to path and glob characters to avoid shell injection.
            if not re.fullmatch(r"[A-Za-z0-9_./*?\[\]\-]+", p):
                raise ValueError(
                    f"Invalid collect pattern {p!r}: contains unsupported characters."
                )

            cleaned_patterns.append(p)

        if not cleaned_patterns:
            raise ValueError("collect was specified, but no valid patterns were provided.")

        return cleaned_patterns

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def collect_results(self) -> None:
        """
        Transfer results from the remote sampling directory to the local sampling directory.

        Behavior:
        - If config.collect is None, archive and retrieve the entire remote sampling directory.
        - Otherwise, archive and retrieve only the specified files/directories/glob patterns,
        interpreted relative to the remote sampling directory.
        """
        remote_sampling_dir = ppath.join(self.config.get("remote_root"), self.sampling_directory)
        self.logger.log(
            f"Transferring results from {self.conn.host}:{remote_sampling_dir} -> {self.sampling_directory}",
            local=True,
        )

        archive_name = f"{self.config.get('job_name')}.tar.gz"
        remote_archive_path = ppath.join(self.config.get("remote_root"), archive_name)

        do_collection = pack_results(self.logger, lambda cmd: self.conn.run(cmd), remote_sampling_dir, remote_archive_path, self.patterns)

        if not do_collection:
            return

        # Copy remote archive to local
        self.conn.get(remote_archive_path, archive_name)
        self.logger.log(f"Copied remote archive to local: {archive_name}")

        # Clean up remote archive
        self.conn.run(f"rm -f {shlex.quote(remote_archive_path)}")
        self.logger.debug("Cleaned up remote archive.")

        # Unpack local archive into local sampling directory
        os.makedirs(self.sampling_directory, exist_ok=True)
        safe_extract_tar(local_cmd, archive_name, os.path.abspath(self.sampling_directory))
        self.logger.log(f"Results collected in {self.sampling_directory}", local=True)
