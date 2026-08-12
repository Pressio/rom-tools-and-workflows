import os
import re
import shlex
import tarfile
import shutil

import posixpath as ppath
from typing import List, Optional

from romtools.hpc.util.logger import Logger
from romtools.hpc.connection import Connection

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
    # Packing and unpacking remote results
    # ------------------------------------------------------------------

    def __pack_remote_results(self, remote_sampling_dir: str, remote_archive_path: str) -> None:
        """
        Create a tar.gz archive of remote results.

        If self.config["collect"] is None, archives the entire sampling directory.
        Otherwise, archives only the validated files/directories/glob patterns
        relative to the sampling directory.

        Wildcard patterns are expanded remotely relative to remote_sampling_dir.
        Unmatched wildcard patterns are ignored with a warning.

        Returns True if collection was performed, False if collection was skipped.
        """
        # Collect nothing
        if self.patterns is None or len(self.patterns) == 0:
            self.logger.log(
                "SKIPPING RESULTS COLLECTION: "
                "Specify files, directories, or glob patterns to bring back to your "
                "local host by setting the 'collect' field in your configuration.")
            return False

        # Collect everything
        collect_all = ["*", "all", "everything", "any"]
        if any(p.lower() in collect_all for p in self.patterns):
            pack_cmd = (
                f"tar -czf {shlex.quote(remote_archive_path)} "
                f"-C {shlex.quote(remote_sampling_dir)} ."
            )
            pack_result = self.conn.run(pack_cmd)
            if not pack_result.ok:
                raise RuntimeError(f"Remote result archive failed: {pack_result.stderr}")

            self.logger.log(f"Packed remote results into archive: {remote_archive_path}")
            return True

        # Collect the specified files/directories/patterns
        resolved_paths = []
        warnings = []

        def is_glob_pattern(pattern: str) -> bool:
            return any(ch in pattern for ch in ("*", "?", "["))

        for pattern in self.patterns:
            if is_glob_pattern(pattern):
                expand_cmd = (
                    f"cd {shlex.quote(remote_sampling_dir)} && "
                    f"for f in {pattern}; do "
                    f'if [ -e "$f" ]; then printf "%s\\n" "$f"; fi; '
                    f"done"
                )
                result = self.conn.run(expand_cmd)

                if not result.ok:
                    warnings.append(f"Failed to evaluate collect pattern {pattern!r}: {result.stderr.strip()}")
                    continue

                matches = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if not matches:
                    warnings.append(f"No files matched collect pattern {pattern!r}")
                    continue

                for match in matches:
                    if match not in resolved_paths:
                        resolved_paths.append(match)
            else:
                test_cmd = (
                    f"cd {shlex.quote(remote_sampling_dir)} && "
                    f"test -e {shlex.quote(pattern)}"
                )
                result = self.conn.run(test_cmd)

                if result.ok:
                    if pattern not in resolved_paths:
                        resolved_paths.append(pattern)
                else:
                    warnings.append(f"Requested collect path {pattern!r} does not exist")

        for warning in warnings:
            self.logger.log(f"Warning while collecting selected results: {warning}", local=True)

        if not resolved_paths:
            raise RuntimeError("No files matched the requested collect patterns.")

        path_args = " ".join(shlex.quote(p) for p in resolved_paths)
        pack_cmd = (
            f"tar -czf {shlex.quote(remote_archive_path)} "
            f"-C {shlex.quote(remote_sampling_dir)} -- {path_args}"
        )

        pack_result = self.conn.run(pack_cmd)
        if not pack_result.ok:
            raise RuntimeError(f"Remote result archive failed: {pack_result.stderr}")

        self.logger.log(f"Packed remote results into archive: {remote_archive_path}")

        return True

    @staticmethod
    def __safe_extract_tar(archive_path: str, target_dir: str) -> None:
        target_abs = os.path.abspath(target_dir)
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                member_abs = os.path.abspath(os.path.join(target_abs, member.name))
                if os.path.commonpath([target_abs, member_abs]) != target_abs:
                    raise RuntimeError(
                        f"Refusing to extract unsafe archive member: {member.name!r}"
                    )
            tar.extractall(path=target_abs)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def remove_local_dir(self):
        if os.path.exists(self.sampling_directory) and os.path.isdir(self.sampling_directory):
            self.logger.log(f"Overwriting existing results in {self.sampling_directory}")
            shutil.rmtree(self.sampling_directory)

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

        do_collection = self.__pack_remote_results(remote_sampling_dir, remote_archive_path)

        if not do_collection:
            return

        # Copy remote archive to local
        self.conn.get(remote_archive_path, archive_name)
        self.logger.log(f"Copied remote archive to local: {archive_name}")

        # Unpack local archive into local sampling directory
        os.makedirs(self.sampling_directory, exist_ok=True)
        self.__safe_extract_tar(archive_name, self.sampling_directory)
        self.logger.log(f"Results collected in {self.sampling_directory}", local=True)

        # Clean up both archives
        os.remove(archive_name)
        self.conn.run(f"rm -f {shlex.quote(remote_archive_path)}")
        self.logger.debug("Cleaned up archives.")
