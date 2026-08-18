
import shlex
import subprocess
import re

from typing import Optional, List
from collections.abc import Callable

from ..connection import Result

# Helper local cmd function to be passed into the ones below
def local_cmd(cmd: str):
    res = subprocess.run(
        ["bash", "-c", cmd],
        cwd=".",
        check=True,
        capture_output=True,
        text=True
    )
    return Result(res.stdout, res.stderr, 0)

def validate(collect_patterns: List[str]) -> Optional[List[str]]:
    """
    Validate and normalize file & directory patterns.

    Returns:
        - None if collect_patterns is not specified
        - List[str] of cleaned patterns otherwise

    Raises:
        ValueError: if any pattern is invalid.
    """
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
        raise ValueError("PATTERN VALIDATE: no valid patterns were provided.")

    return cleaned_patterns

def pack_results(log: Callable[[str], None], run_cmd: Callable[[str], Result], working_dir: str, archive_path: str, patterns: Optional[List[str]]) -> bool:
    """
    Create a tar.gz archive of results. Intended to work on either
    remote or local machine, depending on run_cmd.

    Archives only the validated files/directories/glob patterns
    relative to the working directory.

    Wildcard patterns are expanded relative to working_dir.
    Unmatched wildcard patterns are ignored with a warning.

    Returns True if compression was performed, False if it was skipped.
    """
    # Collect nothing
    if patterns is None or len(patterns) == 0:
        log(
            "SKIPPING TARRING: "
            "No files, directories, or glob patterns to bundle have been specified.")
        return False

    # Collect everything
    collect_all = ["*", "all", "everything", "any"]
    if any(p.lower() in collect_all for p in patterns):
        pack_cmd = (
            f"tar -czf {shlex.quote(archive_path)} "
            f"-C {shlex.quote(working_dir)} ."
        )
        pack_result = run_cmd(pack_cmd)
        if not pack_result.ok:
            raise RuntimeError(f"Archive failed: {pack_result.stderr}")

        log(f"Packed results into archive: {archive_path}")
        return True

    # Collect the specified files/directories/patterns
    resolved_paths = []
    warnings = []

    def is_glob_pattern(pattern: str) -> bool:
        return any(ch in pattern for ch in ("*", "?", "["))

    for pattern in patterns:
        if is_glob_pattern(pattern):
            expand_cmd = (
                f"cd {shlex.quote(working_dir)} && "
                f"for f in {pattern}; do "
                f'if [ -e "$f" ]; then printf "%s\\n" "$f"; fi; '
                f"done"
            )
            result = run_cmd(expand_cmd)

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
                f"cd {shlex.quote(working_dir)} && "
                f"test -e {shlex.quote(pattern)}"
            )
            result = run_cmd(test_cmd)

            if result.ok:
                if pattern not in resolved_paths:
                    resolved_paths.append(pattern)
            else:
                warnings.append(f"Requested collect path {pattern!r} does not exist")

    for warning in warnings:
        log(f"Warning while packing selected results: {warning}", local=True)

    if not resolved_paths:
        raise RuntimeError("No files matched the requested collect patterns.")

    path_args = " ".join(shlex.quote(p) for p in resolved_paths)
    pack_cmd = (
        f"tar -czf {shlex.quote(archive_path)} "
        f"-C {shlex.quote(working_dir)} -- {path_args}"
    )

    pack_result = run_cmd(pack_cmd)
    if not pack_result.ok:
        raise RuntimeError(f"Archive failed: {pack_result.stderr}")

    log(f"Packed results into archive: {archive_path}")

    return True


def safe_extract_tar(run_cmd: Callable[[str], Result], archive_path: str, target_dir: str) -> Result:
    """
    Safely extract a tarball on local or remote machine.
    Sends the proper bash command into given run_cmd.

    Deletes the archive after successful extraction.
    """

    cmd = f'''
set -euo pipefail

archive_path="{archive_path}"
target_dir="{target_dir}"

mkdir -p "$target_dir"

while IFS= read -r member; do
    case "$member" in
        /*|..|../*|*/..|*/../*)
            echo "Refusing to extract unsafe archive member: '$member'" >&2
            exit 1
            ;;
    esac
done < <(tar -tzf "$archive_path")

tar -xzf "$archive_path" -C "$target_dir"

rm -f -- "$archive_path"
'''
    return run_cmd(cmd)
