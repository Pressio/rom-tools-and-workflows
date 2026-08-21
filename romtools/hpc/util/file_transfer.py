
import shlex
import subprocess
import re

from typing import Optional, List
from collections.abc import Callable

from ..connection import Result

def validate_file_patterns(collect_patterns: List[str]) -> Optional[List[str]]:
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

def create_tarball(log: Callable[[str], None], run_cmd: Callable[[str], Result], working_dir: str, archive_path: str, patterns: Optional[List[str]]) -> None:
    """
    Create a tar.gz archive of results. Intended to work on either
    remote or local machine, depending on run_cmd.

    Archives only the validated files/directories/glob patterns
    relative to the working directory.

    Wildcard patterns are expanded relative to working_dir.
    Unmatched wildcard patterns are ignored with a warning.

    Raises:
        ValueError: If the source_dir or input arguments are empty/invalid.
        FileNotFoundError: If the validated paths do not exist.
        RuntimeError: If the tarball creation or compression fails.
    """
    # Collect nothing
    if patterns is None or len(patterns) == 0:
        raise ValueError(
            "SKIPPING TARRING: "
            "No files, directories, or glob patterns to bundle have been specified.")

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

        return

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
        log(f"Warning while packing selected results: {warning}")

    if not resolved_paths:
        raise FileNotFoundError("No files matched the requested collect patterns.")

    path_args = " ".join(shlex.quote(p) for p in resolved_paths)
    pack_cmd = (
        f"tar -czf {shlex.quote(archive_path)} "
        f"-C {shlex.quote(working_dir)} -- {path_args}"
    )

    pack_result = run_cmd(pack_cmd)
    if not pack_result.ok:
        raise RuntimeError(f"Archive failed: {pack_result.stderr}")


def safe_extract_tar(run_cmd: Callable[[str], Result], archive_path: str, target_dir: str) -> Result:
    """
    Safely extract a tarball on local or remote machine using POSIX sh.

    Deletes the archive after successful extraction.
    """

    script = r'''
set -eu

archive_path=$1
target_dir=$2

tmpdir=${TMPDIR:-/tmp}/safe_extract_tar.$$
i=0

while ! (umask 077 && mkdir "$tmpdir") 2>/dev/null; do
    i=$((i + 1))
    if [ "$i" -ge 10 ]; then
        printf '%s\n' "Failed to create temporary directory" >&2
        exit 1
    fi
    tmpdir=${TMPDIR:-/tmp}/safe_extract_tar.$$.$i
done

trap 'rm -rf "$tmpdir"' 0
trap 'rm -rf "$tmpdir"; exit 1' 1 2 3 15

members_file=$tmpdir/members

mkdir -p "$target_dir"

tar -tzf "$archive_path" > "$members_file"

while IFS= read -r member || [ -n "$member" ]; do
    case "$member" in
        /*|..|../*|*/..|*/../*)
            printf "Refusing to extract unsafe archive member: '%s'\n" "$member" >&2
            exit 1
            ;;
    esac
done < "$members_file"

tar -xzf "$archive_path" -C "$target_dir"

rm -f "$archive_path"
'''

    cmd = (
        "sh -c "
        + shlex.quote(script)
        + " safe_extract_tar "
        + shlex.quote(archive_path)
        + " "
        + shlex.quote(target_dir)
    )

    return run_cmd(cmd)
