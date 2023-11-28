import sys
import subprocess
import re
import argparse


def parse_diff_output(changed_files):
    # Regex to capture filename and the line numbers of the changes
    file_pattern = re.compile(r"^\+\+\+ b/(.*?)$", re.MULTILINE)
    line_pattern = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)

    files = {}
    for match in file_pattern.finditer(changed_files):
        file_name = match.group(1)

        # Filtering for python files in test directories
        if file_name.endswith((".py")) and any(
            included in file_name
            for included in [
                "test/",
                "tests/",
                "test_",
                "example/",
                "examples/"
            ]
        ):
            # Find the lines that changed for this file
            lines_start_at = match.end()
            next_file_match = file_pattern.search(changed_files, pos=match.span(0)[1])

            # Slice out the part of the diff that pertains to this file
            file_diff = changed_files[
                lines_start_at : next_file_match.span(0)[0] if next_file_match else None
            ]

            # Extract line numbers of the changes
            changed_lines = []
            for line_match in line_pattern.finditer(file_diff):
                start_line = int(line_match.group(1))
                num_lines = int(line_match.group(2) or 1)

                # The start and end positions for this chunk of diff
                chunk_start = line_match.end()
                next_chunk = line_pattern.search(file_diff, pos=line_match.span(0)[1])
                chunk_diff = file_diff[
                    chunk_start : next_chunk.span(0)[0] if next_chunk else None
                ]

                lines = chunk_diff.splitlines()
                line_counter = 0
                for line in lines:
                    if line.startswith("+"):
                        if (
                            "@pytest.mark.mpi(min_size" in line
                            and not ("min_size=1" or "min_size=3" or "min_size=4") in line
                        ):
                            # Only include lines where we set the number of mpi processes
                            # to be different from 1, 3, or 4.
                            changed_lines.append(start_line + line_counter)

                        line_counter += 1

            if changed_lines:
                files[file_name] = changed_lines

    return files


def get_common_ancestor(target_branch, feature_branch):
    cmd = ["git", "merge-base", target_branch, feature_branch]
    return subprocess.check_output(cmd).decode("utf-8").strip()


def get_changed_files(target_branch, feature_branch):
    """Get a dictionary of files and their changed lines between the common ancestor and feature_branch."""
    start_commit = get_common_ancestor(target_branch, feature_branch)
    cmd = [
        "git",
        "diff",
        "-U0",
        "--ignore-all-space",
        start_commit,
        feature_branch
    ]
    result = subprocess.check_output(cmd).decode("utf-8")

    return parse_diff_output(result)


def print_occurences(changed_files, title):
    print(title)
    for file_name, lines in changed_files.items():
        print("-----")
        print(f"File: {file_name}")
        print("Changed Lines:", ", ".join(map(str, lines)))
        print("-----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base", default="origin/develop", help="BASE commit (default: %(default)s)"
    )
    parser.add_argument(
        "--head", default="HEAD", help="HEAD commit (default: %(default)s)"
    )

    start_commit = parser.parse_args().base
    print(f"Start commit: {start_commit}")

    end_commit = parser.parse_args().head
    print(f"End commit: {end_commit}")

    invalid_num_rank_detected = get_changed_files(start_commit, end_commit)

    if invalid_num_rank_detected:
        print_occurences(
            invalid_num_rank_detected, "Found test with invalid number of ranks in the following file(s):"
        )

        sys.exit(1)  # Exit with an error code to fail the GitHub Action
    else:
        print("No MPI tests introduced with incorrect number of ranks.")
        sys.exit(0)