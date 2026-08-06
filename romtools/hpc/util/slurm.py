import textwrap
import shlex

SLURM_TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}

def create_slurm_script(job_name: str, num_nodes: int, tasks_per_node: int, wall_time: str, wcid: str, partition: str, command: str) -> str:
    """
    Create a SLURM job script with the specified parameters.

    Parameters:
    - job_name: Name of the SLURM job
    - num_nodes: Number of nodes to request
    - tasks_per_node: Number of tasks to run on each node
    - wall_time: Maximum wall time for the job (format: HH:MM:SS)
    - wcid: The account to charge for the job
    - partition: The partition to submit the job to ("batch", "short")
    - command: The command to execute in the job

    Returns:
    A string containing the SLURM job script.
    """
    slurm_script = textwrap.dedent(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --time={wall_time}
#SBATCH --account={wcid}
#SBATCH -p {partition}
{command}
""")
    return slurm_script

def slurm_exitcode_to_python_style(exitcode: str) -> int:
    if exitcode is None:
        return None
    status_str, signal_str = exitcode.split(":")
    status = int(status_str)
    signal = int(signal_str)

    if signal != 0:
        return -signal
    return status

def parse_sbatch_args(script: str) -> tuple[str | None, str | None]:
    if not script:
        return None, None

    specified_out, specified_err = None, None

    with open(script, "r") as runscript:
        for line in runscript:
            line = line.strip()

            if not line.startswith("#SBATCH"):
                continue

            args = shlex.split(line[len("#SBATCH"):].strip())

            i = 0
            while i < len(args):
                arg = args[i]

                if arg.startswith("--output="):
                    specified_out = arg.split("=", 1)[1]

                elif arg == "--output" and i + 1 < len(args):
                    specified_out = args[i + 1]
                    i += 1

                elif arg.startswith("-o") and arg != "-o":
                    specified_out = arg[2:]

                elif arg == "-o" and i + 1 < len(args):
                    specified_out = args[i + 1]
                    i += 1

                elif arg.startswith("--error="):
                    specified_err = arg.split("=", 1)[1]

                elif arg == "--error" and i + 1 < len(args):
                    specified_err = args[i + 1]
                    i += 1

                elif arg.startswith("-e") and arg != "-e":
                    specified_err = arg[2:]

                elif arg == "-e" and i + 1 < len(args):
                    specified_err = args[i + 1]
                    i += 1

                i += 1

            if specified_out is not None and specified_err is not None:
                break
    return specified_out, specified_err
