import textwrap

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
