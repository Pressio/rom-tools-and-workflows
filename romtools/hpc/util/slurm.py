import textwrap
import shlex

from romtools.hpc.connection import Connection

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

def get_sacct_status(job_id: str, conn: Connection, logger):
    """
    Return the SLURM accounting state and exit code for a completed/disappeared job.

    Returns:
        tuple[str, str] | tuple[None, None]:
            (state, exit_code), or (None, None) if sacct does not have the record yet.
    """
    jid = shlex.quote(str(job_id))

    cmd = (
        f"sacct -j {jid} -X -n -P "
        "--format=JobIDRaw,State%30,ExitCode"
    )

    result = conn.run(cmd)

    if not result.ok:
        logger.log(f"sacct failed for job {job_id}: {result.stderr}")
        return None, None

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split("|")
        if len(parts) < 3:
            continue

        sacct_job_id = parts[0].strip()
        state = parts[1].strip().split()[0].upper()
        exit_code = parts[2].strip()

        if sacct_job_id == str(job_id):
            return state, exit_code

    return None, None
