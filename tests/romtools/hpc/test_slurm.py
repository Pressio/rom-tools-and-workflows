from romtools.hpc.util.slurm import create_slurm_script


def test_create_slurm_script_renders_all_fields():
    script = create_slurm_script(
        job_name="my_job",
        num_nodes=2,
        tasks_per_node=4,
        wall_time="01:30:00",
        wcid="acct123",
        partition="batch",
        command="srun ./my_app",
    )

    assert "#!/bin/bash" in script
    assert "#SBATCH --job-name=my_job" in script
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --ntasks-per-node=4" in script
    assert "#SBATCH --time=01:30:00" in script
    assert "#SBATCH --account=acct123" in script
    assert "#SBATCH -p batch" in script
    assert script.strip().endswith("srun ./my_app")


def test_create_slurm_script_preserves_multiline_command():
    command = "echo start\necho end"
    script = create_slurm_script(
        job_name="job",
        num_nodes=1,
        tasks_per_node=1,
        wall_time="00:01:00",
        wcid="acct",
        partition="short",
        command=command,
    )

    assert "echo start\necho end" in script
