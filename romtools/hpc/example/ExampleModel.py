import textwrap
from typing import Optional

from romtools.hpc.dispatchers import BaseDispatcher, resolve_dispatcher

class ExampleModel:

    def __init__(self, dispatcher: Optional[BaseDispatcher] = None):
        # A LocalDispatcher reaches SLURM directly when this process runs on a
        # cluster node, so submit_job() works with or without a connection.
        self.dispatcher = resolve_dispatcher(dispatcher)

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        pass

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:

        file_name = "output-$(hostname).txt"
        cmd = textwrap.dedent(f"""\
            srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
            cat > "{file_name}" << EOF
                $(hostname)
                $(date)
            EOF
            '
        """)
        result = self.dispatcher.submit_job(cmd, run_directory=run_directory)

        # SLURM gives us no exit code when the job never reached a terminal state
        return result.exit_code if result.exit_code is not None else 1
