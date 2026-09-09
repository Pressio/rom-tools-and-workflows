import textwrap
from typing import Optional

from romtools.hpc.dispatchers import BaseDispatcher

class ExampleModel:

    def __init__(self, dispatcher: BaseDispatcher = None):
        self.dispatcher = dispatcher

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
        self.dispatcher.submit_job(cmd, run_directory=run_directory)

        return 0
