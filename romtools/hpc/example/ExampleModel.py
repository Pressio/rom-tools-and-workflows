import os
import time
import socket
import textwrap

from romtools.hpc.dispatcher_base import DispatcherBase
from romtools.hpc.local_dispatcher import LocalDispatcher

class ExampleModel:

    def __init__(self, dispatcher: DispatcherBase = LocalDispatcher()):
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
        self.dispatcher.dispatch(cmd, run_directory=run_directory)

        return 0
