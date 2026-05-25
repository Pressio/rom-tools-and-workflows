import os
import time
import socket
import textwrap

from romtools.hpc.remote_dispatcher import RemoteDispatcher

class ExampleModel:

    def populate_run_directory(self, run_directory: str, parameter_sample: dict, dispatcher: RemoteDispatcher = None) -> None:

        # Run directory is on remote host if dispatcher exists
        pass

    def run_model(self, run_directory: str, parameter_sample: dict, dispatcher: RemoteDispatcher) -> int:

        file_name = "output-$(hostname).txt"
        cmd = textwrap.dedent(f"""\
            srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
            cat > "{file_name}" << EOF
                $(hostname)
                $(date)
            EOF
            '
        """)
        dispatcher.dispatch(cmd, run_directory=run_directory)
