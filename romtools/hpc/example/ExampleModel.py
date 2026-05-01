import os
import time
import socket
import textwrap

import numpy as np

from romtools.hpc.dispatcher import Dispatcher

class ExampleModel:

    def __init__(self, dispatcher: Dispatcher = None) -> None:
        self.dispatcher = dispatcher

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:

        # Run directory is on remote host if dispatcher exists
        pass

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:

        if self.dispatcher is not None:
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

        file_name = f"output-{socket.gethostname()}.txt"
        path = os.path.join(run_directory, file_name)
        content = f"{socket.gethostname()}\n{int(time.time())}\n"
        with open(path, "w") as f:
            f.write(f"{content}\n")
        return

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        pass
