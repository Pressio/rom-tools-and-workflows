import os
import time
import socket
import textwrap

from romtools.hpc.dispatcher_base import DispatcherBase

class ExampleModel:

    def populate_run_directory(self, run_directory: str, parameter_sample: dict, dispatcher: DispatcherBase = None) -> None:

        # Run directory is on remote host if dispatcher exists
        pass

    def run_model(self, run_directory: str, parameter_sample: dict, dispatcher: DispatcherBase = None) -> int:

        if dispatcher is not None:
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
            return 0

        file_name = f"output-{socket.gethostname()}.txt"
        path = os.path.join(run_directory, file_name)
        content = f"{socket.gethostname()}\n{int(time.time())}\n"
        with open(path, "w") as f:
            f.write(f"{content}\n")
        return 0
