import textwrap
from typing import Optional

from romtools.hpc.dispatchers import BaseDispatcher, resolve_dispatcher

class ExampleModel:

    def __init__(self, dispatcher: Optional[BaseDispatcher] = None):
        # Resolves to a LocalDispatcher if none is provided
        self.dispatcher = resolve_dispatcher(dispatcher)

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        # Here's where you would use the file management methods of the Dispatcher
        pass

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        # The core method of the model. You can configure dispatcher to use
        # a pre-written SLURM script, or you can build one with the Dispatcher
        # like we do here
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
