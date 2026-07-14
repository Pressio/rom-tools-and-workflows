import os
import time
import socket

from romtools.hpc.dispatcher_base import DispatcherBase

class ExampleModelNoConn:

    def __init__(self, dispatcher: DispatcherBase = None):
        self.dispatcher = dispatcher

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        pass

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        file_name = f"output-{socket.gethostname()}.txt"
        path = os.path.join(run_directory, file_name)
        content = f"{socket.gethostname()}\n{int(time.time())}\n"
        with open(path, "w") as f:
            f.write(f"{content}\n")
        return 0
