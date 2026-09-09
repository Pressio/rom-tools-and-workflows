import os
import sys
from typing import Optional

import numpy as np

from romtools.hpc.dispatchers import BaseDispatcher, LocalDispatcher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "docs", "source", "demos", "notebooks", "adr_1d"))
from adr_1d import advectionDiffusionProblem


class Adr1dModel:

    def __init__(self, nx: int = 33, dispatcher: Optional[BaseDispatcher] = None) -> None:
        if dispatcher is None:
            dispatcher = LocalDispatcher()
        self.dispatcher = dispatcher
        self.problem = advectionDiffusionProblem(nx)

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        with open(os.path.join(run_directory, "params.txt"), "w", encoding="utf-8") as handle:
            for key, value in parameter_sample.items():
                handle.write(f"{key}: {value}\n")

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        c = float(parameter_sample["c"])
        nu = float(parameter_sample["nu"])
        u = self.problem.solve(c, nu)
        np.savez(
            os.path.join(run_directory, "solution.npz"),
            u=u,
            x=self.problem.x_,
            c=c,
            nu=nu,
        )
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        solution = np.load(os.path.join(run_directory, "solution.npz"))
        return solution["u"]
