"""Small local forward model: y(t) = amplitude * exp(-rate * t)."""

import json
from pathlib import Path

from typing import Optional

import numpy as np

from romtools.hpc.dispatchers import BaseDispatcher, LocalDispatcher

class DecayModel:
    def __init__(self, dispatcher: Optional[BaseDispatcher] = None) -> None:
        if dispatcher is None:
            dispatcher = LocalDispatcher()
        self.dispatcher = dispatcher
        self.times = np.linspace(0.0, 4.0, 9)

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        Path(run_directory, "parameters.json").write_text(
            json.dumps({key: float(value) for key, value in parameter_sample.items()}, indent=2)
            + "\n", encoding="utf-8"
        )

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        values = parameter_sample["amplitude"] * np.exp(-parameter_sample["rate"] * self.times)
        if not np.all(np.isfinite(values)):
            raise ValueError("The decay model produced non-finite observations")
        np.savez(Path(run_directory, "solution.npz"), times=self.times, values=values)
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        with np.load(Path(run_directory, "solution.npz")) as solution:
            return solution["values"]
