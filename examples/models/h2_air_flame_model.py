"""romtools QoI-model wrapper for the pure-Python H2-air flame benchmark."""

import os
from typing import Dict

import numpy as np

from h2_air_flame import H2AirFlame, extract_temperature_sensors


class H2AirFlameQoiModel:
    """Sensor-based ``QoiModel`` wrapper around :class:`H2AirFlame`.

    The QoI consists of nondimensional temperature samples on regular spatial
    and temporal strides. Full state histories are saved to each run directory
    so the same FOM evaluations can also be reused for ROM construction.
    """

    parameter_names = H2AirFlame.parameter_names

    def __init__(
        self,
        nx: int = 25,
        ny: int = 13,
        dt: float = 1.0e-4,
        t_end: float = 5.0e-3,
        snapshot_stride: int = 10,
        spatial_sensor_stride: int = 4,
        temporal_sensor_stride: int = 1,
    ) -> None:
        self._model_kwargs = {
            "nx": nx,
            "ny": ny,
            "dt": dt,
            "t_end": t_end,
            "snapshot_stride": snapshot_stride,
        }
        self.spatial_sensor_stride = int(spatial_sensor_stride)
        self.temporal_sensor_stride = int(temporal_sensor_stride)
        if self.spatial_sensor_stride < 1 or self.temporal_sensor_stride < 1:
            raise ValueError("sensor strides must be at least 1")

    def populate_run_directory(self, run_directory: str, parameter_sample: Dict) -> None:
        """Create the run directory required by the workflow model protocol."""
        del parameter_sample
        os.makedirs(run_directory, exist_ok=True)

    def _parameters(self, parameter_sample: Dict):
        missing = [name for name in self.parameter_names if name not in parameter_sample]
        if missing:
            raise KeyError(
                "parameter_sample is missing required keys: {}".format(missing)
            )
        return [float(parameter_sample[name]) for name in self.parameter_names]

    def run_model(self, run_directory: str, parameter_sample: Dict) -> int:
        """Run the flame model and save states, times, parameters, and QoI."""
        model = H2AirFlame(**self._model_kwargs)
        parameters = self._parameters(parameter_sample)
        states, times = model.solve(*parameters)
        qoi = extract_temperature_sensors(
            states,
            spatial_stride=self.spatial_sensor_stride,
            temporal_stride=self.temporal_sensor_stride,
        )
        np.savez(
            os.path.join(run_directory, "solution.npz"),
            states=states,
            times=times,
            qoi=qoi,
            parameter_names=np.asarray(self.parameter_names),
            parameters=np.asarray(parameters),
        )
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: Dict) -> np.ndarray:
        """Read and return the flattened temperature-sensor QoI."""
        del parameter_sample
        with np.load(os.path.join(run_directory, "solution.npz")) as data:
            return np.asarray(data["qoi"], dtype=float)


if __name__ == "__main__":
    import tempfile

    sample = {
        "kappa": 2.0,
        "scaled_activation_energy": 8.0,
        "beta_x": 40.0,
        "beta_y": 7.0,
    }
    with tempfile.TemporaryDirectory() as directory:
        model = H2AirFlameQoiModel(
            nx=12, ny=7, t_end=2.0e-4, snapshot_stride=1
        )
        model.populate_run_directory(directory, sample)
        model.run_model(directory, sample)
        qoi = model.compute_qoi(directory, sample)
        print("H2-air flame QoI size:", qoi.size)
