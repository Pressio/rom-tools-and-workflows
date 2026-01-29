from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class GaussianProcessKernel:
    length_scale: float = 1.0
    signal_variance: float = 1.0

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        x1_sq = np.sum(x1 * x1, axis=1)[:, None]
        x2_sq = np.sum(x2 * x2, axis=1)[None, :]
        sq_dist = x1_sq + x2_sq - 2.0 * x1 @ x2.T
        return self.signal_variance * np.exp(-0.5 * sq_dist / (self.length_scale ** 2))


class GaussianProcessRegressorLite:
    def __init__(self,
                 kernel: Optional[GaussianProcessKernel] = None,
                 noise_variance: float = 1e-10,
                 jitter: float = 1e-12) -> None:
        self.kernel = kernel if kernel is not None else GaussianProcessKernel()
        self.noise_variance = noise_variance
        self.jitter = jitter
        self._x_train: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._chol: Optional[np.ndarray] = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float).reshape(-1, 1)
        kxx = self.kernel(x_train, x_train)
        kxx = kxx + (self.noise_variance + self.jitter) * np.eye(kxx.shape[0])
        self._chol = np.linalg.cholesky(kxx)
        self._alpha = np.linalg.solve(self._chol.T, np.linalg.solve(self._chol, y_train))
        self._x_train = x_train

    def predict(self, x_query: np.ndarray) -> np.ndarray:
        if self._x_train is None or self._alpha is None:
            raise RuntimeError("GaussianProcessRegressorLite has not been fit yet.")
        x_query = np.asarray(x_query, dtype=float)
        kx = self.kernel(x_query, self._x_train)
        y_mean = kx @ self._alpha
        return y_mean.ravel()


class GaussianProcessQoiModel:
    """
    Gaussian Process QoI surrogate model that follows the QoiModel API.

    For vector QoIs, a POD is performed and a GP is trained for each
    reduced coefficient.
    """

    def __init__(self,
                 parameters: np.ndarray,
                 qois: np.ndarray,
                 parameter_names: Optional[Sequence[str]] = None,
                 pod_energy_fraction: float = 0.999999,
                 max_pod_modes: Optional[int] = None,
                 kernel: Optional[GaussianProcessKernel] = None,
                 noise_variance: float = 1e-10) -> None:
        self.parameter_names = list(parameter_names) if parameter_names is not None else None
        self.parameters = np.asarray(parameters, dtype=float)
        self.qois = np.asarray(qois, dtype=float)
        self.pod_energy_fraction = pod_energy_fraction
        self.max_pod_modes = max_pod_modes
        self.kernel = kernel if kernel is not None else GaussianProcessKernel()
        self.noise_variance = noise_variance

        self._mean_qoi: Optional[np.ndarray] = None
        self._pod_modes: Optional[np.ndarray] = None
        self._gps: List[GaussianProcessRegressorLite] = []
        self._fit()

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        return None

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        x = self._parameter_sample_to_array(parameter_sample)
        coeffs = np.array([gp.predict(x[None, :])[0] for gp in self._gps], dtype=float)
        if self._pod_modes is None:
            return np.asarray([coeffs[0]])
        qoi = self._mean_qoi + self._pod_modes @ coeffs
        return np.asarray(qoi).ravel()

    def _parameter_sample_to_array(self, parameter_sample: dict) -> np.ndarray:
        if isinstance(parameter_sample, dict):
            if self.parameter_names is None:
                raise ValueError("parameter_names must be provided to map dict inputs.")
            return np.array([parameter_sample[name] for name in self.parameter_names], dtype=float)
        return np.asarray(parameter_sample, dtype=float).ravel()

    def _fit(self) -> None:
        if self.qois.ndim == 1 or self.qois.shape[1] == 1:
            y = self.qois.reshape(-1, 1)
            self._mean_qoi = np.zeros(1, dtype=float)
            self._pod_modes = None
            gp = GaussianProcessRegressorLite(kernel=self.kernel, noise_variance=self.noise_variance)
            gp.fit(self.parameters, y.ravel())
            self._gps = [gp]
            return

        mean_qoi = np.mean(self.qois, axis=0)
        centered = self.qois - mean_qoi[None, :]
        _, svals, vt = np.linalg.svd(centered, full_matrices=False)
        energy = np.cumsum(svals ** 2)
        total_energy = energy[-1] if energy.size > 0 else 0.0
        if total_energy == 0.0:
            modes = np.zeros((self.qois.shape[1], 1))
        else:
            k = int(np.searchsorted(energy / total_energy, self.pod_energy_fraction) + 1)
            if self.max_pod_modes is not None:
                k = min(k, self.max_pod_modes)
            k = max(k, 1)
            modes = vt[:k, :].T

        coeffs = centered @ modes
        self._mean_qoi = mean_qoi
        self._pod_modes = modes
        self._gps = []
        for i in range(coeffs.shape[1]):
            gp = GaussianProcessRegressorLite(kernel=self.kernel, noise_variance=self.noise_variance)
            gp.fit(self.parameters, coeffs[:, i])
            self._gps.append(gp)
