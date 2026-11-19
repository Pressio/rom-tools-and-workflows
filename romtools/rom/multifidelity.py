"""Jointly conditioned autoregressive multifidelity Gaussian-process QoI surrogate.

Each fidelity has a different, potentially non-nested set of training inputs.
The public interface supports any number of ordered fidelities (two or more).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from romtools.rom.qoi_surrogates import GaussianProcessKernel


@dataclass
class FidelityData:
    """Accepted observations at one fidelity, in matching row order."""

    parameters: np.ndarray
    qois: np.ndarray
    training_dirs: tuple[str, ...] = ()
    name: Optional[str] = None


@dataclass
class GPLevelConfig:
    """Base kernel for level zero; discrepancy kernel and rho for later levels."""

    kernel: GaussianProcessKernel
    rho: float = 1.0
    noise_variance: float = 1e-8


@dataclass
class AutoregressiveGPConfig:
    """Configuration of a joint autoregressive GP hierarchy."""

    levels: Sequence[GPLevelConfig]
    tune_hyperparameters: bool = True
    normalize_parameters: bool = True
    optimizer_restarts: int = 4
    length_scale_bounds: tuple[float, float] = (1e-2, 1e2)
    signal_variance_bounds: tuple[float, float] = (1e-6, 1e2)
    rho_bounds: tuple[float, float] = (-5.0, 5.0)


class AutoregressiveGPQoiModel:
    """Scalar-QoI Kennedy--O'Hagan autoregressive GP with joint conditioning.

    f_0 = g_0; f_l = rho_l * f_(l-1) + delta_l, where the base
    process and all discrepancies are independent zero-mean GPs.
    Hyperparameters are fitted by joint marginal likelihood; in particular,
    the high-fidelity observations also inform the base process posterior.
    """

    def __init__(
        self,
        fidelity_data: Sequence[FidelityData],
        config: AutoregressiveGPConfig,
        parameter_names: Optional[Sequence[str]] = None,
    ) -> None:
        if len(fidelity_data) < 2:
            raise ValueError("at least two fidelity levels are required")
        if len(config.levels) != len(fidelity_data):
            raise ValueError("one GPLevelConfig is required for each fidelity level")
        if (not isinstance(config.optimizer_restarts, int)
                or config.optimizer_restarts < 1):
            raise ValueError("optimizer_restarts must be a positive integer")
        for name, bounds in (
            ("length_scale_bounds", config.length_scale_bounds),
            ("signal_variance_bounds", config.signal_variance_bounds),
        ):
            if (len(bounds) != 2 or not np.all(np.isfinite(bounds))
                    or bounds[0] <= 0 or bounds[0] >= bounds[1]):
                raise ValueError(f"{name} must have finite increasing positive bounds")
        if (len(config.rho_bounds) != 2
                or not np.all(np.isfinite(config.rho_bounds))
                or config.rho_bounds[0] >= config.rho_bounds[1]):
            raise ValueError("rho_bounds must be finite and increasing")

        xs, ys = [], []
        for index, data in enumerate(fidelity_data):
            x = np.asarray(data.parameters, dtype=float)
            y = np.asarray(data.qois, dtype=float)
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            if (x.ndim != 2 or not len(x) or y.ndim != 1
                    or len(y) != len(x)
                    or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))):
                raise ValueError(f"invalid scalar QoI training data at fidelity {index}")
            if xs and x.shape[1] != xs[0].shape[1]:
                raise ValueError("parameter dimension must match across fidelity levels")
            if data.training_dirs and len(data.training_dirs) != len(x):
                raise ValueError("training_dirs must match the number of samples")
            xs.append(x.copy())
            ys.append(y.copy())

        self.parameter_names = (
            tuple(parameter_names) if parameter_names is not None else None
        )
        if self.parameter_names is not None and len(self.parameter_names) != xs[0].shape[1]:
            raise ValueError("parameter_names must match parameter dimension")
        self.config = config
        self.num_fidelities = len(xs)
        self._param_min = np.min(np.vstack(xs), axis=0)
        param_range = np.ptp(np.vstack(xs), axis=0)
        self._param_scale = np.where(param_range > 0, param_range, 1.0)
        if config.normalize_parameters:
            self._xs = [(x - self._param_min) / self._param_scale for x in xs]
        else:
            self._xs = xs
        self._ys = ys
        self._y = np.concatenate(ys)
        self._levels = np.concatenate([
            np.full(len(y), l, dtype=int) for l, y in enumerate(ys)
        ])
        self._kernels = [
            GaussianProcessKernel(
                length_scale=np.asarray(c.kernel.length_scale, dtype=float).copy(),
                signal_variance=float(c.kernel.signal_variance),
            ) for c in config.levels
        ]
        self._rhos = np.array([c.rho for c in config.levels[1:]], dtype=float)
        self._noise = np.array([c.noise_variance for c in config.levels], dtype=float)
        if not np.all(np.isfinite(self._rhos)):
            raise ValueError("rho must be finite")
        if not np.all(np.isfinite(self._noise)) or np.any(self._noise < 0):
            raise ValueError("noise variances must be finite and nonnegative")
        self._validate_kernels()
        if config.tune_hyperparameters:
            self._tune()
        self._fit()

    def _validate_kernels(self) -> None:
        d = self._xs[0].shape[1]
        for kernel in self._kernels:
            # Reuse existing kernel validation with correctly shaped inputs.
            kernel(np.zeros((1, d)), np.zeros((1, d)))

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return ((x - self._param_min) / self._param_scale
                if self.config.normalize_parameters else x)

    def _weights(self, level: int) -> np.ndarray:
        weights = np.ones(level + 1)
        for k in range(level):
            weights[k] = np.prod(self._rhos[k:level])
        return weights

    def _cov(self, x_a: np.ndarray, a: int, x_b: np.ndarray, b: int) -> np.ndarray:
        wa, wb = self._weights(a), self._weights(b)
        result = np.zeros((len(x_a), len(x_b)))
        for k in range(min(a, b) + 1):
            result += wa[k] * wb[k] * self._kernels[k](x_a, x_b)
        return result

    def _training_cov(self) -> np.ndarray:
        sizes = [len(x) for x in self._xs]
        offsets = np.cumsum([0] + sizes)
        result = np.empty((len(self._y), len(self._y)))
        for a in range(self.num_fidelities):
            ia = slice(offsets[a], offsets[a + 1])
            for b in range(a + 1):
                ib = slice(offsets[b], offsets[b + 1])
                block = self._cov(self._xs[a], a, self._xs[b], b)
                result[ia, ib] = block
                if a != b:
                    result[ib, ia] = block.T
            result[ia, ia] += self._noise[a] * np.eye(sizes[a])
        return result

    def _factor(self):
        matrix = self._training_cov()
        # Fixed numerical jitter is not included in the observation noise.
        matrix.flat[::len(matrix) + 1] += 1e-12
        return cho_factor(matrix, lower=True, check_finite=True)

    def _pack(self):
        values = []
        for k in self._kernels:
            scales = np.asarray(k.length_scale, dtype=float)
            if scales.ndim == 0:
                scales = np.full(self._xs[0].shape[1], float(scales))
            values.extend(np.log(scales))
            values.append(np.log(k.signal_variance))
        values.extend(self._rhos)
        return np.asarray(values)

    def _unpack(self, theta):
        d = self._xs[0].shape[1]
        pos = 0
        for kernel in self._kernels:
            kernel.length_scale = np.exp(theta[pos:pos + d])
            kernel.signal_variance = float(np.exp(theta[pos + d]))
            pos += d + 1
        self._rhos = np.asarray(theta[pos:], dtype=float).copy()

    def _tune(self):
        d = self._xs[0].shape[1]
        bounds = []
        for _ in range(self.num_fidelities):
            bounds.extend([tuple(np.log(self.config.length_scale_bounds))] * d)
            bounds.append(tuple(np.log(self.config.signal_variance_bounds)))
        bounds.extend([self.config.rho_bounds] * (self.num_fidelities - 1))
        initial = np.clip(self._pack(), [b[0] for b in bounds], [b[1] for b in bounds])
        rng = np.random.default_rng(0)
        starts = [initial]
        starts.extend(rng.uniform([b[0] for b in bounds],
                                  [b[1] for b in bounds])
                      for _ in range(self.config.optimizer_restarts - 1))

        def objective(theta):
            self._unpack(theta)
            try:
                chol = self._factor()
                alpha = cho_solve(chol, self._y)
                logdet_half = np.sum(np.log(np.diag(chol[0])))
                return (0.5 * self._y @ alpha + logdet_half
                        + 0.5 * len(self._y) * np.log(2 * np.pi))
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                return 1e100

        results = [minimize(objective, start, method="L-BFGS-B", bounds=bounds)
                   for start in starts]
        valid = [r for r in results if np.isfinite(r.fun) and r.fun < 1e99]
        if not valid:
            raise RuntimeError("joint GP hyperparameter optimization failed")
        best = min(valid, key=lambda r: r.fun)
        self._unpack(best.x)
        self._validate_kernels()

    def _fit(self) -> None:
        self._chol = self._factor()
        self._alpha = cho_solve(self._chol, self._y)

    def _resolve_fidelity(self, fidelity: int) -> int:
        if not isinstance(fidelity, (int, np.integer)):
            raise TypeError("fidelity must be an integer")
        index = fidelity if fidelity >= 0 else self.num_fidelities + fidelity
        if index < 0 or index >= self.num_fidelities:
            raise ValueError("fidelity index out of range")
        return int(index)

    def predict(self, parameters, fidelity: int = -1, return_std: bool = False):
        """Predict latent QoI mean and optionally posterior standard deviation."""
        level = self._resolve_fidelity(fidelity)
        x = np.asarray(parameters, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self._xs[0].shape[1]:
            raise ValueError("prediction parameter dimension mismatch")
        if not np.all(np.isfinite(x)):
            raise ValueError("prediction parameters must be finite")
        x = self._scale(x)
        k_star = np.hstack([
            self._cov(x, level, xx, l) for l, xx in enumerate(self._xs)
        ])
        mean = k_star @ self._alpha
        if not return_std:
            return mean
        variance = np.diag(self._cov(x, level, x, level))
        variance -= np.sum(k_star * cho_solve(self._chol, k_star.T).T, axis=1)
        return mean, np.sqrt(np.maximum(variance, 0.0))

    def _parameter_row(self, parameter_sample) -> np.ndarray:
        if isinstance(parameter_sample, dict):
            if self.parameter_names is None:
                raise ValueError("parameter_names required for dictionary inputs")
            return np.array([parameter_sample[name] for name in self.parameter_names])
        return np.asarray(parameter_sample, dtype=float).ravel()

    def populate_run_directory(self, run_directory, parameter_sample) -> None:
        pass

    def run_model(self, run_directory, parameter_sample) -> int:
        return 0

    def compute_qoi(self, run_directory, parameter_sample, fidelity: int = -1):
        return self.predict(self._parameter_row(parameter_sample), fidelity=fidelity)

    def compute_error_estimate(
        self, run_directory, parameter_sample, fidelity: int = -1,
    ) -> float:
        _, std = self.predict(self._parameter_row(parameter_sample),
                              fidelity=fidelity, return_std=True)
        return float(std[0])


class AutoregressiveGPBuilder:
    """Build a jointly trained AR-GP from all accepted fidelity datasets."""

    def __init__(self, config: AutoregressiveGPConfig,
                 parameter_names: Optional[Sequence[str]] = None):
        self.config = config
        self.parameter_names = parameter_names

    def build(self, offline_data_dir: str, training_data: Sequence[FidelityData]):
        return AutoregressiveGPQoiModel(
            fidelity_data=training_data, config=self.config,
            parameter_names=self.parameter_names,
        )
