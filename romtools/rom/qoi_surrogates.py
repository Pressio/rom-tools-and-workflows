from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

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

    def predict_mean_and_std(self, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._x_train is None or self._alpha is None:
            raise RuntimeError("GaussianProcessRegressorLite has not been fit yet.")
        x_query = np.asarray(x_query, dtype=float)
        kx = self.kernel(x_query, self._x_train)
        y_mean = kx @ self._alpha

        cov = self.kernel(x_query, x_query) - kx @ np.linalg.solve(
            self._chol.T, np.linalg.solve(self._chol, kx.T)
        )
        y_var = np.clip(np.diag(cov), 0.0, None)
        y_std = np.sqrt(y_var)
        return y_mean.ravel(), y_std.ravel()

    @staticmethod
    def log_marginal_likelihood(x_train: np.ndarray,
                                y_train: np.ndarray,
                                kernel: GaussianProcessKernel,
                                noise_variance: float,
                                jitter: float = 1e-12) -> float:
        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float).reshape(-1, 1)
        kxx = kernel(x_train, x_train)
        kxx = kxx + (noise_variance + jitter) * np.eye(kxx.shape[0])
        try:
            chol = np.linalg.cholesky(kxx)
        except np.linalg.LinAlgError:
            return float("-inf")
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_train))
        log_det = np.sum(np.log(np.diag(chol)))
        n = y_train.shape[0]
        lml = -0.5 * float(np.squeeze(y_train.T @ alpha)) - log_det - 0.5 * n * np.log(2.0 * np.pi)
        return lml


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
                 noise_variance: Optional[float] = None,
                 auto_noise_variance: bool = False,
                 noise_variance_fraction: float = 1e-6,
                 tune_hyperparameters: bool = False,
                 length_scale_grid: Optional[Sequence[float]] = None,
                 signal_variance_grid: Optional[Sequence[float]] = None,
                 normalize_parameters: bool = False,
                 normalize_targets: bool = False) -> None:
        self.parameter_names = list(parameter_names) if parameter_names is not None else None
        self.parameters = np.asarray(parameters, dtype=float)
        self.qois = np.asarray(qois, dtype=float)
        self.pod_energy_fraction = pod_energy_fraction
        self.max_pod_modes = max_pod_modes
        self.kernel = kernel if kernel is not None else GaussianProcessKernel()
        self.noise_variance = noise_variance
        self.auto_noise_variance = auto_noise_variance
        self.noise_variance_fraction = noise_variance_fraction
        self.tune_hyperparameters = tune_hyperparameters
        self.length_scale_grid = length_scale_grid
        self.signal_variance_grid = signal_variance_grid
        self.normalize_parameters = normalize_parameters
        self.normalize_targets = normalize_targets

        self._mean_qoi: Optional[np.ndarray] = None
        self._pod_modes: Optional[np.ndarray] = None
        self._gps: List[GaussianProcessRegressorLite] = []
        self._param_min: Optional[np.ndarray] = None
        self._param_scale: Optional[np.ndarray] = None
        self._scaled_parameters: Optional[np.ndarray] = None
        self._target_min: Optional[np.ndarray] = None
        self._target_scale: Optional[np.ndarray] = None
        self._fit()

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        return None

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        x = self._parameter_sample_to_array(parameter_sample)
        coeffs = np.array([gp.predict(x[None, :])[0] for gp in self._gps], dtype=float)
        coeffs = self._unscale_targets(coeffs)
        if self._pod_modes is None:
            return np.asarray([coeffs[0]])
        qoi = self._mean_qoi + self._pod_modes @ coeffs
        return np.asarray(qoi).ravel()

    def compute_qoi_and_var(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        x = self._parameter_sample_to_array(parameter_sample)

        predictions = [gp.predict_mean_and_std(x[None, :]) for gp in self._gps]
        mean_coeffs = np.array([prediction[0] for prediction in predictions], dtype=float)
        std_coeffs = np.array([prediction[1] for prediction in predictions], dtype=float)
        mean_coeffs = self._unscale_targets(mean_coeffs)
        std_coeffs = self._unscale_targets(std_coeffs)
        if self._pod_modes is None:
            return np.asarray([mean_coeffs[0]]), np.asarray([std_coeffs[0]])
        qoi_mean = self._mean_qoi + self._pod_modes @ mean_coeffs
        # assume pod modes are not correlated (should revisit this assumption)
        qoi_std = np.sqrt(self._pod_modes @ (std_coeffs * std_coeffs))
        return np.asarray(qoi_mean).ravel(), np.asarray(qoi_std).ravel()

    def _parameter_sample_to_array(self, parameter_sample: dict) -> np.ndarray:
        if isinstance(parameter_sample, dict):
            if self.parameter_names is None:
                raise ValueError("parameter_names must be provided to map dict inputs.")
            params = np.array([parameter_sample[name] for name in self.parameter_names], dtype=float)
        else:
            params = np.asarray(parameter_sample, dtype=float).ravel()
        return self._scale_parameters(params)

    def _fit(self) -> None:
        self._initialize_parameter_scaling()
        parameters = self._scaled_parameters if self._scaled_parameters is not None else self.parameters
        if self.qois.ndim == 1 or self.qois.shape[1] == 1:
            y = self.qois.reshape(-1, 1)
            self._mean_qoi = np.zeros(1, dtype=float)
            self._pod_modes = None
            y_scaled = self._scale_targets(y.ravel())
            noise_variance = self._resolve_noise_variance(y_scaled)
            kernel = self._tuned_kernel(parameters, y_scaled, noise_variance)
            gp = GaussianProcessRegressorLite(kernel=kernel, noise_variance=noise_variance)
            gp.fit(parameters, y_scaled)
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
        if self.normalize_targets:
            coeff_min = np.min(coeffs, axis=0)
            coeff_max = np.max(coeffs, axis=0)
            scale = coeff_max - coeff_min
            scale[scale == 0.0] = 1.0
            self._target_min = coeff_min
            self._target_scale = scale
        self._mean_qoi = mean_qoi
        self._pod_modes = modes
        self._gps = []
        for i in range(coeffs.shape[1]):
            target = coeffs[:, i]
            target_scaled = self._scale_targets(target, index=i)
            noise_variance = self._resolve_noise_variance(target_scaled)
            kernel = self._tuned_kernel(parameters, target_scaled, noise_variance)
            gp = GaussianProcessRegressorLite(kernel=kernel, noise_variance=noise_variance)
            gp.fit(parameters, target_scaled)
            self._gps.append(gp)

    def _resolve_noise_variance(self, target: np.ndarray) -> float:
        if self.noise_variance is not None:
            return float(self.noise_variance)
        if self.auto_noise_variance:
            variance = float(np.var(target))
            return self.noise_variance_fraction * variance
        return 1e-10

    def _initialize_parameter_scaling(self) -> None:
        if not self.normalize_parameters:
            self._param_min = None
            self._param_scale = None
            self._scaled_parameters = None
            return
        param_min = np.min(self.parameters, axis=0)
        param_max = np.max(self.parameters, axis=0)
        scale = param_max - param_min
        scale[scale == 0.0] = 1.0
        self._param_min = param_min
        self._param_scale = scale
        self._scaled_parameters = (self.parameters - param_min) / scale

    def _scale_parameters(self, params: np.ndarray) -> np.ndarray:
        if not self.normalize_parameters or self._param_min is None or self._param_scale is None:
            return params
        return (params - self._param_min) / self._param_scale

    def _scale_targets(self, target: np.ndarray, index: Optional[int] = None) -> np.ndarray:
        if not self.normalize_targets:
            return np.asarray(target, dtype=float).ravel()
        target = np.asarray(target, dtype=float).ravel()
        if self._target_min is None or self._target_scale is None:
            target_min = np.min(target)
            target_max = np.max(target)
            scale = target_max - target_min
            if scale == 0.0:
                scale = 1.0
            self._target_min = np.array([target_min], dtype=float)
            self._target_scale = np.array([scale], dtype=float)
        if index is not None and self._target_min.size > 1:
            target_min = self._target_min[index]
            scale = self._target_scale[index]
        else:
            target_min = float(self._target_min[0])
            scale = float(self._target_scale[0])
        return (target - target_min) / scale

    def _unscale_targets(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).ravel()
        if not self.normalize_targets or self._target_min is None or self._target_scale is None:
            return values
        if self._target_min.size == 1:
            return values * self._target_scale[0] + self._target_min[0]
        return values * self._target_scale + self._target_min

    def _tuned_kernel(self,
                      parameters: np.ndarray,
                      target: np.ndarray,
                      noise_variance: float) -> GaussianProcessKernel:
        if not self.tune_hyperparameters:
            return self.kernel
        length_scales = (self.length_scale_grid
                         if self.length_scale_grid is not None
                         else np.logspace(-2, 2, 9))
        signal_variances = (self.signal_variance_grid
                            if self.signal_variance_grid is not None
                            else np.logspace(-6, 2, 9))
        best_kernel = self.kernel
        best_lml = float("-inf")
        for length_scale in length_scales:
            for signal_variance in signal_variances:
                candidate = GaussianProcessKernel(length_scale=float(length_scale),
                                                  signal_variance=float(signal_variance))
                lml = GaussianProcessRegressorLite.log_marginal_likelihood(
                    parameters, target, candidate, noise_variance
                )
                if lml > best_lml:
                    best_lml = lml
                    best_kernel = candidate
        return best_kernel


@dataclass
class NeuralNetworkConfig:
    """Configuration for the torch neural-network QoI surrogate.

    The default architecture contains two hidden layers. If
    ``hidden_neurons_per_layer`` is not provided, each hidden layer contains
    ``hidden_width_multiplier * parameter_dimension`` neurons, giving the
    baseline width of three times the parameter dimension.
    """

    num_hidden_layers: int = 2
    hidden_neurons_per_layer: Optional[int] = None
    hidden_width_multiplier: float = 3.0
    activation: str = "tanh"
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    training_iterations: int = 5000
    weight_decay: float = 0.0
    random_seed: int = 1
    dtype: str = "float64"
    device: str = "cpu"

    def hidden_layer_width(self, parameter_dimension: int) -> int:
        if parameter_dimension < 1:
            raise ValueError("parameter_dimension must be positive")
        if self.hidden_neurons_per_layer is not None:
            width = int(self.hidden_neurons_per_layer)
        else:
            width = int(round(self.hidden_width_multiplier * parameter_dimension))
        if width < 1:
            raise ValueError("hidden layer width must be positive")
        return width

    def validate(self) -> None:
        if self.num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be nonnegative")
        if self.hidden_neurons_per_layer is not None and self.hidden_neurons_per_layer < 1:
            raise ValueError("hidden_neurons_per_layer must be positive")
        if self.hidden_width_multiplier <= 0.0:
            raise ValueError("hidden_width_multiplier must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.training_iterations < 1:
            raise ValueError("training_iterations must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative")


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for NeuralNetworkQoiModel. "
            "Install romtools with the 'WithTorch' optional dependency."
        ) from exc
    return torch


class NeuralNetworkQoiModel:
    """Torch neural-network QoI surrogate following the QoiModel API.

    Scalar QoIs are learned directly. Vector QoIs use the same POD reduction
    strategy as :class:`GaussianProcessQoiModel`: the training QoIs are
    mean-centered, truncated according to ``pod_energy_fraction`` and
    ``max_pod_modes``, and the network predicts the retained POD coefficients.
    A single multi-output network is used for all retained coefficients.
    """

    def __init__(self,
                 parameters: np.ndarray,
                 qois: np.ndarray,
                 parameter_names: Optional[Sequence[str]] = None,
                 pod_energy_fraction: float = 0.999999,
                 max_pod_modes: Optional[int] = None,
                 network_config: Optional[NeuralNetworkConfig] = None,
                 normalize_parameters: bool = True,
                 normalize_targets: bool = True) -> None:
        self.parameter_names = list(parameter_names) if parameter_names is not None else None
        self.parameters = np.asarray(parameters, dtype=float)
        self.qois = np.asarray(qois, dtype=float)
        self.pod_energy_fraction = float(pod_energy_fraction)
        self.max_pod_modes = max_pod_modes
        self.network_config = network_config if network_config is not None else NeuralNetworkConfig()
        self.normalize_parameters = bool(normalize_parameters)
        self.normalize_targets = bool(normalize_targets)

        if self.parameters.ndim != 2:
            raise ValueError("parameters must have shape (num_samples, parameter_dimension)")
        if self.parameters.shape[0] < 1:
            raise ValueError("at least one training sample is required")
        if self.qois.ndim not in (1, 2):
            raise ValueError("qois must have shape (num_samples,) or (num_samples, qoi_dimension)")
        if self.qois.shape[0] != self.parameters.shape[0]:
            raise ValueError("parameters and qois must contain the same number of samples")
        if not 0.0 < self.pod_energy_fraction <= 1.0:
            raise ValueError("pod_energy_fraction must be in (0, 1]")
        if self.max_pod_modes is not None and self.max_pod_modes < 1:
            raise ValueError("max_pod_modes must be positive")
        self.network_config.validate()

        self._mean_qoi: Optional[np.ndarray] = None
        self._pod_modes: Optional[np.ndarray] = None
        self._param_min: Optional[np.ndarray] = None
        self._param_scale: Optional[np.ndarray] = None
        self._target_min: Optional[np.ndarray] = None
        self._target_scale: Optional[np.ndarray] = None
        self._network: Optional[Any] = None
        self._torch = None
        self._torch_dtype = None
        self.hidden_neurons_per_layer = self.network_config.hidden_layer_width(
            self.parameters.shape[1]
        )
        self.final_training_loss: Optional[float] = None
        self._fit()

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        return None

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        coefficients = self._predict_reduced_targets(parameter_sample)
        if self._pod_modes is None:
            return np.asarray([coefficients[0]], dtype=float)
        qoi = self._mean_qoi + self._pod_modes @ coefficients
        return np.asarray(qoi, dtype=float).ravel()

    def _parameter_sample_to_array(self, parameter_sample) -> np.ndarray:
        if isinstance(parameter_sample, dict):
            if self.parameter_names is None:
                raise ValueError("parameter_names must be provided to map dict inputs.")
            parameters = np.array(
                [parameter_sample[name] for name in self.parameter_names], dtype=float
            )
        else:
            parameters = np.asarray(parameter_sample, dtype=float).ravel()
        if parameters.size != self.parameters.shape[1]:
            raise ValueError(
                "parameter sample dimension does not match the network input dimension"
            )
        return self._scale_parameters(parameters)

    def _fit(self) -> None:
        torch = _import_torch()
        self._torch = torch
        try:
            self._torch_dtype = getattr(torch, self.network_config.dtype)
        except AttributeError as exc:
            raise ValueError(
                f"Unsupported torch dtype '{self.network_config.dtype}'."
            ) from exc
        device = torch.device(self.network_config.device)

        parameters = self._initialize_parameter_scaling()
        targets = self._build_reduced_targets()
        targets = self._initialize_target_scaling(targets)

        torch.manual_seed(self.network_config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.network_config.random_seed)

        input_dimension = parameters.shape[1]
        output_dimension = targets.shape[1]
        self._network = self._build_network(
            input_dimension, output_dimension, torch
        ).to(device=device, dtype=self._torch_dtype)

        x_train = torch.as_tensor(
            parameters, dtype=self._torch_dtype, device=device
        )
        y_train = torch.as_tensor(
            targets, dtype=self._torch_dtype, device=device
        )
        optimizer = self._build_optimizer(torch)
        loss_function = torch.nn.MSELoss()

        self._network.train()
        for _ in range(self.network_config.training_iterations):
            optimizer.zero_grad()
            prediction = self._network(x_train)
            loss = loss_function(prediction, y_train)
            loss.backward()
            optimizer.step()
        self.final_training_loss = float(loss.detach().cpu().item())
        self._network.eval()

    def _build_reduced_targets(self) -> np.ndarray:
        if self.qois.ndim == 1 or self.qois.shape[1] == 1:
            self._mean_qoi = np.zeros(1, dtype=float)
            self._pod_modes = None
            return self.qois.reshape(-1, 1)

        mean_qoi = np.mean(self.qois, axis=0)
        centered = self.qois - mean_qoi[None, :]
        _, singular_values, right_singular_vectors = np.linalg.svd(
            centered, full_matrices=False
        )
        cumulative_energy = np.cumsum(singular_values ** 2)
        total_energy = cumulative_energy[-1] if cumulative_energy.size > 0 else 0.0
        if total_energy == 0.0:
            modes = np.zeros((self.qois.shape[1], 1), dtype=float)
        else:
            num_modes = int(
                np.searchsorted(
                    cumulative_energy / total_energy, self.pod_energy_fraction
                ) + 1
            )
            if self.max_pod_modes is not None:
                num_modes = min(num_modes, self.max_pod_modes)
            num_modes = max(num_modes, 1)
            modes = right_singular_vectors[:num_modes, :].T

        self._mean_qoi = mean_qoi
        self._pod_modes = modes
        return centered @ modes

    def _initialize_parameter_scaling(self) -> np.ndarray:
        if not self.normalize_parameters:
            self._param_min = None
            self._param_scale = None
            return self.parameters
        self._param_min = np.min(self.parameters, axis=0)
        parameter_max = np.max(self.parameters, axis=0)
        self._param_scale = parameter_max - self._param_min
        self._param_scale[self._param_scale == 0.0] = 1.0
        return (self.parameters - self._param_min) / self._param_scale

    def _scale_parameters(self, parameters: np.ndarray) -> np.ndarray:
        if not self.normalize_parameters:
            return np.asarray(parameters, dtype=float)
        return (np.asarray(parameters, dtype=float) - self._param_min) / self._param_scale

    def _initialize_target_scaling(self, targets: np.ndarray) -> np.ndarray:
        targets = np.asarray(targets, dtype=float)
        if not self.normalize_targets:
            self._target_min = None
            self._target_scale = None
            return targets
        self._target_min = np.min(targets, axis=0)
        target_max = np.max(targets, axis=0)
        self._target_scale = target_max - self._target_min
        self._target_scale[self._target_scale == 0.0] = 1.0
        return (targets - self._target_min[None, :]) / self._target_scale[None, :]

    def _unscale_targets(self, targets: np.ndarray) -> np.ndarray:
        targets = np.asarray(targets, dtype=float).ravel()
        if not self.normalize_targets:
            return targets
        return targets * self._target_scale + self._target_min

    def _build_network(self, input_dimension: int, output_dimension: int, torch):
        activation_type = self._activation_type(torch)
        layers = []
        previous_width = input_dimension
        for _ in range(self.network_config.num_hidden_layers):
            layers.append(torch.nn.Linear(previous_width, self.hidden_neurons_per_layer))
            layers.append(activation_type())
            previous_width = self.hidden_neurons_per_layer
        layers.append(torch.nn.Linear(previous_width, output_dimension))
        return torch.nn.Sequential(*layers)

    def _activation_type(self, torch):
        activations = {
            "tanh": torch.nn.Tanh,
            "relu": torch.nn.ReLU,
            "gelu": torch.nn.GELU,
            "silu": torch.nn.SiLU,
        }
        activation = self.network_config.activation.strip().lower()
        if activation not in activations:
            raise ValueError(
                f"Unsupported neural-network activation '{self.network_config.activation}'."
            )
        return activations[activation]

    def _build_optimizer(self, torch):
        optimizer_name = self.network_config.optimizer.strip().lower()
        optimizer_options = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }
        if optimizer_name not in optimizer_options:
            raise ValueError(
                f"Unsupported neural-network optimizer '{self.network_config.optimizer}'."
            )
        return optimizer_options[optimizer_name](
            self._network.parameters(),
            lr=self.network_config.learning_rate,
            weight_decay=self.network_config.weight_decay,
        )

    def _predict_reduced_targets(self, parameter_sample) -> np.ndarray:
        parameters = self._parameter_sample_to_array(parameter_sample)
        device = next(self._network.parameters()).device
        x = self._torch.as_tensor(
            parameters[None, :], dtype=self._torch_dtype, device=device
        )
        with self._torch.no_grad():
            prediction = self._network(x).detach().cpu().numpy().ravel()
        return self._unscale_targets(prediction)
