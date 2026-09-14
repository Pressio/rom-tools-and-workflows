"""Shared configuration/model helpers for the H2-air flame VI benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MODELS_PATH = REPOSITORY_ROOT / "examples" / "models"
for path in (REPOSITORY_ROOT, EXAMPLE_MODELS_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from h2_air_flame_model import H2AirFlameQoiModel  # noqa: E402
from romtools.workflows.inverse import VISampleReuseConfig  # noqa: E402
from romtools.workflows.inverse.vi_optimization_methods import (  # noqa: E402
    VIAdamOptimizerConfig,
    VINewtonOptimizerConfig,
    VIStochasticNonmonotoneLineSearchConfig,
)
from romtools.workflows.parameter_spaces import (  # noqa: E402
    GaussianParameterSpace,
    MonteCarloSampler,
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return data


def jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def parameter_arrays(config: dict):
    p = config["parameters"]
    names = list(p["names"])
    arrays = [
        np.asarray(p[key], dtype=float)
        for key in (
            "mins",
            "maxes",
            "truth",
            "prior_mean",
            "prior_std",
            "initial_mean",
            "initial_std",
        )
    ]
    mins, maxes, truth, prior_mean, prior_std, initial_mean, initial_std = arrays
    if any(array.shape != (len(names),) for array in arrays):
        raise ValueError("All parameter vectors must have one entry per parameter name.")
    if np.any(maxes <= mins):
        raise ValueError("Parameter maxima must be larger than parameter minima.")
    if np.any((truth <= mins) | (truth >= maxes)):
        raise ValueError("The synthetic truth must lie strictly inside the bounds.")
    if np.allclose(initial_mean, truth):
        raise ValueError("initial_mean must not be centered on the synthetic truth.")
    return names, mins, maxes, truth, prior_mean, prior_std, initial_mean, initial_std


def make_parameter_spaces(config: dict):
    names, mins, maxes, truth, prior_mean, prior_std, initial_mean, initial_std = (
        parameter_arrays(config)
    )
    prior = GaussianParameterSpace(names, prior_mean, prior_std, MonteCarloSampler)
    initial = GaussianParameterSpace(names, initial_mean, initial_std, MonteCarloSampler)
    return names, mins, maxes, truth, prior, initial


def make_model(config: dict) -> H2AirFlameQoiModel:
    model = config["model"]
    return H2AirFlameQoiModel(
        nx=int(model["nx"]),
        ny=int(model["ny"]),
        dt=float(model["dt"]),
        t_end=float(model["t_end"]),
        snapshot_stride=int(model["snapshot_stride"]),
        spatial_sensor_stride=int(model["spatial_sensor_stride"]),
        temporal_sensor_stride=int(model["temporal_sensor_stride"]),
    )


def truth_dict(names: Iterable[str], truth: np.ndarray):
    return {name: float(value) for name, value in zip(names, truth)}


def prepare_observations(config: dict, root: Path, names: list[str], truth: np.ndarray):
    """Generate the one noisy synthetic observation shared by every run."""
    model = make_model(config)
    parameters = truth_dict(names, truth)
    truth_dir = root / "observation" / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    model.populate_run_directory(str(truth_dir), parameters)
    model.run_model(str(truth_dir), parameters)
    clean = np.asarray(model.compute_qoi(str(truth_dir), parameters), dtype=float)

    obs = config["observations"]
    ratio = float(obs["global_rms_noise_ratio"])
    seed = int(obs["seed"])
    sigma = ratio * np.linalg.norm(clean) / np.sqrt(clean.size)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, clean.shape)
    noisy = clean + noise
    covariance = np.eye(clean.size) * sigma**2
    realized_ratio = float(np.linalg.norm(noise) / max(np.linalg.norm(clean), 1e-300))

    observation_dir = root / "observation"
    observation_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        observation_dir / "observation.npz",
        clean=clean,
        noisy=noisy,
        noise=noise,
        covariance=covariance,
        sigma_obs=sigma,
        observation_seed=seed,
        realized_noise_ratio=realized_ratio,
    )
    metadata = {
        "seed": seed,
        "sigma_obs": float(sigma),
        "clean": clean,
        "noisy": noisy,
        "noise": noise,
        "realized_noise_ratio": realized_ratio,
        "qoi_dimension": int(clean.size),
    }
    return noisy, covariance, metadata


def optimizer_and_line_search(config: dict, optimizer_name: str):
    optimization = config["optimization"]
    max_iterations = int(config["benchmark"]["max_iterations"])
    if optimizer_name == "adam":
        s = optimization.get("adam", {})
        return VIAdamOptimizerConfig(
            gradient_method=s.get("gradient_method", "natural"),
            learning_rate=s.get("learning_rate"),
            learning_rate_scale=float(s.get("learning_rate_scale", 0.1)),
            beta1=float(s.get("beta1", 0.9)),
            beta2=float(s.get("beta2", 0.999)),
            epsilon=float(s.get("epsilon", 1e-8)),
            fisher_damping_initial=float(s.get("fisher_damping_initial", 1e-2)),
            fisher_damping_decay_start=int(s.get("fisher_damping_decay_start", 50)),
            fisher_damping_min=float(s.get("fisher_damping_min", 1e-6)),
            gradient_clip_norm=float(s.get("gradient_clip_norm", 1e6)),
            gradient_norm_tolerance=float(s.get("gradient_norm_tolerance", 0.0)),
            max_iterations=max_iterations,
            max_log_std_update=float(s.get("max_log_std_update", 0.5)),
            min_variational_std=float(s.get("min_variational_std", 1e-6)),
            max_variational_std=float(s.get("max_variational_std", 1e6)),
        ), None

    if optimizer_name != "newton":
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")
    s = optimization.get("newton", {})
    optimizer = VINewtonOptimizerConfig(
        gradient_norm_tolerance=float(s.get("gradient_norm_tolerance", 0.0)),
        max_iterations=max_iterations,
        max_log_std_update=float(s.get("max_log_std_update", 0.5)),
        max_mean_update_std=s.get("max_mean_update_std"),
        min_variational_std=float(s.get("min_variational_std", 1e-8)),
        max_variational_std=float(s.get("max_variational_std", 1e6)),
        newton_metric=s.get("newton_metric", "standard"),
        newton_regularization=float(s.get("newton_regularization", 1e-2)),
        newton_hessian_type=s.get("newton_hessian_type", "full"),
        newton_curvature_strategy=s.get("newton_curvature_strategy", "lagged"),
        newton_hessian_num_samples=s.get("newton_hessian_num_samples"),
        newton_hessian_averaging_factor=float(s.get("newton_hessian_averaging_factor", 0.5)),
    )
    s = optimization.get("line_search", {})
    line_search = VIStochasticNonmonotoneLineSearchConfig(
        initial_step_size=float(s.get("initial_step_size", 1e-2)),
        max_step_size=float(s.get("max_step_size", 1.0)),
        step_size_growth_factor=float(s.get("step_size_growth_factor", 1.05)),
        step_size_decay_factor=float(s.get("step_size_decay_factor", 2.0)),
        max_step_size_decrease_trys=int(s.get("max_step_size_decrease_trys", 5)),
        relaxation_parameter=float(s.get("relaxation_parameter", 3.05)),
        line_search_objective=s.get("line_search_objective", "elbo"),
        line_search_nonmonotone_window=int(s.get("line_search_nonmonotone_window", 5)),
        line_search_armijo_coefficient=float(s.get("line_search_armijo_coefficient", 1e-6)),
        line_search_uncertainty_sigma=float(s.get("line_search_uncertainty_sigma", 4.0)),
        line_search_sample_growth_factor=float(s.get("line_search_sample_growth_factor", 1.0)),
        log_std_learning_rate_factor=float(s.get("log_std_learning_rate_factor", 1.0)),
    )
    return optimizer, line_search


def reuse_config(config: dict, enabled: bool):
    if not enabled:
        return None
    s = config.get("sample_reuse", {})
    return VISampleReuseConfig(
        enabled=True,
        history_batches=int(s.get("history_batches", 10)),
        ess_threshold=s.get("ess_threshold"),
        use_score_diagnostic=bool(s.get("use_score_diagnostic", True)),
        score_error_scale=float(s.get("score_error_scale", 1.0)),
        periodic_refresh=s.get("periodic_refresh", 100),
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
