"""Importance-sampling sample reuse for VI and MFVI.

This module implements an opt-in ABRIS-style reuse layer around ROMTools'
existing VI drivers. Expensive high-fidelity evaluations are cached in batches
together with the variational distribution that generated them.  A bounded
inner sampling loop adds fresh batches until the reuse diagnostics pass, then
forms the update from the complete retained archive.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import re
from typing import Optional

import numpy as np

from romtools.hpc.dispatchers import resolve_dispatcher, resolve_local_dispatcher
from romtools.workflows.inverse.vi import drivers as _vi
from romtools.workflows.inverse.vi import mf_drivers as _mf
from romtools.workflows.inverse.vi.optimization import _normalize_optimization_method


@dataclass(frozen=True)
class VISampleReuseConfig:
    """Configuration for ABRIS-style reuse of previous VI samples.

    ``history_batches`` controls both the maximum moving-window size and the
    maximum number of fresh batches allowed in one state evaluation.  During
    zero-based iteration ``i``, both limits grow as
    ``min(history_batches, i + 1)``.  Every returned update is formed from the
    retained archive, including evaluations that sampled fresh batches.
    """

    enabled: bool = True
    history_batches: int = 10
    ess_threshold: Optional[float] = None
    use_score_diagnostic: bool = True
    score_error_scale: float = 1.0
    periodic_refresh: Optional[int] = 100

    use_hessian_score_diagnostic: bool = True
    hessian_score_error_scale: float = 2.0
    hessian_relative_standard_error_threshold: Optional[float] = 0.25

    def __post_init__(self):
        if self.history_batches < 1:
            raise ValueError("history_batches must be at least 1")
        if self.ess_threshold is not None and self.ess_threshold <= 0.0:
            raise ValueError("ess_threshold must be positive when provided")
        if not np.isfinite(self.score_error_scale) or self.score_error_scale < 0.0:
            raise ValueError("score_error_scale must be finite and non-negative")
        if self.periodic_refresh is not None and self.periodic_refresh < 1:
            raise ValueError("periodic_refresh must be at least 1 when provided")

        if (
            not np.isfinite(self.hessian_score_error_scale)
            or self.hessian_score_error_scale < 0.0
        ):
            raise ValueError(
                "hessian_score_error_scale must be finite and non-negative"
            )
        threshold = self.hessian_relative_standard_error_threshold
        if threshold is not None and (
            not np.isfinite(threshold) or threshold <= 0.0
        ):
            raise ValueError(
                "hessian_relative_standard_error_threshold must be positive "
                "and finite when provided"
            )


@dataclass
class _ReuseBatch:
    optimizer_samples: np.ndarray
    parameter_samples: np.ndarray
    qois: np.ndarray
    errors: np.ndarray
    variational_mean: np.ndarray
    variational_log_std: np.ndarray
    variational_correlation_cholesky: Optional[np.ndarray]
    iteration: int

    @property
    def size(self) -> int:
        return int(self.optimizer_samples.shape[0])


class _ReuseArchive:
    def __init__(self, config: VISampleReuseConfig):
        self.config = config
        self.optimization_method = "gradient"
        self.batches: list[_ReuseBatch] = []

    def append(self, batch: _ReuseBatch, capacity: Optional[int] = None) -> None:
        self.batches.append(batch)
        retained = self.config.history_batches if capacity is None else int(capacity)
        if retained < 1:
            raise ValueError("archive capacity must be at least 1")
        if len(self.batches) > retained:
            self.batches = self.batches[-retained:]

    @property
    def sample_count(self) -> int:
        return sum(batch.size for batch in self.batches)


def _iteration_from_path(path: str) -> int:
    match = re.search(r"iteration_(\d+)", str(path))
    return 0 if match is None else int(match.group(1))


def _variational_std(log_std: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(log_std, dtype=float))


def _log_q(samples: np.ndarray,
           mean: np.ndarray,
           log_std: np.ndarray,
           correlation_cholesky: Optional[np.ndarray]) -> np.ndarray:
    return _vi._compute_variational_log_densities(
        samples,
        np.asarray(mean, dtype=float),
        _variational_std(log_std),
        correlation_cholesky,
    )


def _logsumexp(values: np.ndarray, axis: int = 0) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    finite_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    shifted = values - finite_maximum
    result = finite_maximum + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _archive_arrays(archive: _ReuseArchive):
    optimizer_samples = np.vstack([batch.optimizer_samples for batch in archive.batches])
    parameter_samples = np.vstack([batch.parameter_samples for batch in archive.batches])
    qois = np.hstack([batch.qois for batch in archive.batches])
    errors = np.hstack([batch.errors for batch in archive.batches])
    return optimizer_samples, parameter_samples, qois, errors


def _compute_importance_weights(archive: _ReuseArchive,
                                current_mean: np.ndarray,
                                current_log_std: np.ndarray,
                                current_correlation_cholesky: Optional[np.ndarray]):
    samples, _, _, _ = _archive_arrays(archive)
    current_log_density = _log_q(
        samples, current_mean, current_log_std, current_correlation_cholesky
    )

    batch_sizes = np.asarray([batch.size for batch in archive.batches], dtype=float)
    log_beta = np.log(batch_sizes / np.sum(batch_sizes))
    component_logs = []
    for beta, batch in zip(log_beta, archive.batches):
        component_logs.append(
            beta
            + _log_q(
                samples,
                batch.variational_mean,
                batch.variational_log_std,
                batch.variational_correlation_cholesky,
            )
        )
    mixture_log_density = _logsumexp(np.vstack(component_logs), axis=0)
    log_weights = current_log_density - mixture_log_density
    weights = np.exp(np.clip(log_weights, -50.0, 50.0))

    origin_weights = np.empty(samples.shape[0], dtype=float)
    offset = 0
    for batch in archive.batches:
        sl = slice(offset, offset + batch.size)
        origin_log_density = _log_q(
            samples[sl],
            batch.variational_mean,
            batch.variational_log_std,
            batch.variational_correlation_cholesky,
        )
        log_origin_ratio = current_log_density[sl] - origin_log_density
        origin_weights[sl] = np.exp(np.clip(log_origin_ratio, -50.0, 50.0))
        offset += batch.size
    return weights, origin_weights


def _effective_sample_size(weights: np.ndarray) -> float:
    numerator = float(np.sum(weights)) ** 2
    denominator = float(np.sum(np.asarray(weights) ** 2))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 0.0
    return numerator / denominator


def _score_functions(samples: np.ndarray,
                     mean: np.ndarray,
                     log_std: np.ndarray,
                     correlation_cholesky: Optional[np.ndarray]):
    std = _variational_std(log_std)
    centered = samples - mean[None, :]
    normalized = centered / std[None, :]
    if correlation_cholesky is None:
        score_mean = centered / (std[None, :] ** 2)
        score_log_std = normalized ** 2 - 1.0
    else:
        correlation_solve = np.linalg.solve(
            correlation_cholesky, normalized.transpose()
        )
        inverse_times_normalized = np.linalg.solve(
            correlation_cholesky.transpose(), correlation_solve
        ).transpose()
        score_mean = inverse_times_normalized / std[None, :]
        score_log_std = normalized * inverse_times_normalized - 1.0
    return score_mean, score_log_std


def _draw_optimizer_only(mean: np.ndarray,
                         log_std: np.ndarray,
                         sample_count: int,
                         correlation_cholesky: Optional[np.ndarray]):
    standard_normal = np.random.randn(sample_count, mean.size)
    if correlation_cholesky is not None:
        standard_normal = standard_normal @ correlation_cholesky.transpose()
    return mean[None, :] + _variational_std(log_std)[None, :] * standard_normal


def _score_diagnostics(archive: _ReuseArchive,
                       current_mean: np.ndarray,
                       current_log_std: np.ndarray,
                       current_correlation_cholesky: Optional[np.ndarray],
                       weights: np.ndarray,
                       reference_sample_count: int):
    samples, _, _, _ = _archive_arrays(archive)
    score_mean, score_log_std = _score_functions(
        samples, current_mean, current_log_std, current_correlation_cholesky
    )
    recycled_error = np.mean(
        weights[:, None] * np.hstack([score_mean, score_log_std]), axis=0
    )
    reference_samples = _draw_optimizer_only(
        current_mean,
        current_log_std,
        reference_sample_count,
        current_correlation_cholesky,
    )
    ref_mean, ref_log_std = _score_functions(
        reference_samples,
        current_mean,
        current_log_std,
        current_correlation_cholesky,
    )
    reference_error = np.mean(np.hstack([ref_mean, ref_log_std]), axis=0)
    # The inverse Fisher metric for mean-field Gaussian parameters
    # (mean, log-standard-deviation) is diag(std**2, 1/2).
    metric_diagonal = np.concatenate([
        _variational_std(current_log_std) ** 2,
        0.5 * np.ones_like(current_log_std, dtype=float),
    ])

    def metric_norm(error):
        return float(np.sqrt(np.sum(metric_diagonal * np.asarray(error) ** 2)))

    return metric_norm(recycled_error), metric_norm(reference_error)


def _weighted_loo_baseline(values: np.ndarray, origin_weights: np.ndarray) -> np.ndarray:
    """Origin-proposal LOO baseline for a heterogeneous proposal archive."""
    values = np.asarray(values, dtype=float)
    origin_weights = np.asarray(origin_weights, dtype=float)
    if values.size <= 1:
        return np.zeros_like(values)
    weighted_values = origin_weights * values
    return (np.sum(weighted_values) - weighted_values) / float(values.size - 1)


def _gradient_from_archive(samples: np.ndarray,
                           values: np.ndarray,
                           mean: np.ndarray,
                           log_std: np.ndarray,
                           correlation_cholesky: Optional[np.ndarray],
                           importance_weights: np.ndarray,
                           origin_weights: np.ndarray,
                           baseline_method: str,
                           elbo_scaling_factor: float,
                           entropy_strategy: str):
    score_mean, score_log_std = _score_functions(
        samples, mean, log_std, correlation_cholesky
    )
    gradient_values = np.asarray(values, dtype=float)
    if entropy_strategy == "joint":
        gradient_values = gradient_values - elbo_scaling_factor * _log_q(
            samples, mean, log_std, correlation_cholesky
        )

    if baseline_method == "loo":
        centered = gradient_values - _weighted_loo_baseline(
            gradient_values, origin_weights
        )
    elif baseline_method == "none":
        centered = gradient_values
    else:
        raise NotImplementedError(
            "Sample reuse currently supports baseline_method='none' or 'loo'."
        )

    weighted_mean_terms = importance_weights[:, None] * centered[:, None] * score_mean
    weighted_log_std_terms = (
        importance_weights[:, None] * centered[:, None] * score_log_std
    )
    gradient_mean = np.mean(weighted_mean_terms, axis=0)
    gradient_log_std = np.mean(weighted_log_std_terms, axis=0)
    if entropy_strategy == "analytic":
        gradient_log_std += elbo_scaling_factor
    standard_error = np.concatenate([
        _vi._compute_component_standard_error(weighted_mean_terms),
        _vi._compute_component_standard_error(weighted_log_std_terms),
    ])
    snr = _vi._compute_gradient_signal_to_noise_ratio(
        np.concatenate([gradient_mean, gradient_log_std]), standard_error
    )
    zeros = np.zeros_like(mean)
    return gradient_mean, gradient_log_std, zeros, zeros, snr, standard_error


def _refresh_decision(archive: _ReuseArchive,
                      current_mean: np.ndarray,
                      current_log_std: np.ndarray,
                      current_correlation_cholesky: Optional[np.ndarray],
                      nominal_batch_size: int,
                      periodic_due: bool = False):
    if not archive.batches:
        return True, ("empty",), None, np.nan

    weights, origin_weights = _compute_importance_weights(
        archive,
        current_mean,
        current_log_std,
        current_correlation_cholesky,
    )
    if not np.all(np.isfinite(weights)) or np.sum(weights) <= 0.0:
        return True, ("nonfinite_weights",), (weights, origin_weights), 0.0

    ess = _effective_sample_size(weights)
    threshold = (
        float(nominal_batch_size)
        if archive.config.ess_threshold is None
        else float(archive.config.ess_threshold)
    )
    reasons = []
    if ess <= threshold:
        reasons.append("ess")

    if periodic_due:
        reasons.append("periodic")

    if archive.config.use_score_diagnostic:
        recycled_error, reference_error = _score_diagnostics(
            archive,
            current_mean,
            current_log_std,
            current_correlation_cholesky,
            weights,
            nominal_batch_size,
        )
        reference_scale = max(reference_error, np.sqrt(np.finfo(float).eps))
        if recycled_error > archive.config.score_error_scale * reference_scale:
            reasons.append("score")

    if reasons:
        return True, tuple(reasons), (weights, origin_weights), ess
    return False, (), (weights, origin_weights), ess


def _iteration_archive_limit(config: VISampleReuseConfig, iteration: int) -> int:
    """Return the zero-based ABRIS moving-window and sampling-loop limit."""
    return min(config.history_batches, max(int(iteration) + 1, 1))


def _annotate_reuse_state(state,
                          archive: _ReuseArchive,
                          ess: float,
                          refresh_reasons,
                          refresh_limit_reached: bool,
                          diagnostics=None):
    reasons = tuple(refresh_reasons)
    state["sample_reuse_used"] = True
    state["sample_reuse_refreshed"] = bool(reasons)
    state["sample_reuse_refresh_count"] = len(reasons)
    state["sample_reuse_refresh_reasons"] = reasons
    state["sample_reuse_refresh_reason"] = reasons[-1] if reasons else "reuse"
    state["sample_reuse_refresh_limit_reached"] = bool(refresh_limit_reached)
    state["sample_reuse_final_update_used_archive"] = True
    state["sample_reuse_ess"] = float(ess)
    state["sample_reuse_archive_samples"] = archive.sample_count
    state["sample_reuse_archive_batches"] = len(archive.batches)
    if diagnostics:
        state.update(diagnostics)
    return state


def _batch_from_vi_state(state,
                         mean: np.ndarray,
                         log_std: np.ndarray,
                         correlation_cholesky: Optional[np.ndarray],
                         iteration: int) -> _ReuseBatch:
    return _ReuseBatch(
        optimizer_samples=np.asarray(state["optimizer_samples"]).copy(),
        parameter_samples=np.asarray(state["parameter_samples"]).copy(),
        qois=np.asarray(state["qois"]).copy(),
        errors=np.asarray(state["errors"]).copy(),
        variational_mean=np.asarray(mean).copy(),
        variational_log_std=np.asarray(log_std).copy(),
        variational_correlation_cholesky=(
            None if correlation_cholesky is None
            else np.asarray(correlation_cholesky).copy()
        ),
        iteration=iteration,
    )


def _build_reused_vi_state(a, archive: _ReuseArchive, weights, origin_weights,
                           ess: float):
    optimizer_samples, parameter_samples, qois, errors = _archive_arrays(archive)
    mean = np.asarray(a["variational_mean"])
    log_std = np.asarray(a["variational_log_std"])
    std, log_std = _vi._compute_variational_std(
        log_std, a["min_variational_std"], a["max_variational_std"]
    )
    corr = a.get("variational_correlation_cholesky")
    observations = np.asarray(a["observations"])
    log_likelihoods, misfits = _vi._compute_log_likelihoods(
        errors,
        a["observations_covariance"],
        a["covariance_regularization"],
        precision_operator=a.get("log_likelihood_precision_operator"),
    )
    log_priors, log_jacobian, log_joint = _vi._compute_log_prior_and_joint_terms(
        log_likelihoods,
        parameter_samples,
        optimizer_samples,
        a["prior_mean"],
        a["prior_precision_operator"],
        a["prior_covariance_log_det"],
        a["bounded_parameter_handling"],
        a["parameter_mins"],
        a["parameter_maxes"],
        a["transform_interior_margin"],
        a["transform_map"],
    )
    scale = float(a["elbo_scaling_factor"])
    entropy_strategy = _vi._normalize_score_function_entropy_strategy(
        a["score_function_entropy_strategy"]
    )
    gradient = _gradient_from_archive(
        optimizer_samples,
        scale * log_joint,
        mean,
        log_std,
        corr,
        weights,
        origin_weights,
        _vi._normalize_baseline_method(a["baseline_method"]),
        scale,
        entropy_strategy,
    )
    gradient_mean, gradient_log_std = gradient[:2]
    update_mean, update_log_std, gradient_method = _vi._compute_update_directions(
        gradient_mean, gradient_log_std, std, a["gradient_method"]
    )
    dimensionality = mean.size
    entropy = np.sum(log_std) + 0.5 * dimensionality * (1.0 + np.log(2.0 * np.pi))
    if corr is not None:
        entropy += np.sum(np.log(np.diag(corr)))
    entropy *= scale
    weighted_joint = weights * log_joint
    normalized_weights = weights / np.sum(weights)
    zeros = np.zeros(mean.size)
    state = {
        "optimizer_samples": optimizer_samples,
        "parameter_samples": parameter_samples,
        "qois": qois,
        "mean_qoi": qois @ normalized_weights,
        "errors": errors,
        "log_likelihoods": weights * log_likelihoods,
        "raw_log_likelihoods": log_likelihoods,
        "log_priors": weights * log_priors,
        "raw_log_priors": log_priors,
        "log_joint_terms": weighted_joint,
        "raw_log_joint_terms": log_joint,
        "log_transform_jacobian": log_jacobian,
        "mean_misfit": float(np.mean(weights * misfits)),
        "mean_relative_mse": float(np.mean(
            weights * _vi._compute_relative_mse(errors, observations)
        )),
        "entropy": entropy,
        "elbo": scale * float(np.mean(weighted_joint)) + entropy,
        "gradient_mean": gradient_mean,
        "gradient_log_std": gradient_log_std,
        "hessian_diagonal_mean": zeros.copy(),
        "hessian_diagonal_log_std": zeros.copy(),
        "hessian_full": np.zeros((2 * mean.size, 2 * mean.size)),
        "update_direction_mean": update_mean,
        "update_direction_log_std": update_log_std,
        "gradient_method": gradient_method,
        "baseline_mean": gradient[2],
        "baseline_log_std": gradient[3],
        "gradient_signal_to_noise_ratio": gradient[4],
        "gradient_standard_error": gradient[5],
        "sample_reuse_used": True,
        "sample_reuse_ess": float(ess),
        "sample_reuse_archive_samples": archive.sample_count,
        "sample_reuse_archive_batches": len(archive.batches),
        "sample_reuse_refresh_reason": "reuse",
        "importance_weights": weights,
        "importance_origin_weights": origin_weights,
    }

    _, clipped_log_std = _vi._compute_variational_std(
        np.asarray(a["variational_log_std"]),
        a["min_variational_std"],
        a["max_variational_std"],
    )
    hessian_full = _hessian_from_archive(
        state["optimizer_samples"],
        float(a["elbo_scaling_factor"]) * state["raw_log_joint_terms"],
        np.asarray(a["variational_mean"]),
        clipped_log_std,
        a.get("variational_correlation_cholesky"),
        weights,
        origin_weights,
        a["baseline_method"],
    )
    dimensionality = np.asarray(a["variational_mean"]).size
    state["hessian_full"] = hessian_full
    state["hessian_diagonal_mean"] = np.diag(
        hessian_full[:dimensionality, :dimensionality]
    )
    state["hessian_diagonal_log_std"] = np.diag(
        hessian_full[dimensionality:, dimensionality:]
    )
    return state


class _VIReuseController:
    def __init__(self, config: VISampleReuseConfig):
        self.archive = _ReuseArchive(config)
        self._periodic_refresh_iterations = set()

    def _periodic_due(self, iteration: int) -> bool:
        period = self.archive.config.periodic_refresh
        return (
            period is not None
            and iteration > 0
            and iteration % period == 0
            and iteration not in self._periodic_refresh_iterations
        )

    def _additional_reuse_quality(self, a, state, weights, origin_weights):
        """Run Newton safeguards inside the controller's shared refresh loop."""
        if not _is_newton_reuse(self.archive):
            return True, "reuse", {}
        _, clipped_log_std = _vi._compute_variational_std(
            np.asarray(a["variational_log_std"]),
            a["min_variational_std"],
            a["max_variational_std"],
        )
        return _hessian_reuse_quality(
            self.archive,
            np.asarray(state["optimizer_samples"]),
            float(a["elbo_scaling_factor"]) * np.asarray(state["raw_log_joint_terms"]),
            np.asarray(a["variational_mean"]),
            clipped_log_std,
            a.get("variational_correlation_cholesky"),
            np.asarray(weights),
            np.asarray(origin_weights),
            a["baseline_method"],
            np.asarray(state["hessian_full"]),
            int(a["sample_size"]),
        )


    def _evaluate_current_archive(self, *args, **kwargs):
        bound = inspect.signature(_vi._evaluate_vi_state).bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        iteration = _iteration_from_path(a["run_directory_base"])
        limit = _iteration_archive_limit(self.archive.config, iteration)
        refresh_reasons = []
        refresh_limit_reached = False
        final_diagnostics = {}

        while True:
            periodic_due = self._periodic_due(iteration)
            refresh, reasons, weight_pair, ess = _refresh_decision(
                self.archive,
                np.asarray(a["variational_mean"]),
                np.asarray(a["variational_log_std"]),
                a.get("variational_correlation_cholesky"),
                int(a["sample_size"]),
                periodic_due=periodic_due,
            )
            candidate = None
            if not refresh:
                weights, origin_weights = weight_pair
                candidate = _build_reused_vi_state(
                    a, self.archive, weights, origin_weights, ess
                )
                quality_ok, reason, final_diagnostics = (
                    self._additional_reuse_quality(
                        a, candidate, weights, origin_weights
                    )
                )
                if not quality_ok:
                    refresh = True
                    reasons = (reason,)

            if not refresh:
                return _annotate_reuse_state(
                    candidate,
                    self.archive,
                    ess,
                    refresh_reasons,
                    refresh_limit_reached,
                    final_diagnostics,
                )

            if len(refresh_reasons) >= limit:
                refresh_limit_reached = True
                if candidate is None:
                    weights, origin_weights = _compute_importance_weights(
                        self.archive,
                        np.asarray(a["variational_mean"]),
                        np.asarray(a["variational_log_std"]),
                        a.get("variational_correlation_cholesky"),
                    )
                    ess = _effective_sample_size(weights)
                    candidate = _build_reused_vi_state(
                        a, self.archive, weights, origin_weights, ess
                    )
                return _annotate_reuse_state(
                    candidate,
                    self.archive,
                    ess,
                    refresh_reasons,
                    refresh_limit_reached,
                    final_diagnostics,
                )

            reason = "+".join(reasons)
            if periodic_due:
                self._periodic_refresh_iterations.add(iteration)
            fresh_kwargs = dict(a)
            if refresh_reasons:
                fresh_kwargs["run_directory_base"] = (
                    f'{a["run_directory_base"]}abris_refresh_'
                    f'{len(refresh_reasons)}_'
                )
            fresh_state = _vi._evaluate_vi_state(**fresh_kwargs)
            self.archive.append(
                _batch_from_vi_state(
                    fresh_state,
                    a["variational_mean"],
                    a["variational_log_std"],
                    a.get("variational_correlation_cholesky"),
                    iteration,
                ),
                capacity=limit,
            )
            refresh_reasons.append(reason)

    def evaluate_candidate(self, *args, **kwargs):
        bound = inspect.signature(_vi._evaluate_vi_candidate_for_line_search).bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        if a["line_search_objective"] != "elbo":
            raise NotImplementedError(
                "Sample reuse currently requires line_search_objective='elbo'."
            )
        state = self.evaluate_state(
            model=a["model"],
            observations=a["observations"],
            observations_covariance=a["observations_covariance"],
            run_directory_base=a["run_directory_base"],
            parameter_names=a["parameter_names"],
            variational_mean=a["variational_mean"],
            variational_log_std=a["variational_log_std"],
            prior_mean=a["prior_mean"],
            prior_precision_operator=a["prior_precision_operator"],
            prior_covariance_log_det=a["prior_covariance_log_det"],
            sample_size=a["sample_size"],
            evaluation_concurrency=a["evaluation_concurrency"],
            covariance_regularization=a["covariance_regularization"],
            baseline_method=a["baseline_method"],
            gradient_method=a["gradient_method"],
            bounded_parameter_handling=a["bounded_parameter_handling"],
            min_variational_std=a["min_variational_std"],
            max_variational_std=a["max_variational_std"],
            parameter_mins=a["parameter_mins"],
            parameter_maxes=a["parameter_maxes"],
            transform_interior_margin=a["transform_interior_margin"],
            transform_map=a["transform_map"],
            variational_correlation_cholesky=a["variational_correlation_cholesky"],
            elbo_scaling_factor=a["elbo_scaling_factor"],
            log_likelihood_precision_operator=a["log_likelihood_precision_operator"],
            sampling_method="mc",
            dispatcher=a["dispatcher"],
            score_function_entropy_strategy=a["score_function_entropy_strategy"],
        )
        return {
            "optimizer_samples": state["optimizer_samples"],
            "parameter_samples": state["parameter_samples"],
            "iteration_results": {
                "qois": state["qois"],
                "mean-qoi": state["mean_qoi"],
                "errors": state["errors"],
            },
            "mean_relative_mse": state["mean_relative_mse"],
            "state": state,
        }


    def evaluate_state(self, *args, **kwargs):
        bound = inspect.signature(_vi._evaluate_vi_state).bind(*args, **kwargs)
        bound.apply_defaults()
        run_directory_base = str(bound.arguments["run_directory_base"])
        if "hessian_run_" not in run_directory_base:
            return self._evaluate_current_archive(*args, **kwargs)

        if not hasattr(self, "_independent_hessian_archive"):
            self._independent_hessian_archive = _ReuseArchive(self.archive.config)
        self._independent_hessian_archive.optimization_method = self.archive.optimization_method
        primary_archive = self.archive
        self.archive = self._independent_hessian_archive
        try:
            return self._evaluate_current_archive(*args, **kwargs)
        finally:
            self.archive = primary_archive


def _validate_common_reuse_request(call_args, config):
    if _vi._normalize_sampling_method(call_args.get("sampling_method", "mc")) != "mc":
        raise NotImplementedError("Sample reuse currently supports sampling_method='mc' only.")
    baseline = call_args.get("baseline_method")
    baseline = "loo" if baseline is None else _vi._normalize_baseline_method(baseline)
    if baseline not in ("none", "loo"):
        raise NotImplementedError(
            "Sample reuse currently supports baseline_method='none' or 'loo'."
        )
    optimization_method = _normalize_optimization_method(
        call_args.get("optimizer_method", "gradient")
    )
    return optimization_method


def _apply_alpha(base: np.ndarray, delta: np.ndarray, alpha, mode: str) -> np.ndarray:
    mode = _mf._normalize_mfmc_control_variate_mode(mode)
    if mode == "matrix":
        return base + delta @ alpha
    if mode == "scalar":
        return base + float(np.asarray(alpha)) * delta
    return base + np.asarray(alpha).reshape(-1) * delta


def _mf_gradient_from_reuse(optimizer_samples_fom,
                            optimizer_samples_rom_extra,
                            mean,
                            log_std,
                            corr,
                            fom_values,
                            rom_base_values,
                            rom_extra_values,
                            importance_weights,
                            origin_weights,
                            baseline_method,
                            scale,
                            entropy_strategy,
                            use_control_variate,
                            control_variate_mode, mfmc_fixed_gain=1.0):
    score_mean_h, score_log_h = _score_functions(
        optimizer_samples_fom, mean, log_std, corr
    )
    if optimizer_samples_rom_extra.shape[0] > 0:
        score_mean_e, score_log_e = _score_functions(
            optimizer_samples_rom_extra, mean, log_std, corr
        )
    else:
        score_mean_e = score_log_e = None

    def prepare(values, samples):
        result = scale * np.asarray(values)
        if entropy_strategy == "joint":
            result = result - scale * _log_q(samples, mean, log_std, corr)
        return result

    high_values = prepare(fom_values, optimizer_samples_fom)
    low_base_values = prepare(rom_base_values, optimizer_samples_fom)
    low_extra_values = (
        prepare(rom_extra_values, optimizer_samples_rom_extra)
        if score_mean_e is not None else np.zeros(0)
    )
    if baseline_method == "loo":
        high_centered = high_values - _weighted_loo_baseline(high_values, origin_weights)
        low_base_centered = low_base_values - _weighted_loo_baseline(
            low_base_values, origin_weights
        )
        low_extra_centered = (
            low_extra_values - _vi._compute_leave_one_out_baseline(low_extra_values)
            if low_extra_values.size else low_extra_values
        )
    else:
        high_centered, low_base_centered, low_extra_centered = (
            high_values, low_base_values, low_extra_values
        )

    def estimate(score_h, score_e):
        high_terms = importance_weights[:, None] * high_centered[:, None] * score_h
        low_base_terms = (
            importance_weights[:, None] * low_base_centered[:, None] * score_h
        )
        high_mean = np.mean(high_terms, axis=0)
        low_base_mean = np.mean(low_base_terms, axis=0)
        if score_e is None:
            return (
                high_mean,
                _mf._constant_mfmc_gain(0.0, high_mean.shape, control_variate_mode),
                _vi._compute_component_standard_error(high_terms),
            )
        low_extra_terms = low_extra_centered[:, None] * score_e
        low_extra_mean = np.mean(low_extra_terms, axis=0)
        if use_control_variate:
            alpha = _mf._compute_mfmc_alpha(
                high_terms, low_base_terms, mode=control_variate_mode
            )
        else:
            mode = _mf._normalize_mfmc_control_variate_mode(control_variate_mode)
            alpha = _mf._constant_mfmc_gain(mfmc_fixed_gain, high_mean.shape, mode)
        ratio = score_e.shape[0] / float(score_h.shape[0] + score_e.shape[0])
        estimate_value = _apply_alpha(
            high_mean,
            ratio * (low_extra_mean - low_base_mean),
            alpha,
            control_variate_mode,
        )
        return estimate_value, alpha, _vi._compute_component_standard_error(high_terms)

    gradient_mean, alpha_mean, se_mean = estimate(score_mean_h, score_mean_e)
    gradient_log_std, alpha_log, se_log = estimate(score_log_h, score_log_e)
    if entropy_strategy == "analytic":
        gradient_log_std += scale
    standard_error = np.concatenate([se_mean, se_log])
    snr = _vi._compute_gradient_signal_to_noise_ratio(
        np.concatenate([gradient_mean, gradient_log_std]), standard_error
    )
    return gradient_mean, gradient_log_std, alpha_mean, alpha_log, snr, standard_error


class _MFReuseController:
    def __init__(self, config: VISampleReuseConfig):
        self.archive = _ReuseArchive(config)
        self._periodic_refresh_iterations = set()

    def _periodic_due(self, iteration: int) -> bool:
        period = self.archive.config.periodic_refresh
        return (
            period is not None
            and iteration > 0
            and iteration % period == 0
            and iteration not in self._periodic_refresh_iterations
        )

    def _additional_reuse_quality(self, a, state, weights, origin_weights):
        """Run MF-Newton safeguards inside the shared refresh loop."""
        if not _is_newton_reuse(self.archive):
            return True, "reuse", {}
        optimizer_fom = _archive_arrays(self.archive)[0]
        _, clipped_log_std = _vi._compute_variational_std(
            np.asarray(a["variational_log_std"]),
            a["min_variational_std"],
            a["max_variational_std"],
        )
        # The variance trigger intentionally monitors the IS-reused HF curvature
        # contribution. Fresh ROM-only enrichment and the existing MF gain are left
        # unchanged; this isolates the quality of the recycled expensive samples.
        return _hessian_reuse_quality(
            self.archive,
            optimizer_fom,
            float(a["elbo_scaling_factor"])
            * np.asarray(state["raw_log_joint_terms_fom"]),
            np.asarray(a["variational_mean"]),
            clipped_log_std,
            a.get("variational_correlation_cholesky"),
            np.asarray(weights),
            np.asarray(origin_weights),
            a["baseline_method"],
            np.asarray(state["hessian_full"]),
            int(a["fom_sample_size"]),
        )


    def _append_from_state(self, state, mean, log_std, corr, iteration,
                           capacity=None):
        n = state["parameter_samples_fom"].shape[0]
        self.archive.append(_ReuseBatch(
            optimizer_samples=np.asarray(state["optimizer_samples"])[:n].copy(),
            parameter_samples=np.asarray(state["parameter_samples_fom"]).copy(),
            qois=np.asarray(state["qois_fom"]).copy(),
            errors=np.asarray(state["errors_fom"]).copy(),
            variational_mean=np.asarray(mean).copy(),
            variational_log_std=np.asarray(log_std).copy(),
            variational_correlation_cholesky=(
                None if corr is None else np.asarray(corr).copy()
            ),
            iteration=iteration,
        ), capacity=capacity)

    def _evaluate_current_archive(self, *args, **kwargs):
        bound = inspect.signature(_mf._evaluate_mf_vi_state).bind(*args, **kwargs)
        bound.apply_defaults()
        a = dict(bound.arguments)
        iteration = _iteration_from_path(a["iteration_directory"])
        limit = _iteration_archive_limit(self.archive.config, iteration)
        refresh_reasons = []
        refresh_limit_reached = False
        final_diagnostics = {}

        while True:
            periodic_due = self._periodic_due(iteration)
            refresh, reasons, weight_pair, ess = _refresh_decision(
                self.archive,
                np.asarray(a["variational_mean"]),
                np.asarray(a["variational_log_std"]),
                a.get("variational_correlation_cholesky"),
                int(a["fom_sample_size"]),
                periodic_due=periodic_due,
            )
            candidate = None
            if not refresh and a.get("rom_model") is not None:
                weights, origin_weights = weight_pair
                candidate = self._build_reused_state(
                    a, weights, origin_weights, ess
                )
                quality_ok, reason, final_diagnostics = (
                    self._additional_reuse_quality(
                        a, candidate, weights, origin_weights
                    )
                )
                if not quality_ok:
                    refresh = True
                    reasons = (reason,)
            elif a.get("rom_model") is None and not refresh:
                refresh = True
                reasons = ("missing_rom",)

            if not refresh:
                return _annotate_reuse_state(
                    candidate,
                    self.archive,
                    ess,
                    refresh_reasons,
                    refresh_limit_reached,
                    final_diagnostics,
                )

            if len(refresh_reasons) >= limit:
                refresh_limit_reached = True
                weights, origin_weights = _compute_importance_weights(
                    self.archive,
                    np.asarray(a["variational_mean"]),
                    np.asarray(a["variational_log_std"]),
                    a.get("variational_correlation_cholesky"),
                )
                ess = _effective_sample_size(weights)
                candidate = self._build_reused_state(
                    a, weights, origin_weights, ess
                )
                return _annotate_reuse_state(
                    candidate,
                    self.archive,
                    ess,
                    refresh_reasons,
                    refresh_limit_reached,
                    final_diagnostics,
                )

            reason = "+".join(reasons)
            if periodic_due:
                self._periodic_refresh_iterations.add(iteration)
            fresh_args = dict(a)
            if refresh_reasons:
                fresh_args["iteration_directory"] = (
                    f'{a["iteration_directory"]}/abris_refresh_'
                    f'{len(refresh_reasons)}'
                )
            fresh_state = _mf._evaluate_mf_vi_state(**fresh_args)
            self._append_from_state(
                fresh_state,
                a["variational_mean"],
                a["variational_log_std"],
                a.get("variational_correlation_cholesky"),
                iteration,
                capacity=limit,
            )
            refresh_reasons.append(reason)
            for key in (
                "rom_model",
                "training_dirs",
                "training_parameters",
                "training_qois",
                "rom_training_dirs",
                "rom_training_parameters",
                "rom_training_qois",
            ):
                a[key] = fresh_state[key]

    def _build_reused_state(self, a, weights, origin_weights, ess):
        dispatcher = resolve_dispatcher(a["dispatcher"])
        rom_dispatcher = resolve_local_dispatcher(dispatcher)
        optimizer_fom, parameter_fom, qois_fom, errors_fom = _archive_arrays(self.archive)
        mean = np.asarray(a["variational_mean"])
        log_std = np.asarray(a["variational_log_std"])
        corr = a.get("variational_correlation_cholesky")
        rom_model = a["rom_model"]
        parameter_names = a["parameter_names"]
        iteration_directory = a["iteration_directory"]
        bootstrap = _iteration_from_path(iteration_directory) == 0

        rom_base = _vi.run_vi_iteration(
            rom_model,
            a["observations"],
            f"{iteration_directory}/run_rom_reuse_base_",
            parameter_names,
            parameter_fom,
            a["rom_evaluation_concurrency"],
            rom_dispatcher,
        )
        n_extra = int(a["rom_extra_sample_size"])
        if n_extra > 0:
            optimizer_extra, parameter_extra = _vi._draw_parameter_samples(
                mean,
                log_std,
                n_extra,
                a["min_variational_std"],
                a["max_variational_std"],
                a["bounded_parameter_handling"],
                a["parameter_mins"],
                a["parameter_maxes"],
                transform_interior_margin=a["transform_interior_margin"],
                transform_map=a["transform_map"],
                variational_correlation_cholesky=corr,
                sampling_method="mc",
            )
            rom_extra = _vi.run_vi_iteration(
                rom_model,
                a["observations"],
                f"{iteration_directory}/run_rom_reuse_extra_",
                parameter_names,
                parameter_extra,
                a["rom_evaluation_concurrency"],
                rom_dispatcher,
            )
        else:
            optimizer_extra = np.zeros((0, mean.size))
            parameter_extra = np.zeros((0, mean.size))
            rom_extra = {
                "qois": np.zeros((qois_fom.shape[0], 0)),
                "mean-qoi": np.zeros(qois_fom.shape[0]),
                "errors": np.zeros((errors_fom.shape[0], 0)),
            }

        precision = a.get("log_likelihood_precision_operator")
        fom_ll, fom_misfits = _vi._compute_log_likelihoods(
            errors_fom, a["observations_covariance"], a["covariance_regularization"],
            precision_operator=precision,
        )
        base_ll, _ = _vi._compute_log_likelihoods(
            rom_base["errors"], a["observations_covariance"], a["covariance_regularization"],
            precision_operator=precision,
        )
        extra_ll = (
            _vi._compute_log_likelihoods(
                rom_extra["errors"], a["observations_covariance"], a["covariance_regularization"],
                precision_operator=precision,
            )[0]
            if n_extra > 0 else np.zeros(0)
        )

        common_joint_args = (
            a["prior_mean"], a["prior_precision_operator"], a["prior_covariance_log_det"],
            a["bounded_parameter_handling"], a["parameter_mins"], a["parameter_maxes"],
            a["transform_interior_margin"], a["transform_map"],
        )
        fom_lp, fom_jac, fom_joint = _vi._compute_log_prior_and_joint_terms(
            fom_ll, parameter_fom, optimizer_fom, *common_joint_args
        )
        base_lp, base_jac, base_joint = _vi._compute_log_prior_and_joint_terms(
            base_ll, parameter_fom, optimizer_fom, *common_joint_args
        )
        if n_extra > 0:
            extra_lp, extra_jac, extra_joint = _vi._compute_log_prior_and_joint_terms(
                extra_ll, parameter_extra, optimizer_extra, *common_joint_args
            )
        else:
            extra_lp = extra_jac = extra_joint = np.zeros(0)

        baseline = _vi._normalize_baseline_method(a["baseline_method"])
        entropy_strategy = _vi._normalize_score_function_entropy_strategy(
            a["score_function_entropy_strategy"]
        )
        scale = float(a["elbo_scaling_factor"])
        gradient = _mf_gradient_from_reuse(
            optimizer_fom,
            optimizer_extra,
            mean,
            log_std,
            corr,
            fom_joint,
            base_joint,
            extra_joint,
            weights,
            origin_weights,
            baseline,
            scale,
            entropy_strategy,
            a["use_mfmc_control_variate"] and not bootstrap,
            a["mfmc_control_variate_mode"],
            mfmc_fixed_gain=(1.0 if bootstrap and a["use_mfmc_control_variate"] else a["mfmc_fixed_gain"]),
        )
        gradient_mean, gradient_log_std, alpha_mean, alpha_log = gradient[:4]
        std, clipped_log_std = _vi._compute_variational_std(
            log_std, a["min_variational_std"], a["max_variational_std"]
        )
        update_mean, update_log, gradient_method = _vi._compute_update_directions(
            gradient_mean, gradient_log_std, std, a["gradient_method"]
        )

        dimensionality = mean.size
        entropy = np.sum(clipped_log_std) + 0.5 * dimensionality * (1.0 + np.log(2.0 * np.pi))
        if corr is not None:
            entropy += np.sum(np.log(np.diag(corr)))
        entropy *= scale
        hf_mean = float(np.mean(weights * fom_joint))
        if n_extra > 0:
            ratio = n_extra / float(optimizer_fom.shape[0] + n_extra)
            joint_mean = hf_mean + ratio * (
                float(np.mean(extra_joint)) - float(np.mean(weights * base_joint))
            )
        else:
            joint_mean = hf_mean

        normalized_weights = weights / np.sum(weights)
        zeros = np.zeros(mean.size)
        weighted_fom_joint = weights * fom_joint
        weighted_base_joint = weights * base_joint
        optimizer_all = np.vstack([optimizer_fom, optimizer_fom, optimizer_extra])
        parameter_all = np.vstack([parameter_fom, parameter_fom, parameter_extra])
        state = {
            "rom_retraining_strategy": a["rom_retraining_strategy"],
            "gradient_gain_estimation_strategy": a["gradient_gain_estimation_strategy"],
            "mfmc_fixed_gain": a["mfmc_fixed_gain"],
            "hessian_gain_estimation_strategy": a["hessian_gain_estimation_strategy"],
            "bootstrap_rom": bootstrap,
            "rom_adaptation_required": False,
            "pending_rom_model": None,
            "pending_rom_training_dirs": [],
            "pending_rom_training_parameters": None,
            "pending_rom_training_qois": None,
            "gradient_gain_samples": {
                "mean": np.broadcast_to(alpha_mean, (optimizer_fom.shape[0],) + np.shape(alpha_mean)),
                "log_std": np.broadcast_to(alpha_log, (optimizer_fom.shape[0],) + np.shape(alpha_log)),
            },
            "hessian_gain_samples": {},
            "optimizer_samples": optimizer_all,
            "parameter_samples": parameter_all,
            "parameter_samples_fom": parameter_fom,
            "parameter_samples_rom_base": parameter_fom.copy(),
            "parameter_samples_rom_only": parameter_extra,
            "qois_fom": qois_fom,
            "mean_qoi_fom": qois_fom @ normalized_weights,
            "errors_fom": errors_fom,
            "qois_rom_base": rom_base["qois"],
            "qois_rom_coupled": rom_base["qois"],
            "qois_rom_only": rom_extra["qois"],
            "log_likelihoods_fom": weights * fom_ll,
            "raw_log_likelihoods_fom": fom_ll,
            "log_priors_fom": weights * fom_lp,
            "raw_log_priors_fom": fom_lp,
            "log_joint_terms_fom": weighted_fom_joint,
            "raw_log_joint_terms_fom": fom_joint,
            "log_transform_jacobian_fom": fom_jac,
            "log_likelihoods_rom_base": weights * base_ll,
            "log_priors_rom_base": weights * base_lp,
            "log_joint_terms_rom_base": weighted_base_joint,
            "log_transform_jacobian_rom_base": base_jac,
            "log_likelihoods_rom_coupled": weights * base_ll,
            "log_likelihoods_rom_only": extra_ll,
            "log_priors_rom_only": extra_lp,
            "log_joint_terms_rom_only": extra_joint,
            "log_transform_jacobian_rom_only": extra_jac,
            "mean_misfit": float(np.mean(weights * fom_misfits)),
            "mean_relative_mse": float(np.mean(
                weights * _vi._compute_relative_mse(errors_fom, a["observations"])
            )),
            "entropy": entropy,
            "elbo": scale * joint_mean + entropy,
            "gradient_mean": gradient_mean,
            "gradient_log_std": gradient_log_std,
            "hessian_diagonal_mean": zeros.copy(),
            "hessian_diagonal_log_std": zeros.copy(),
            "hessian_full": np.zeros((2 * mean.size, 2 * mean.size)),
            "update_direction_mean": update_mean,
            "update_direction_log_std": update_log,
            "gradient_method": gradient_method,
            "baseline_mean": zeros.copy(),
            "baseline_log_std": zeros.copy(),
            "gradient_signal_to_noise_ratio": gradient[4],
            "gradient_standard_error": gradient[5],
            "mfmc_alpha_mean": alpha_mean,
            "mfmc_alpha_log_std": alpha_log,
            "rom_error": _mf._compute_rom_relative_error(rom_base["qois"], qois_fom),
            "rom_rebuilt_this_iteration": False,
            "rom_model": rom_model,
            "training_dirs": a["training_dirs"],
            "training_parameters": a["training_parameters"],
            "training_qois": a["training_qois"],
            "rom_training_dirs": a["rom_training_dirs"],
            "rom_training_parameters": a["rom_training_parameters"],
            "rom_training_qois": a["rom_training_qois"],
            "sample_reuse_used": True,
            "sample_reuse_ess": float(ess),
            "sample_reuse_archive_samples": self.archive.sample_count,
            "sample_reuse_archive_batches": len(self.archive.batches),
            "sample_reuse_refresh_reason": "reuse",
            "importance_weights": weights,
            "importance_origin_weights": origin_weights,
        }

        optimizer_fom = _archive_arrays(self.archive)[0]
        n_fom = optimizer_fom.shape[0]
        optimizer_extra = np.asarray(state["optimizer_samples"])[2 * n_fom :]

        rom_base_errors = (
            np.asarray(state["qois_rom_base"])
            - np.asarray(a["observations"])[:, None]
        )
        rom_base_log_likelihoods, _ = _vi._compute_log_likelihoods(
            rom_base_errors,
            a["observations_covariance"],
            a["covariance_regularization"],
            precision_operator=a.get("log_likelihood_precision_operator"),
        )
        _, _, raw_rom_base_joint = _vi._compute_log_prior_and_joint_terms(
            rom_base_log_likelihoods,
            np.asarray(state["parameter_samples_rom_base"]),
            optimizer_fom,
            a["prior_mean"],
            a["prior_precision_operator"],
            a["prior_covariance_log_det"],
            a["bounded_parameter_handling"],
            a["parameter_mins"],
            a["parameter_maxes"],
            a["transform_interior_margin"],
            a["transform_map"],
        )
        hessian_full = _mf_hessian_from_reuse(
            optimizer_fom,
            optimizer_extra,
            np.asarray(a["variational_mean"]),
            _vi._compute_variational_std(
                np.asarray(a["variational_log_std"]),
                a["min_variational_std"],
                a["max_variational_std"],
            )[1],
            a.get("variational_correlation_cholesky"),
            np.asarray(state["raw_log_joint_terms_fom"]),
            raw_rom_base_joint,
            np.asarray(state["log_joint_terms_rom_only"]),
            weights,
            origin_weights,
            a["baseline_method"],
            a["elbo_scaling_factor"],
            a["use_mfmc_control_variate"],
            a["mfmc_control_variate_mode"],
            mfmc_fixed_gain=a["mfmc_fixed_gain"],
        )
        dimensionality = np.asarray(a["variational_mean"]).size
        state["hessian_full"] = hessian_full
        state["hessian_diagonal_mean"] = np.diag(
            hessian_full[:dimensionality, :dimensionality]
        )
        state["hessian_diagonal_log_std"] = np.diag(
            hessian_full[dimensionality:, dimensionality:]
        )
        return state


    def evaluate_state(self, *args, **kwargs):
        bound = inspect.signature(_mf._evaluate_mf_vi_state).bind(*args, **kwargs)
        bound.apply_defaults()
        iteration_directory = str(bound.arguments["iteration_directory"]).rstrip("/")
        if not iteration_directory.endswith("/hessian"):
            return self._evaluate_current_archive(*args, **kwargs)

        if not hasattr(self, "_independent_hessian_archive"):
            self._independent_hessian_archive = _ReuseArchive(self.archive.config)
        self._independent_hessian_archive.optimization_method = self.archive.optimization_method
        primary_archive = self.archive
        self.archive = self._independent_hessian_archive
        try:
            return self._evaluate_current_archive(*args, **kwargs)
        finally:
            self.archive = primary_archive


def _compute_hessian_score_blocks(samples: np.ndarray,
                                  mean: np.ndarray,
                                  log_std: np.ndarray,
                                  correlation_cholesky):
    """Return score-Hessian factors for mean, log-std, and cross blocks."""
    std = np.exp(np.asarray(log_std, dtype=float))
    centered = samples - mean[None, :]
    normalized = centered / std[None, :]
    dimensionality = mean.size

    if correlation_cholesky is None:
        score_mean = centered / (std[None, :] ** 2)
        score_log_std = normalized ** 2 - 1.0

        score_mean_outer = score_mean[:, :, None] * score_mean[:, None, :]
        score_log_std_outer = score_log_std[:, :, None] * score_log_std[:, None, :]
        score_cross_outer = score_mean[:, :, None] * score_log_std[:, None, :]

        second_mean = np.zeros_like(score_mean_outer)
        second_log_std = np.zeros_like(score_log_std_outer)
        second_cross = np.zeros_like(score_cross_outer)
        index = np.arange(dimensionality)
        second_mean[:, index, index] = -1.0 / (std ** 2)
        second_log_std[:, index, index] = -2.0 * (normalized ** 2)
        second_cross[:, index, index] = -2.0 * score_mean
    else:
        identity = np.eye(dimensionality)
        correlation_solve = np.linalg.solve(correlation_cholesky, identity)
        correlation_inverse = np.linalg.solve(
            correlation_cholesky.transpose(), correlation_solve
        )
        correlation_solve_samples = np.linalg.solve(
            correlation_cholesky, normalized.transpose()
        )
        inverse_times_normalized = np.linalg.solve(
            correlation_cholesky.transpose(), correlation_solve_samples
        ).transpose()

        score_mean = inverse_times_normalized / std[None, :]
        score_log_std = normalized * inverse_times_normalized - 1.0
        score_mean_outer = score_mean[:, :, None] * score_mean[:, None, :]
        score_log_std_outer = score_log_std[:, :, None] * score_log_std[:, None, :]
        score_cross_outer = score_mean[:, :, None] * score_log_std[:, None, :]

        sample_count = samples.shape[0]
        inverse_std_outer = 1.0 / (std[:, None] * std[None, :])
        second_mean = np.broadcast_to(
            -correlation_inverse * inverse_std_outer,
            (sample_count, dimensionality, dimensionality),
        ).copy()
        outer_normalized = normalized[:, :, None] * normalized[:, None, :]
        second_log_std = -(outer_normalized * correlation_inverse[None, :, :])
        index = np.arange(dimensionality)
        second_log_std[:, index, index] -= normalized * inverse_times_normalized
        second_cross = -(
            (normalized[:, None, :] * correlation_inverse[None, :, :])
            / std[None, :, None]
        )
        second_cross[:, index, index] -= score_mean

    return (
        score_mean_outer + second_mean,
        score_log_std_outer + second_log_std,
        score_cross_outer + second_cross,
    )


def _assemble_hessian_score_matrices(score_blocks) -> np.ndarray:
    mean_scores, log_std_scores, cross_scores = score_blocks
    top = np.concatenate([mean_scores, cross_scores], axis=2)
    bottom = np.concatenate(
        [np.swapaxes(cross_scores, 1, 2), log_std_scores], axis=2
    )
    return np.concatenate([top, bottom], axis=1)


def _center_reused_hessian_values(values: np.ndarray,
                                  origin_weights: np.ndarray,
                                  baseline_method: str) -> np.ndarray:
    baseline_method = _vi._normalize_baseline_method(baseline_method)
    if baseline_method == "loo":
        return values - _weighted_loo_baseline(values, origin_weights)
    if baseline_method == "none":
        return values
    raise NotImplementedError(
        "Sample reuse currently supports baseline_method='none' or 'loo'."
    )


def _hessian_sample_terms(samples: np.ndarray,
                          values: np.ndarray,
                          mean: np.ndarray,
                          log_std: np.ndarray,
                          correlation_cholesky,
                          importance_weights: np.ndarray,
                          origin_weights: np.ndarray,
                          baseline_method: str) -> np.ndarray:
    score_matrices = _assemble_hessian_score_matrices(
        _compute_hessian_score_blocks(
            samples, mean, log_std, correlation_cholesky
        )
    )
    centered = _center_reused_hessian_values(
        np.asarray(values, dtype=float), origin_weights, baseline_method
    )
    return (
        np.asarray(importance_weights)[:, None, None]
        * centered[:, None, None]
        * score_matrices
    )


def _hessian_from_archive(samples: np.ndarray,
                          values: np.ndarray,
                          mean: np.ndarray,
                          log_std: np.ndarray,
                          correlation_cholesky,
                          importance_weights: np.ndarray,
                          origin_weights: np.ndarray,
                          baseline_method: str) -> np.ndarray:
    """Estimate the score-function Hessian from a heterogeneous MIS archive."""
    terms = _hessian_sample_terms(
        samples,
        values,
        mean,
        log_std,
        correlation_cholesky,
        importance_weights,
        origin_weights,
        baseline_method,
    )
    hessian_full = np.mean(terms, axis=0)
    hessian_full = 0.5 * (hessian_full + hessian_full.transpose())
    return np.nan_to_num(hessian_full, nan=0.0, posinf=0.0, neginf=0.0)


def _second_order_score_diagnostics(archive,
                                    mean: np.ndarray,
                                    log_std: np.ndarray,
                                    correlation_cholesky,
                                    importance_weights: np.ndarray,
                                    reference_sample_count: int):
    """Compare recycled and fresh estimates of the zero second-order score moment."""
    samples = _archive_arrays(archive)[0]
    recycled_scores = _assemble_hessian_score_matrices(
        _compute_hessian_score_blocks(
            samples, mean, log_std, correlation_cholesky
        )
    )
    recycled_error_matrix = np.mean(
        importance_weights[:, None, None] * recycled_scores, axis=0
    )

    reference_samples = _draw_optimizer_only(
        mean,
        log_std,
        reference_sample_count,
        correlation_cholesky,
    )
    reference_scores = _assemble_hessian_score_matrices(
        _compute_hessian_score_blocks(
            reference_samples, mean, log_std, correlation_cholesky
        )
    )
    reference_error_matrix = np.mean(reference_scores, axis=0)
    return (
        float(np.linalg.norm(recycled_error_matrix, ord="fro")),
        float(np.linalg.norm(reference_error_matrix, ord="fro")),
    )


def _batch_aware_hessian_standard_error(sample_terms: np.ndarray, archive) -> float:
    """Return the Frobenius aggregate SE for a deterministic-mixture archive.

    Each retained batch is treated as an independent stratum. For batch ``b``
    with fraction ``beta_b`` and sample covariance ``S_b``, the covariance of
    the deterministic-mixture mean contributes ``beta_b**2 S_b / N_b``. Only
    the trace of that covariance is required for the Frobenius aggregate.
    """
    sample_terms = np.asarray(sample_terms, dtype=float)
    total_count = int(sample_terms.shape[0])
    if total_count <= 1:
        return np.inf

    variance_sum = 0.0
    offset = 0
    used_batch_variance = False
    for batch in archive.batches:
        batch_count = int(batch.size)
        sl = slice(offset, offset + batch_count)
        flat_terms = sample_terms[sl].reshape(batch_count, -1)
        if batch_count > 1:
            beta = batch_count / float(total_count)
            component_variances = np.var(flat_terms, axis=0, ddof=1)
            variance_sum += (
                beta ** 2 * float(np.sum(component_variances)) / batch_count
            )
            used_batch_variance = True
        offset += batch_count

    if not used_batch_variance:
        flat_terms = sample_terms.reshape(total_count, -1)
        variance_sum = float(np.sum(np.var(flat_terms, axis=0, ddof=1))) / total_count

    return float(np.sqrt(max(variance_sum, 0.0)))


def _hessian_reuse_quality(archive,
                           samples: np.ndarray,
                           values: np.ndarray,
                           mean: np.ndarray,
                           log_std: np.ndarray,
                           correlation_cholesky,
                           importance_weights: np.ndarray,
                           origin_weights: np.ndarray,
                           baseline_method: str,
                           _hessian_full: np.ndarray,
                           reference_sample_count: int):
    """Evaluate Newton-specific reuse diagnostics without new model calls."""
    config = archive.config
    diagnostics = {
        "sample_reuse_hessian_score_error": np.nan,
        "sample_reuse_hessian_score_reference_error": np.nan,
        "sample_reuse_hessian_standard_error": np.nan,
        "sample_reuse_hessian_relative_standard_error": np.nan,
    }

    if getattr(config, "use_hessian_score_diagnostic", False):
        recycled_error, reference_error = _second_order_score_diagnostics(
            archive,
            mean,
            log_std,
            correlation_cholesky,
            importance_weights,
            reference_sample_count,
        )
        diagnostics["sample_reuse_hessian_score_error"] = recycled_error
        diagnostics["sample_reuse_hessian_score_reference_error"] = reference_error
        reference_scale = max(reference_error, np.sqrt(np.finfo(float).eps))
        error_scale = float(getattr(config, "hessian_score_error_scale", 2.0))
        if recycled_error > error_scale * reference_scale:
            return False, "hessian_score", diagnostics

    threshold = getattr(
        config, "hessian_relative_standard_error_threshold", None
    )
    if threshold is not None:
        terms = _hessian_sample_terms(
            samples,
            values,
            mean,
            log_std,
            correlation_cholesky,
            importance_weights,
            origin_weights,
            baseline_method,
        )
        standard_error = _batch_aware_hessian_standard_error(terms, archive)
        # Normalize by the same recycled HF curvature contribution whose
        # uncertainty is measured above. In MFVI, using the full multifidelity
        # Hessian here can spuriously inflate the relative error when the ROM
        # control-variate correction cancels part of the HF curvature.
        recycled_hessian = np.mean(terms, axis=0)
        recycled_hessian = 0.5 * (
            recycled_hessian + recycled_hessian.transpose()
        )
        hessian_norm = float(np.linalg.norm(recycled_hessian, ord="fro"))
        denominator = max(hessian_norm, np.sqrt(np.finfo(float).eps))
        relative_standard_error = standard_error / denominator
        diagnostics["sample_reuse_hessian_standard_error"] = standard_error
        diagnostics[
            "sample_reuse_hessian_relative_standard_error"
        ] = relative_standard_error
        if not np.isfinite(relative_standard_error) or relative_standard_error > threshold:
            return False, "hessian_variance", diagnostics

    return True, "reuse", diagnostics


def _mf_hessian_from_reuse(optimizer_samples_fom: np.ndarray,
                           optimizer_samples_rom_extra: np.ndarray,
                           mean: np.ndarray,
                           log_std: np.ndarray,
                           correlation_cholesky,
                           fom_values: np.ndarray,
                           rom_base_values: np.ndarray,
                           rom_extra_values: np.ndarray,
                           importance_weights: np.ndarray,
                           origin_weights: np.ndarray,
                           baseline_method: str,
                           elbo_scaling_factor: float,
                           use_control_variate: bool,
                           control_variate_mode: str,
                           mfmc_fixed_gain: float = 1.0) -> np.ndarray:
    """MFVI Hessian with MIS-reused coupled HF/LF samples and fresh LF extras."""
    high_scores = _compute_hessian_score_blocks(
        optimizer_samples_fom, mean, log_std, correlation_cholesky
    )
    low_base_scores = high_scores
    if optimizer_samples_rom_extra.shape[0] > 0:
        low_extra_scores = _compute_hessian_score_blocks(
            optimizer_samples_rom_extra, mean, log_std, correlation_cholesky
        )
    else:
        low_extra_scores = (None, None, None)

    scale = float(elbo_scaling_factor)
    high_values = scale * np.asarray(fom_values, dtype=float)
    low_base_values = scale * np.asarray(rom_base_values, dtype=float)
    low_extra_values = scale * np.asarray(rom_extra_values, dtype=float)
    baseline_method = _vi._normalize_baseline_method(baseline_method)
    if baseline_method == "loo":
        high_centered = high_values - _weighted_loo_baseline(
            high_values, origin_weights
        )
        low_base_centered = low_base_values - _weighted_loo_baseline(
            low_base_values, origin_weights
        )
        low_extra_centered = (
            low_extra_values - _vi._compute_leave_one_out_baseline(low_extra_values)
            if low_extra_values.size else low_extra_values
        )
    elif baseline_method == "none":
        high_centered = high_values
        low_base_centered = low_base_values
        low_extra_centered = low_extra_values
    else:
        raise NotImplementedError(
            "Sample reuse currently supports baseline_method='none' or 'loo'."
        )

    def estimate_block(high_score, low_base_score, low_extra_score):
        high_terms = (
            importance_weights[:, None, None]
            * high_centered[:, None, None]
            * high_score
        )
        low_base_terms = (
            importance_weights[:, None, None]
            * low_base_centered[:, None, None]
            * low_base_score
        )
        low_extra_terms = None
        if low_extra_score is not None:
            low_extra_terms = low_extra_centered[:, None, None] * low_extra_score
        estimate, _ = _mf._mfmc_gradient_estimator(
            high_terms,
            low_base_terms,
            low_extra_terms,
            use_control_variate,
            control_variate_mode,
            mfmc_fixed_gain=mfmc_fixed_gain,
        )
        return estimate

    hessian_mean = estimate_block(high_scores[0], low_base_scores[0], low_extra_scores[0])
    hessian_log_std = estimate_block(
        high_scores[1], low_base_scores[1], low_extra_scores[1]
    )
    hessian_cross = estimate_block(
        high_scores[2], low_base_scores[2], low_extra_scores[2]
    )
    hessian_full = np.block([
        [hessian_mean, hessian_cross],
        [hessian_cross.transpose(), hessian_log_std],
    ])
    hessian_full = 0.5 * (hessian_full + hessian_full.transpose())
    return np.nan_to_num(hessian_full, nan=0.0, posinf=0.0, neginf=0.0)


def _is_newton_reuse(archive) -> bool:
    return archive.optimization_method == "newton"


def make_controller(config, call_args, *, multifidelity=False):
    """Create a private evaluator for this run, leaving module globals untouched."""
    if config is None or not config.enabled:
        return None
    if not isinstance(config, VISampleReuseConfig):
        raise TypeError("sample_reuse_config must be VISampleReuseConfig or None")
    method = _validate_common_reuse_request(call_args, config)
    if multifidelity and (call_args["rom_retraining_strategy"] != "same_sample"
            or call_args["gradient_gain_estimation_strategy"] != "same_sample"):
        raise NotImplementedError(
            "MFVI sample reuse requires rom_retraining_strategy='same_sample' "
            "and gradient_gain_estimation_strategy='same_sample'."
        )
    controller = (_MFReuseController if multifidelity else _VIReuseController)(config)
    controller.archive.optimization_method = method
    return controller
