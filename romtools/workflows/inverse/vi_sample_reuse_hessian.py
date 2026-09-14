"""Newton/Hessian support for importance-sampling VI sample reuse.

This module extends :mod:`vi_sample_reuse` with score-function Hessian reuse
and Newton-specific archive quality checks. Reused FOM evaluations are combined
with current score-function Hessian factors using the same deterministic-mixture
importance weights as the gradient. The existing VI/MFVI Newton solvers,
curvature strategies, absolute-Hessian treatment, and multifidelity gain
machinery remain responsible for constructing the optimization step.

Two additional safeguards are applied only to Newton runs: a second-order score
identity diagnostic and a batch-aware relative standard-error estimate for the
importance-sampled Hessian contribution.
"""

from dataclasses import dataclass
import inspect
import sys
from typing import Optional

import numpy as np

from romtools.workflows.inverse import mf_vi_drivers as _mf
from romtools.workflows.inverse import vi_drivers as _vi
from romtools.workflows.inverse import vi_sample_reuse as _reuse


_ORIGINAL_BUILD_REUSED_VI_STATE = _reuse._build_reused_vi_state
_ORIGINAL_MF_BUILD_REUSED_STATE = _reuse._MFReuseController._build_reused_state
_ORIGINAL_VI_CONTROLLER_EVALUATE_STATE = _reuse._VIReuseController.evaluate_state
_ORIGINAL_MF_CONTROLLER_EVALUATE_STATE = _reuse._MFReuseController.evaluate_state
_BASE_SAMPLE_REUSE_CONFIG = _reuse.VISampleReuseConfig
_OPTIMIZATION_METHOD_BY_CONFIG_ID = {}


@dataclass(frozen=True)
class VISampleReuseConfig(_BASE_SAMPLE_REUSE_CONFIG):
    """Sample-reuse configuration including Newton/Hessian safeguards.

    The Hessian checks are only evaluated for Newton runs. First-order VI and
    MFVI therefore retain the original sample-reuse behavior.
    """

    use_hessian_score_diagnostic: bool = True
    hessian_score_error_scale: float = 2.0
    hessian_relative_standard_error_threshold: Optional[float] = 0.25

    def __post_init__(self):
        super().__post_init__()
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


# Replace the first-pass config class before inverse.__init__ finishes wiring
# the public API. The base wrapper resolves this module global at call time.
_reuse.VISampleReuseConfig = VISampleReuseConfig
_parent_inverse_module = sys.modules.get("romtools.workflows.inverse")
if _parent_inverse_module is not None:
    setattr(_parent_inverse_module, "VISampleReuseConfig", VISampleReuseConfig)


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
        return values - _reuse._weighted_loo_baseline(values, origin_weights)
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
    samples = _reuse._archive_arrays(archive)[0]
    recycled_scores = _assemble_hessian_score_matrices(
        _compute_hessian_score_blocks(
            samples, mean, log_std, correlation_cholesky
        )
    )
    recycled_error_matrix = np.mean(
        importance_weights[:, None, None] * recycled_scores, axis=0
    )

    reference_samples = _reuse._draw_optimizer_only(
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
                           hessian_full: np.ndarray,
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


def _attach_hessian_diagnostics(state, diagnostics):
    for key, value in diagnostics.items():
        state[key] = value
    return state


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
                           control_variate_mode: str) -> np.ndarray:
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
        high_centered = high_values - _reuse._weighted_loo_baseline(
            high_values, origin_weights
        )
        low_base_centered = low_base_values - _reuse._weighted_loo_baseline(
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


def _build_reused_vi_state_with_hessian(a, archive, weights, origin_weights, ess):
    state = _ORIGINAL_BUILD_REUSED_VI_STATE(
        a, archive, weights, origin_weights, ess
    )
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


def _build_reused_mf_state_with_hessian(self, a, weights, origin_weights, ess):
    state = _ORIGINAL_MF_BUILD_REUSED_STATE(self, a, weights, origin_weights, ess)
    optimizer_fom = _reuse._archive_arrays(self.archive)[0]
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


def _is_newton_reuse(archive) -> bool:
    return _OPTIMIZATION_METHOD_BY_CONFIG_ID.get(id(archive.config)) == "newton"


def _validate_common_reuse_request_with_newton(call_args, config):
    if _vi._normalize_sampling_method(call_args.get("sampling_method", "mc")) != "mc":
        raise NotImplementedError("Sample reuse currently supports sampling_method='mc' only.")
    baseline = call_args.get("baseline_method")
    baseline = "loo" if baseline is None else _vi._normalize_baseline_method(baseline)
    if baseline not in ("none", "loo"):
        raise NotImplementedError(
            "Sample reuse currently supports baseline_method='none' or 'loo'."
        )
    optimization_method = _reuse._normalize_optimization_method(
        call_args.get("optimizer_method", "gradient")
    )
    _OPTIMIZATION_METHOD_BY_CONFIG_ID[id(config)] = optimization_method


def _evaluate_vi_current_archive(self, *args, **kwargs):
    bound = inspect.signature(_reuse._ORIGINAL_EVALUATE_VI_STATE).bind(*args, **kwargs)
    bound.apply_defaults()
    a = bound.arguments
    state = _ORIGINAL_VI_CONTROLLER_EVALUATE_STATE(self, *args, **kwargs)
    if not state.get("sample_reuse_used", False) or not _is_newton_reuse(self.archive):
        return state

    weights = np.asarray(state["importance_weights"])
    origin_weights = np.asarray(state["importance_origin_weights"])
    _, clipped_log_std = _vi._compute_variational_std(
        np.asarray(a["variational_log_std"]),
        a["min_variational_std"],
        a["max_variational_std"],
    )
    quality_ok, reason, diagnostics = _hessian_reuse_quality(
        self.archive,
        np.asarray(state["optimizer_samples"]),
        float(a["elbo_scaling_factor"]) * np.asarray(state["raw_log_joint_terms"]),
        np.asarray(a["variational_mean"]),
        clipped_log_std,
        a.get("variational_correlation_cholesky"),
        weights,
        origin_weights,
        a["baseline_method"],
        np.asarray(state["hessian_full"]),
        int(a["sample_size"]),
    )
    if quality_ok:
        return _attach_hessian_diagnostics(state, diagnostics)

    fresh_state = _reuse._ORIGINAL_EVALUATE_VI_STATE(*args, **kwargs)
    iteration = _reuse._iteration_from_path(a["run_directory_base"])
    self.archive.append(
        _reuse._batch_from_vi_state(
            fresh_state,
            a["variational_mean"],
            a["variational_log_std"],
            a.get("variational_correlation_cholesky"),
            iteration,
        )
    )
    fresh_state["sample_reuse_used"] = False
    fresh_state["sample_reuse_refresh_reason"] = reason
    fresh_state["sample_reuse_archive_samples"] = self.archive.sample_count
    fresh_state["sample_reuse_archive_batches"] = len(self.archive.batches)
    return _attach_hessian_diagnostics(fresh_state, diagnostics)


def _evaluate_vi_state_with_independent_archive(self, *args, **kwargs):
    bound = inspect.signature(_reuse._ORIGINAL_EVALUATE_VI_STATE).bind(*args, **kwargs)
    bound.apply_defaults()
    run_directory_base = str(bound.arguments["run_directory_base"])
    if "hessian_run_" not in run_directory_base:
        return _evaluate_vi_current_archive(self, *args, **kwargs)

    if not hasattr(self, "_independent_hessian_archive"):
        self._independent_hessian_archive = _reuse._ReuseArchive(self.archive.config)
    primary_archive = self.archive
    self.archive = self._independent_hessian_archive
    try:
        return _evaluate_vi_current_archive(self, *args, **kwargs)
    finally:
        self.archive = primary_archive


def _evaluate_mf_current_archive(self, *args, **kwargs):
    bound = inspect.signature(_reuse._ORIGINAL_EVALUATE_MF_VI_STATE).bind(*args, **kwargs)
    bound.apply_defaults()
    a = bound.arguments
    state = _ORIGINAL_MF_CONTROLLER_EVALUATE_STATE(self, *args, **kwargs)
    if not state.get("sample_reuse_used", False) or not _is_newton_reuse(self.archive):
        return state

    optimizer_fom = _reuse._archive_arrays(self.archive)[0]
    weights = np.asarray(state["importance_weights"])
    origin_weights = np.asarray(state["importance_origin_weights"])
    _, clipped_log_std = _vi._compute_variational_std(
        np.asarray(a["variational_log_std"]),
        a["min_variational_std"],
        a["max_variational_std"],
    )
    # The variance trigger intentionally monitors the IS-reused HF curvature
    # contribution. Fresh ROM-only enrichment and the existing MF gain are left
    # unchanged; this isolates the quality of the recycled expensive samples.
    quality_ok, reason, diagnostics = _hessian_reuse_quality(
        self.archive,
        optimizer_fom,
        float(a["elbo_scaling_factor"])
        * np.asarray(state["raw_log_joint_terms_fom"]),
        np.asarray(a["variational_mean"]),
        clipped_log_std,
        a.get("variational_correlation_cholesky"),
        weights,
        origin_weights,
        a["baseline_method"],
        np.asarray(state["hessian_full"]),
        int(a["fom_sample_size"]),
    )
    if quality_ok:
        return _attach_hessian_diagnostics(state, diagnostics)

    fresh_state = _reuse._ORIGINAL_EVALUATE_MF_VI_STATE(*args, **kwargs)
    iteration = _reuse._iteration_from_path(a["iteration_directory"])
    self._append_from_state(
        fresh_state,
        a["variational_mean"],
        a["variational_log_std"],
        a.get("variational_correlation_cholesky"),
        iteration,
    )
    fresh_state["sample_reuse_used"] = False
    fresh_state["sample_reuse_refresh_reason"] = reason
    fresh_state["sample_reuse_archive_samples"] = self.archive.sample_count
    fresh_state["sample_reuse_archive_batches"] = len(self.archive.batches)
    return _attach_hessian_diagnostics(fresh_state, diagnostics)


def _evaluate_mf_state_with_independent_archive(self, *args, **kwargs):
    bound = inspect.signature(_reuse._ORIGINAL_EVALUATE_MF_VI_STATE).bind(*args, **kwargs)
    bound.apply_defaults()
    iteration_directory = str(bound.arguments["iteration_directory"]).rstrip("/")
    if not iteration_directory.endswith("/hessian"):
        return _evaluate_mf_current_archive(self, *args, **kwargs)

    if not hasattr(self, "_independent_hessian_archive"):
        self._independent_hessian_archive = _reuse._ReuseArchive(self.archive.config)
    primary_archive = self.archive
    self.archive = self._independent_hessian_archive
    try:
        return _evaluate_mf_current_archive(self, *args, **kwargs)
    finally:
        self.archive = primary_archive


def _install_newton_hessian_reuse_extension():
    if getattr(_reuse, "_newton_hessian_reuse_extension_installed", False):
        return
    _reuse._build_reused_vi_state = _build_reused_vi_state_with_hessian
    _reuse._MFReuseController._build_reused_state = _build_reused_mf_state_with_hessian
    _reuse._VIReuseController.evaluate_state = _evaluate_vi_state_with_independent_archive
    _reuse._MFReuseController.evaluate_state = _evaluate_mf_state_with_independent_archive
    _reuse._validate_common_reuse_request = _validate_common_reuse_request_with_newton
    _reuse._newton_hessian_reuse_extension_installed = True


_install_newton_hessian_reuse_extension()
