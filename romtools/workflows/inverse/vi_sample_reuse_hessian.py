"""Newton/Hessian support for importance-sampling VI sample reuse.

This module extends :mod:`vi_sample_reuse` without changing the first-pass
archive implementation. Reused FOM evaluations are combined with current
score-function Hessian factors using the same deterministic-mixture importance
weights as the gradient. The existing VI/MFVI Newton solvers, curvature
strategies, absolute-Hessian treatment, and multifidelity gain machinery remain
responsible for constructing the optimization step.

The extension is installed for side effects by ``inverse.__init__``. Keeping
it separate makes the Hessian-specific assumptions explicit while the sample
reuse API is still experimental.
"""

import inspect

import numpy as np

from romtools.workflows.inverse import mf_vi_drivers as _mf
from romtools.workflows.inverse import vi_drivers as _vi
from romtools.workflows.inverse import vi_sample_reuse as _reuse


_ORIGINAL_BUILD_REUSED_VI_STATE = _reuse._build_reused_vi_state
_ORIGINAL_MF_BUILD_REUSED_STATE = _reuse._MFReuseController._build_reused_state
_ORIGINAL_VI_CONTROLLER_EVALUATE_STATE = _reuse._VIReuseController.evaluate_state
_ORIGINAL_MF_CONTROLLER_EVALUATE_STATE = _reuse._MFReuseController.evaluate_state


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


def _hessian_from_archive(samples: np.ndarray,
                          values: np.ndarray,
                          mean: np.ndarray,
                          log_std: np.ndarray,
                          correlation_cholesky,
                          importance_weights: np.ndarray,
                          origin_weights: np.ndarray,
                          baseline_method: str) -> np.ndarray:
    """Estimate the score-function Hessian from a heterogeneous MIS archive."""
    mean_scores, log_std_scores, cross_scores = _compute_hessian_score_blocks(
        samples, mean, log_std, correlation_cholesky
    )
    centered = _center_reused_hessian_values(
        np.asarray(values, dtype=float), origin_weights, baseline_method
    )
    weighted_values = importance_weights * centered
    hessian_mean = np.mean(weighted_values[:, None, None] * mean_scores, axis=0)
    hessian_log_std = np.mean(
        weighted_values[:, None, None] * log_std_scores, axis=0
    )
    hessian_cross = np.mean(
        weighted_values[:, None, None] * cross_scores, axis=0
    )
    hessian_full = np.block([
        [hessian_mean, hessian_cross],
        [hessian_cross.transpose(), hessian_log_std],
    ])
    hessian_full = 0.5 * (hessian_full + hessian_full.transpose())
    return np.nan_to_num(hessian_full, nan=0.0, posinf=0.0, neginf=0.0)


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


def _validate_common_reuse_request_with_newton(call_args, config):
    _ = config
    if _vi._normalize_sampling_method(call_args.get("sampling_method", "mc")) != "mc":
        raise NotImplementedError("Sample reuse currently supports sampling_method='mc' only.")
    baseline = call_args.get("baseline_method")
    baseline = "loo" if baseline is None else _vi._normalize_baseline_method(baseline)
    if baseline not in ("none", "loo"):
        raise NotImplementedError(
            "Sample reuse currently supports baseline_method='none' or 'loo'."
        )


def _evaluate_vi_state_with_independent_archive(self, *args, **kwargs):
    bound = inspect.signature(_reuse._ORIGINAL_EVALUATE_VI_STATE).bind(*args, **kwargs)
    bound.apply_defaults()
    run_directory_base = str(bound.arguments["run_directory_base"])
    if "hessian_run_" not in run_directory_base:
        return _ORIGINAL_VI_CONTROLLER_EVALUATE_STATE(self, *args, **kwargs)

    if not hasattr(self, "_independent_hessian_archive"):
        self._independent_hessian_archive = _reuse._ReuseArchive(self.archive.config)
    primary_archive = self.archive
    self.archive = self._independent_hessian_archive
    try:
        return _ORIGINAL_VI_CONTROLLER_EVALUATE_STATE(self, *args, **kwargs)
    finally:
        self.archive = primary_archive


def _evaluate_mf_state_with_independent_archive(self, *args, **kwargs):
    bound = inspect.signature(_reuse._ORIGINAL_EVALUATE_MF_VI_STATE).bind(*args, **kwargs)
    bound.apply_defaults()
    iteration_directory = str(bound.arguments["iteration_directory"]).rstrip("/")
    if not iteration_directory.endswith("/hessian"):
        return _ORIGINAL_MF_CONTROLLER_EVALUATE_STATE(self, *args, **kwargs)

    if not hasattr(self, "_independent_hessian_archive"):
        self._independent_hessian_archive = _reuse._ReuseArchive(self.archive.config)
    primary_archive = self.archive
    self.archive = self._independent_hessian_archive
    try:
        return _ORIGINAL_MF_CONTROLLER_EVALUATE_STATE(self, *args, **kwargs)
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
