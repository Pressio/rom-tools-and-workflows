"""Cross-fitted ACV gains from cached score/weight moments."""

from .common import (
    _compute_variational_log_densities, _normalize_mfmc_control_variate_mode,
    _normalize_baseline_method,
)

import numpy as np
from scipy.special import logsumexp


def normalize_strategy(value, name, choices):
    value = value.strip().lower()
    if value not in choices:
        raise ValueError(f"Unsupported {name} '{value}'. Supported options are {choices}.")
    return value


def validate_gain_sampling(gain_strategy, sampling_method):
    if gain_strategy in ('loo', 'k_fold', 'history', 'independent') and sampling_method != 'mc':
        raise ValueError(f"{gain_strategy.upper()} gain estimation requires sampling_method='mc' with independent paired samples.")


def regularized_matrix_gain(variance, cross, epsilon=1e-12, max_rank=None, noise_floor=0.0):
    """Solve the ridge fit on the covariance range, excluding numerical nulls."""
    variance = 0.5*(variance + variance.T)
    values, vectors = np.linalg.eigh(variance)
    if values[-1] <= epsilon:
        return np.zeros_like(cross)
    cutoff = max(noise_floor, np.finfo(float).eps*variance.shape[0]*max(float(values[-1]), 0.0))
    keep = values > cutoff
    if max_rank is not None and max_rank < variance.shape[0]:
        keep[:variance.shape[0]-max_rank] = False
    if not np.any(keep):
        return np.zeros_like(cross)
    basis = vectors[:, keep]
    return basis @ ((basis.T @ cross)/(values[keep, None] + epsilon))


class _FoldMoments:
    """Stable prefix/suffix scatter merges, without subtracting the held-out row."""

    def __init__(self, dimension, matrix):
        self.count = 0
        self.means = {key: np.zeros(dimension) for key in ('h', 'l', 'd', 's')}
        self.pairs = ('ll', 'ld', 'ls', 'ds', 'ss', 'hh', 'hs')
        self.matrix_pairs = set(self.pairs[:5]) if matrix else set()
        self.scatter = {pair: np.zeros((dimension, dimension) if pair in self.matrix_pairs
                                      else dimension) for pair in self.pairs}

    def copy(self):
        result = _FoldMoments(self.means['s'].size, bool(self.matrix_pairs))
        result.count = self.count
        result.means = {key: value.copy() for key, value in self.means.items()}
        result.scatter = {key: value.copy() for key, value in self.scatter.items()}
        return result

    def add(self, values):
        count = self.count + 1
        delta = {key: values[key] - self.means[key] for key in self.means}
        for pair in self.pairs:
            left, right = (delta[key] for key in pair)
            product = np.outer(left, right) if pair in self.matrix_pairs else left*right
            self.scatter[pair] += (self.count/count)*product
        for key in self.means:
            self.means[key] += delta[key]/count
        self.count = count

    def covariance_with(self, other):
        count = self.count + other.count
        factor = self.count*other.count/count
        delta = {key: other.means[key] - self.means[key] for key in self.means}
        result = {}
        for pair in self.pairs:
            left, right = (delta[key] for key in pair)
            product = np.outer(left, right) if pair in self.matrix_pairs else left*right
            result[pair] = (self.scatter[pair] + other.scatter[pair] + factor*product)/count
        return result


def _exclusive_sums(values):
    """Sum rows strictly before/after each row, avoiding total-minus-row cancellation."""
    zero = np.zeros_like(values[:1])
    before = np.concatenate((zero, np.cumsum(values[:-1], axis=0)), axis=0)
    after = np.concatenate((np.cumsum(values[:0:-1], axis=0)[::-1], zero), axis=0)
    return before + after


def validate_gain_num_folds(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 2:
        raise ValueError('gain_num_folds must be an integer greater than or equal to two.')
    return int(value)


def gain_folds(n, num_folds=None):
    """Balanced, value-independent folds; None selects singleton LOO folds."""
    count = n if num_folds is None else min(validate_gain_num_folds(num_folds), n)
    return np.array_split(np.arange(n), count)


def k_fold_acv_estimator(*args, gain_num_folds=5, **kwargs):
    """Fit gains on complementary paired folds, with inner LOO baselines."""
    return loo_acv_estimator(*args, num_folds=validate_gain_num_folds(gain_num_folds), **kwargs)


def loo_acv_estimator(high_weights, low_weights, extra_weights, scores, extra_scores,
                      baseline_method, mode, *, force_unit_gain=False,
                      high_reference_terms=None, epsilon=1e-12,
                      gain_regularization=1.0, return_baselines=False, fixed_gain=None, num_folds=None):
    """Cross-fit stable ACV gains with a fold-only extreme-coefficient guard.

    ``num_folds=None`` uses singleton LOO folds; otherwise use balanced
    contiguous k-fold partitions. Prefix/suffix covariance merges exclude
    the held-out fold without unstable
    scatter downdates. Weight offsets are removed before multiplication by
    scores. The fold containing the reference row is computed separately with
    a reference from its own fitting set. Moment work remains O(n*p**2);
    matrix solves are performed once per fold.

    Ordinary gains are not shrunk toward a preset coefficient. To guard
    against tiny fitting variances, positive ``gain_regularization`` limits
    scalar/componentwise magnitudes and matrix spectral norms to
    ``1 + sqrt(training_count/gain_regularization)/2``. The limit grows with sample
    count; zero disables it. All coefficient decisions exclude the held-out
    fold and LF-only samples. Matrix coefficients use delta_L @ A.
    """
    mode = normalize_strategy(mode, 'mfmc_control_variate_mode', ('scalar', 'componentwise', 'matrix'))
    baseline_method = normalize_strategy(baseline_method, 'baseline_method', ('none', 'loo', 'optimal'))
    if not np.isfinite(gain_regularization) or gain_regularization < 0:
        raise ValueError('gain_regularization must be finite and non-negative.')
    high_weights = np.asarray(high_weights, dtype=float)
    low_weights = np.asarray(low_weights, dtype=float)
    extra_weights = np.asarray(extra_weights, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n, m = high_weights.size, extra_weights.size
    if n < 1 or high_weights.ndim != 1 or low_weights.shape != high_weights.shape:
        raise ValueError('Cross-fitting requires a nonempty paired HF/LF sample set.')
    shape = scores.shape[1:]
    p = int(np.prod(shape))
    s = scores.reshape(n, p)
    matrix = mode == 'matrix'
    folds = gain_folds(n, num_folds)
    ratio = m/(n+m)
    gain_shape = () if mode == 'scalar' else ((p, p) if matrix else shape)
    gains = np.zeros((n,) + gain_shape)
    corrected = np.empty((n, p))
    if return_baselines:
        outer_high_baselines = np.zeros((n, p))
        outer_low_baselines = np.zeros((n, p))

    def prepare(reference):
        # These references belong to every fitting set that uses this cache.
        rh = high_weights[reference] if baseline_method != 'none' and n > 1 else 0.0
        rl = low_weights[reference] if baseline_method != 'none' and n > 1 else 0.0
        wh, wl = high_weights-rh, low_weights-rl
        wd = wh-wl if baseline_method != 'none' else high_weights-low_weights
        basis = {'h': wh[:, None]*s, 'l': wl[:, None]*s,
                 'd': wd[:, None]*s, 's': s}
        return rh, rl, wh, wl, basis

    rh, rl, wh, wl, basis = prepare(0)
    needs_fit = bool(m and fixed_gain is None and not force_unit_gain and any(n-len(fold) >= 2 for fold in folds))
    if needs_fit:
        prefixes = []
        accumulator = _FoldMoments(p, matrix)
        for indices in folds:
            prefixes.append(accumulator.copy())
            for j in indices:
                accumulator.add({key: value[j] for key, value in basis.items()})
        suffix = _FoldMoments(p, matrix)

    def exclude_folds(values):
        return _exclusive_sums(np.asarray([values[indices].sum(axis=0) for indices in folds]))

    def baseline_sums(wh, wl):
        if baseline_method == 'loo':
            return exclude_folds(wh), exclude_folds(wl)
        if baseline_method == 'optimal':
            return (exclude_folds(wh[:, None]*s**2),
                    exclude_folds(wl[:, None]*s**2), exclude_folds(s**2))
        return None

    sums = baseline_sums(wh, wl)
    ref_squares = (None if high_reference_terms is None else
                   np.asarray(high_reference_terms).reshape(n, p)**2)
    reference_sums = None if ref_squares is None else exclude_folds(ref_squares)
    if m:
        extra_s = np.asarray(extra_scores).reshape(m, p)
        # Extra-sample reference affects the correction only, never the gain.
        extra_reference = extra_weights[0]
        extra_centered_mean = ((extra_weights-extra_reference)[:, None]*extra_s).mean(axis=0)
        extra_score_mean = extra_s.mean(axis=0)

    for i in range(len(folds)-1, -1, -1):
        indices = folds[i]
        k = n-len(indices)
        gain_limit = (1 + np.sqrt(k/gain_regularization)/2
                      if gain_regularization > 0 else np.inf)
        fold_rh, fold_rl, fold_wh, fold_wl = rh, rl, wh, wl
        fold_sums = sums
        covariance = prefixes[i].covariance_with(suffix) if needs_fit else None
        if i == 0 and k and baseline_method != 'none':
            # Reference row 0 is held out: neither it nor its large offset may
            # influence this fold's arithmetic or coefficient decisions.
            fold_rh, fold_rl, fold_wh, fold_wl, fold_basis = prepare(int(folds[0][-1])+1)
            if needs_fit:
                fold = _FoldMoments(p, matrix)
                for j in range(int(folds[0][-1])+1, n):
                    fold.add({key: value[j] for key, value in fold_basis.items()})
                covariance = {key: value/k for key, value in fold.scatter.items()}
            fold_sums = baseline_sums(fold_wh, fold_wl)

        held_h = held_l = np.zeros(p)
        a = 1.0
        if k and baseline_method == 'loo':
            held_h = np.full(p, fold_sums[0][i]/k)
            held_l = np.full(p, fold_sums[1][i]/k)
            if k > 1:
                # Within the fitting fold, leave-j-out terms satisfy
                # (w_j - mean_{T minus j})*s_j = k/(k-1)*(w_j-mean_T)*s_j.
                # Exclude the entire outer fold, then the inner row.
                a = k/(k-1)
        elif k and baseline_method == 'optimal':
            numerator_h, numerator_l, denominator = (value[i] for value in fold_sums)
            held_h = np.divide(numerator_h, denominator, out=np.full(p, -fold_rh),
                               where=denominator > epsilon)
            held_l = np.divide(numerator_l, denominator, out=np.full(p, -fold_rl),
                               where=denominator > epsilon)
        fit_h, fit_l = a*held_h, a*held_l
        fit_d = fit_h-fit_l
        if return_baselines:
            outer_high_baselines[indices] = fold_rh+held_h
            outer_low_baselines[indices] = fold_rl+held_l

        if m and force_unit_gain:
            gain = np.eye(p) if matrix else np.ones(gain_shape)
        elif m and fixed_gain is not None:
            gain = np.asarray(fixed_gain).reshape(gain_shape)
        elif m and k >= 2:
            ll, ld, ls, ds, ss = (covariance[key] for key in ('ll', 'ld', 'ls', 'ds', 'ss'))
            ls_diag = np.diag(ls) if matrix else ls
            ds_diag = np.diag(ds) if matrix else ds
            ss_diag = np.diag(ss) if matrix else ss
            high_variance = np.maximum(a*a*covariance['hh'] - 2*a*fit_h*covariance['hs']
                                      + fit_h**2*ss_diag, 0.0)
            if matrix:
                variance = (a*a*ll - a*(ls*fit_l[None, :] + ls.T*fit_l[:, None])
                            + ss*fit_l[:, None]*fit_l[None, :])
                cross = (a*a*ld - a*(ls*fit_d[None, :] + ds.T*fit_l[:, None])
                         + ss*fit_l[:, None]*fit_d[None, :])
                variance = 0.5*(variance + variance.T)
                scale = max(np.max(np.abs(a*a*ll)),
                            np.max(np.abs(a*ls*fit_l[None, :])),
                            np.max(np.abs(ss*fit_l[:, None]*fit_l[None, :])))
                noise_floor = 64*np.finfo(float).eps*max(k, p)*scale
                if np.trace(variance) <= epsilon:
                    gain = np.zeros((p, p))
                else:
                    # A fitted scalar prior preserves H = alpha*L even when
                    # the fold cannot identify a full matrix's null directions.
                    prior = 1 + np.trace(cross)/np.trace(variance)
                    gain = prior*np.eye(p) + regularized_matrix_gain(
                        variance, cross+(1-prior)*variance, epsilon,
                        max_rank=k-1, noise_floor=noise_floor,
                    )
            else:
                variance = np.maximum(a*a*ll - 2*a*fit_l*ls_diag + fit_l**2*ss_diag, 0.0)
                cross = a*a*ld - a*(fit_d*ls_diag + fit_l*ds_diag) + fit_l*fit_d*ss_diag
                if mode == 'scalar':
                    total = variance.sum()
                    gain = np.array(1 + cross.sum()/total if total > epsilon else 0.0)
                else:
                    gain = np.zeros(p)
                    keep = variance > epsilon
                    gain[keep] = 1 + cross[keep]/variance[keep]
                    gain = gain.reshape(shape)
            if np.isfinite(gain_limit):
                if matrix:
                    u, singular, vt = np.linalg.svd(gain, full_matrices=False)
                    if singular[0] > gain_limit:
                        gain = (u*np.minimum(singular, gain_limit)) @ vt
                else:
                    gain = np.clip(gain, -gain_limit, gain_limit)
            negligible = high_variance == 0.0
            if ref_squares is not None:
                reference = np.sqrt(reference_sums[i]/k)
                negligible |= np.sqrt(high_variance) <= 64*np.finfo(float).eps*np.maximum(reference, np.finfo(float).tiny)
            if matrix:
                gain[:, negligible] = 0.0
            elif mode == 'scalar':
                if np.all(negligible):
                    gain = np.array(0.0)
            else:
                gain.reshape(-1)[negligible] = 0.0
        else:
            gain = np.zeros(gain_shape)
        gains[indices] = gain
        high_term = (fold_wh[indices, None]-held_h)*s[indices]
        if m:
            low_term = (fold_wl[indices, None]-held_l)*s[indices]
            # The paired LF term and LF-only mean use exactly the same
            # outer-fold LF baseline. No baseline is fitted on extra samples.
            extra_mean = extra_centered_mean + ((extra_reference-fold_rl)-held_l)*extra_score_mean
            delta = extra_mean-low_term
            projection = delta @ gain if matrix else delta*gain.reshape(-1)
            corrected[indices] = high_term + ratio*projection
        else:
            corrected[indices] = high_term
        if needs_fit:
            for j in indices:
                suffix.add({key: value[j] for key, value in basis.items()})
    result = (corrected.mean(axis=0).reshape(shape), np.asarray(gains.mean(axis=0)), gains)
    if return_baselines:
        return result + (dict(high=outer_high_baselines.reshape((n,)+shape),
                              low=outer_low_baselines.reshape((n,)+shape)),)
    return result


def history_weights(data, scale, entropy_strategy='analytic'):
    if data is None:
        raise ValueError('History gain estimation requires the accepted HF training window.')
    high = scale*np.asarray(data['high_log_joint'])
    low = scale*np.asarray(data['low_log_joint'])
    if entropy_strategy == 'joint':
        from romtools.workflows.inverse.vi.drivers import _compute_variational_log_densities
        logq = _compute_variational_log_densities(
            data['optimizer_samples'], data['mean'], data['std'], data['correlation_cholesky'],
        )
        high = high-scale*logq
        low = low-scale*logq
    return high, low


def sample_ids(directory, count):
    return [f'{directory}::{i}' for i in range(count)]


def append_training_batch(batches, batch, limit):
    """Trim accepted samples to the same count limit used for ROM training."""
    if limit < 1:
        raise ValueError('The history training-window limit must be positive.')
    result = list(batches)+[batch]
    excess = sum(len(item['parameter_samples']) for item in result)-limit
    row_fields = ('parameter_samples', 'optimizer_samples', 'high_log_joint',
                  'log_priors', 'log_transform_jacobian', 'sample_ids')
    while excess > 0:
        first = result[0]
        count = len(first['parameter_samples'])
        if count <= excess:
            result.pop(0)
            excess -= count
        else:
            trimmed = dict(first)
            for key in row_fields:
                values = first[key][excess:]
                trimmed[key] = list(values) if key == 'sample_ids' else values.copy()
            result[0] = trimmed
            excess = 0
    return result


def importance_weights(samples, target_mean, target_std, target_correlation, batches):
    """Balance-mixture MIS weights; normalize in log space without clipping."""
    counts = np.array([len(batch['parameter_samples']) for batch in batches], dtype=float)
    component_logs = [
        np.log(count/counts.sum())+_compute_variational_log_densities(
            samples, batch['mean'], batch['std'], batch['correlation_cholesky'],
        ) for count, batch in zip(counts, batches)
    ]
    mixture = logsumexp(np.stack(component_logs), axis=0)
    target = _compute_variational_log_densities(samples, target_mean, target_std, target_correlation)
    log_weights = target-mixture
    if not np.any(np.isfinite(log_weights)):
        return np.zeros(len(samples)), log_weights, 0.0
    if np.any(np.isnan(log_weights)) or np.any(np.isposinf(log_weights)):
        raise ValueError('History importance densities must produce valid finite weight ratios.')
    weights = np.exp(log_weights-logsumexp(log_weights))
    return weights, log_weights, float(1/np.sum(weights**2))


def fit_history_gain(high_weights, low_weights, scores, baseline_method, mode, sample_weights=None):
    """Weighted covariance fit; probabilities enter moments once, not twice."""
    mode = _normalize_mfmc_control_variate_mode(mode)
    baseline_method = _normalize_baseline_method(baseline_method)
    n = len(high_weights)
    shape = scores.shape[1:]
    p = int(np.prod(shape))
    gain_shape = () if mode == 'scalar' else ((p, p) if mode == 'matrix' else shape)
    weights = np.ones(n)/n if sample_weights is None else np.asarray(sample_weights, dtype=float)
    if weights.shape != (n,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError('History sample weights must be finite nonnegative weights, one per sample.')
    mass = weights.sum()
    if not np.isfinite(mass):
        raise ValueError('History sample weights must have finite total mass.')
    if n < 2 or mass == 0:
        return np.zeros(gain_shape)
    weights = weights/mass
    if 1/np.sum(weights**2) < 2-1e-12:
        return np.zeros(gain_shape)
    flat = np.asarray(scores).reshape(n, p)
    high, low = np.asarray(high_weights), np.asarray(low_weights)
    if baseline_method == 'none':
        h, l = high[:, None]*flat, low[:, None]*flat
    else:
        rh, rl = high[0], low[0]
        high, low = high-rh, low-rl
        if baseline_method == 'loo':
            denominator = _exclusive_sums(weights)[:, None]
            hb = np.divide(_exclusive_sums(weights*high)[:, None], denominator,
                           out=np.full((n, 1), -rh), where=denominator > 1e-12)
            lb = np.divide(_exclusive_sums(weights*low)[:, None], denominator,
                           out=np.full((n, 1), -rl), where=denominator > 1e-12)
        else:
            denominator = np.sum(weights[:, None]*flat**2, axis=0)
            hb = np.divide(np.sum(weights[:, None]*high[:, None]*flat**2, axis=0), denominator,
                           out=np.full(p, -rh), where=denominator > 1e-12)
            lb = np.divide(np.sum(weights[:, None]*low[:, None]*flat**2, axis=0), denominator,
                           out=np.full(p, -rl), where=denominator > 1e-12)
        h, l = (high[:, None]-hb)*flat, (low[:, None]-lb)*flat
    x = l-np.sum(weights[:, None]*l, axis=0)
    y = h-np.sum(weights[:, None]*h, axis=0)
    variance = np.sum(weights[:, None]*x**2, axis=0)
    if mode == 'matrix':
        root = np.sqrt(weights)[:, None]
        u, singular, vt = np.linalg.svd(root*x, full_matrices=False)
        if singular[0]**2 <= 1e-12:
            return np.zeros((p, p))
        keep = singular**2 > np.finfo(float).eps*p*singular[0]**2
        keep[min(n-1, np.count_nonzero(weights)-1):] = False
        return vt[keep].T @ ((singular[keep]/(singular[keep]**2+1e-12))[:, None]
                             * (u[:, keep].T @ (root*y)))
    cross = np.sum(weights[:, None]*x*y, axis=0)
    if mode == 'scalar':
        total = variance.sum()
        return np.array(cross.sum()/total if total > 1e-12 else 0.0)
    return np.divide(cross, variance, out=np.zeros(p), where=variance > 1e-12).reshape(shape)


def gaussian_history_kernels(data):
    """Mean/log-std gradient and Hessian kernels at the current target distribution."""
    x, mean, std = data['optimizer_samples'], data['mean'], data['std']
    z = (x-mean)/std
    chol = data['correlation_cholesky']
    if chol is None:
        inverse = np.eye(len(mean))
        inverse_z = z
    else:
        inverse = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(len(mean))))
        inverse_z = z @ inverse.T
    sm = inverse_z/std
    sl = z*inverse_z-1
    n, d = x.shape
    second_mean = np.broadcast_to(-inverse/(std[:, None]*std[None, :]), (n, d, d)).copy()
    second_log = -(z[:, :, None]*z[:, None, :]*inverse[None, :, :])
    indices = np.arange(d)
    second_log[:, indices, indices] -= z*inverse_z
    second_cross = -z[:, None, :]*inverse[None, :, :]/std[None, :, None]
    second_cross[:, indices, indices] -= sm
    return (sm, sl), (
        sm[:, :, None]*sm[:, None, :]+second_mean,
        sl[:, :, None]*sl[:, None, :]+second_log,
        sm[:, :, None]*sl[:, None, :]+second_cross,
    )
