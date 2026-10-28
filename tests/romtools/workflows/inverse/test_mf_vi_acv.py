import numpy as np
import pytest

from romtools.workflows.inverse.vi import mf_drivers as mf
from romtools.workflows.inverse.vi.gains import loo_acv_estimator


def reference(high, low, extra, scores, extra_scores, baseline, mode, force_unit=False, folds=None):
    n, m = len(high), len(extra)
    shape = scores.shape[1:]
    p = int(np.prod(shape))
    scores = scores.reshape(n, p)
    es = extra_scores.reshape(m, p)
    terms, gains = [], []
    for i in range(n):
        held = [i] if folds is None else next(fold for fold in folds if i in fold)
        keep = ~np.isin(np.arange(n), held)
        hs, ls, ss = high[keep], low[keep], scores[keep]
        hb = lb = np.zeros(p)
        if baseline == 'optimal' and len(hs):
            denominator = np.sum(ss**2, axis=0)
            hb = np.divide(np.sum(hs[:, None]*ss**2, axis=0), denominator,
                           out=np.zeros(p), where=denominator > 1e-12)
            lb = np.divide(np.sum(ls[:, None]*ss**2, axis=0), denominator,
                           out=np.zeros(p), where=denominator > 1e-12)
            hf = (hs[:, None]-hb)*ss
            lf = (ls[:, None]-lb)*ss
        elif baseline == 'loo' and len(hs):
            hb, lb = np.full(p, hs.mean()), np.full(p, ls.mean())
            hloo = (hs.sum()-hs)/(len(hs)-1) if len(hs) > 1 else np.zeros_like(hs)
            lloo = (ls.sum()-ls)/(len(ls)-1) if len(ls) > 1 else np.zeros_like(ls)
            hf, lf = (hs-hloo)[:, None]*ss, (ls-lloo)[:, None]*ss
        else:
            hf, lf = hs[:, None]*ss, ls[:, None]*ss
        gain_shape = () if mode == 'scalar' else ((p, p) if mode == 'matrix' else (p,))
        if not m or len(hs) < 2 and not force_unit:
            gain = np.zeros(gain_shape)
        elif force_unit:
            gain = np.eye(p) if mode == 'matrix' else np.ones(gain_shape)
        else:
            x = lf-lf.mean(axis=0)
            d = hf-lf
            d -= d.mean(axis=0)
            variance = np.mean(x*x, axis=0)
            cross = np.mean(x*d, axis=0)
            if mode == 'matrix':
                u, singular, vt = np.linalg.svd(x, full_matrices=False)
                cutoff = np.finfo(float).eps*p*(singular[0]**2)
                keep = singular**2 > cutoff
                prior = 1+cross.sum()/variance.sum() if variance.sum() > 1e-12 else 0.0
                residual = d+(1-prior)*x
                gain = (prior*np.eye(p) + vt[keep].T @
                        ((singular[keep]/(singular[keep]**2 + len(hs)*1e-12))[:, None]
                         * (u[:, keep].T @ residual))) if variance.sum() > 1e-12 else np.zeros((p, p))
                gu, gs, gvt = np.linalg.svd(gain)
                gain = (gu*np.minimum(gs, 1+np.sqrt(len(hs))/2)) @ gvt
            elif mode == 'scalar':
                total = variance.sum()
                gain = np.array(1+cross.sum()/total if total > 1e-12 else 0.0)
            else:
                gain = np.zeros(p)
                keep = variance > 1e-12
                gain[keep] = 1+cross[keep]/variance[keep]
            if mode != 'matrix':
                limit = 1+np.sqrt(len(hs))/2
                gain = np.clip(gain, -limit, limit)
        h = (high[i]-hb)*scores[i]
        l = (low[i]-lb)*scores[i]
        lm = ((extra[:, None]-lb)*es).mean(axis=0) if m else np.zeros(p)
        delta = lm-l
        terms.append(h + m/(n+m)*(delta @ gain if mode == 'matrix' else delta*gain))
        gains.append(gain)
    gains = np.array(gains)
    if mode == 'componentwise':
        gains = gains.reshape((n,)+shape)
    return np.mean(terms, axis=0).reshape(shape), gains.mean(axis=0), gains


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
@pytest.mark.parametrize('n', [2, 3, 8])
@pytest.mark.parametrize('m', [0, 1, 5])
@pytest.mark.parametrize('shape', [(3,), (2, 2)])
def test_cached_loo_matches_independent_fold_reference(mode, baseline, n, m, shape):
    rng = np.random.default_rng(201)
    high, low, extra = rng.normal(size=n), rng.normal(size=n), rng.normal(size=m)
    scores, es = rng.normal(size=(n,)+shape), rng.normal(size=(m,)+shape)
    actual = loo_acv_estimator(high, low, extra, scores, es, baseline, mode)
    expected = reference(high, low, extra, scores, es, baseline, mode)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
def test_unit_gain_preserves_normalization(mode, baseline):
    rng = np.random.default_rng(33)
    high, low, extra = rng.normal(size=5), rng.normal(size=5), rng.normal(size=7)
    scores, es = rng.normal(size=(5, 2)), rng.normal(size=(7, 2))
    actual = loo_acv_estimator(high, low, extra, scores, es, baseline, mode, force_unit_gain=True)
    expected = reference(high, low, extra, scores, es, baseline, mode, True)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b)


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
def test_matrix_orientation_and_component_coupling(mode):
    rng = np.random.default_rng(72)
    # General weighted-score samples with nontrivial cross-component structure.
    scores = rng.normal(size=(12, 3))
    high = np.sin(scores[:, 0]) + scores[:, 1]
    low = np.cos(scores[:, 2]) - scores[:, 0]
    extra_scores = rng.normal(size=(9, 3))
    extra = extra_scores[:, 1] + 0.4
    actual = loo_acv_estimator(high, low, extra, scores, extra_scores, 'none', mode)
    expected = reference(high, low, extra, scores, extra_scores, 'none', mode)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-10)
    np.testing.assert_allclose(actual[2], expected[2], atol=1e-10)


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
def test_heldout_data_cannot_change_its_gain(mode, baseline):
    rng = np.random.default_rng(711)
    high, low, extra = rng.normal(size=9), rng.normal(size=9), rng.normal(size=4)
    scores, es = rng.normal(size=(9, 3)), rng.normal(size=(4, 3))
    refs = np.abs(scores) + 1.0
    before = loo_acv_estimator(high, low, extra, scores, es, baseline, mode, high_reference_terms=refs)[2]
    high[2], low[2], scores[2], refs[2] = 30.0, -10.0, np.array([2.0, -3.0, 4.0]), np.array([60.0]*3)
    after = loo_acv_estimator(high, low, extra, scores, es, baseline, mode, high_reference_terms=refs)[2]
    np.testing.assert_allclose(after[2], before[2], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
def test_constant_fold_lf_weights_have_zero_gain(mode):
    rng = np.random.default_rng(231)
    scores, es = rng.normal(size=(8, 4)), rng.normal(size=(5, 4))
    high = rng.normal(size=8)
    result = loo_acv_estimator(high, np.full(8, 10.0), np.full(5, 10.0), scores, es, 'loo', mode)
    np.testing.assert_array_equal(result[2], np.zeros_like(result[2]))


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
def test_fixed_rom_unbiasedness_by_exact_enumeration(mode):
    import itertools
    p = 0.2
    expectation = np.zeros(2)
    for values in itertools.product((0.0, 1.0), repeat=5):
        probability = np.prod([p if value else 1-p for value in values])
        main, extra = np.array(values[:3]), np.array(values[3:])
        scores = np.tile([1.0, 2.0], (3, 1))
        es = np.tile([1.0, 2.0], (2, 1))
        estimate, _, _ = loo_acv_estimator(main, main, extra, scores, es, 'none', mode,
                                          high_reference_terms=np.ones_like(scores))
        expectation += probability*estimate
    np.testing.assert_allclose(expectation, p*np.array([1.0, 2.0]), atol=1e-12)


@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
def test_gaussian_gradient_unbiasedness_with_fold_baselines(baseline, mode):
    rng = np.random.default_rng(1982)
    values = []
    for _ in range(500):
        main, extra = rng.normal(size=5), rng.normal(size=7)
        scores, es = main[:, None], extra[:, None]
        # q=N(0,1), Gaussian log joint tilted by x; joint-entropy weights
        # are x + constant. Exact mean gradient is 1.
        estimate, _, _ = loo_acv_estimator(main+0.7, 0.5*main+0.3, 0.5*extra+0.3,
                                          scores, es, baseline, mode)
        values.append(estimate.item())
    values = np.asarray(values)
    error = values.std(ddof=1)/np.sqrt(len(values))
    assert abs(values.mean()-1.0) < 6*error


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
@pytest.mark.parametrize('entropy', ['analytic', 'joint'])
def test_loo_gradient_and_same_sample_hessian_pipeline(mode, baseline, entropy):
    rng = np.random.default_rng(2321)
    mean, std = np.array([0.2, -0.3]), np.array([0.8, 1.1])
    x = mean + std*rng.normal(size=(8, 2))
    z = mean + std*rng.normal(size=(5, 2))
    high = rng.normal(size=8)
    low = rng.normal(size=8)
    extra = rng.normal(size=5)
    scale = 1.3
    actual = mf._compute_mfmc_reinforce_gradients(
        x, x, z, mean, std, high, low, extra, baseline, True, mode,
        elbo_scaling_factor=scale, score_function_entropy_strategy=entropy,
        gradient_gain_estimation_strategy='loo', return_gain_samples=True,
    )
    hw, lw, ew = scale*high, scale*low, scale*extra
    if entropy == 'joint':
        hw -= scale*mf._compute_variational_log_densities(x, mean, std)
        lw -= scale*mf._compute_variational_log_densities(x, mean, std)
        ew -= scale*mf._compute_variational_log_densities(z, mean, std)
    smean, emean = (x-mean)/std**2, (z-mean)/std**2
    slog, elog = ((x-mean)/std)**2-1, ((z-mean)/std)**2-1
    expected_mean = reference(hw, lw, ew, smean, emean, baseline, mode)
    expected_log = reference(hw, lw, ew, slog, elog, baseline, mode)
    np.testing.assert_allclose(actual[0], expected_mean[0], rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(actual[1], expected_log[0] + (scale if entropy == 'analytic' else 0), rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(actual[4], expected_mean[1], rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(actual[8]['mean'], expected_mean[2], rtol=1e-8, atol=1e-9)
    assert np.all(np.isnan(actual[7]))
    from romtools.workflows.inverse.vi import drivers as vi
    kernels = np.array([vi._compute_reinforce_hessian_full(
        sample[None, :], mean, std, np.ones(1), 'none') for sample in x])
    ekernels = np.array([vi._compute_reinforce_hessian_full(
        sample[None, :], mean, std, np.ones(1), 'none') for sample in z])
    blocks = []
    for row, col in ((slice(0, 2), slice(0, 2)), (slice(2, 4), slice(2, 4)), (slice(0, 2), slice(2, 4))):
        sk, esk = kernels[:, row, col], ekernels[:, row, col]
        if baseline == 'optimal':
            hb = vi._compute_optimal_baseline(scale*high, sk.reshape(len(x), -1)).reshape(2, 2)
            ht = (scale*high[:, None, None]-hb)*sk
            lt = (scale*low[:, None, None]-hb)*sk
            et = (scale*extra[:, None, None]-hb)*esk
        else:
            def adjust(weights, scores):
                centered = weights-vi._compute_leave_one_out_baseline(weights) if baseline == 'loo' else weights
                return centered[:, None, None]*scores
            ht, lt, et = adjust(scale*high, sk), adjust(scale*low, sk), adjust(scale*extra, esk)
        blocks.append(mf._mfmc_gradient_estimator(ht, lt, et, True, mode)[0])
    expected = np.block([[blocks[0], blocks[2]], [blocks[2].T, blocks[1]]])
    actual_hessian = mf._compute_mfmc_reinforce_hessian_full(
        x, x, z, mean, std, high, low, extra, baseline, True, mode,
        elbo_scaling_factor=scale, hessian_gain_estimation_strategy='same_sample',
    )
    np.testing.assert_allclose(actual_hessian, expected, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['loo', 'optimal'])
def test_centered_gain_and_correction_are_invariant_to_large_weight_offsets(mode, baseline):
    rng = np.random.default_rng(177)
    scores, es = rng.normal(size=(6, 7)), rng.normal(size=(48, 7))
    # Binary-exact weights/offsets distinguish algorithmic cancellation from
    # precision already lost when the caller constructs its input weights.
    high, low, extra = (rng.integers(-8, 8, size=count)/4 for count in (6, 6, 48))
    original = loo_acv_estimator(high, low, extra, scores, es, baseline, mode)
    low_offset = -(2.0**38)
    shifted = loo_acv_estimator(high+2.0**40, low+low_offset, extra+low_offset,
                               scores, es, baseline, mode)
    for a, b in zip(original, shifted):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
@pytest.mark.parametrize('heldout', [0, 3])
def test_extreme_heldout_sample_cannot_change_its_gain(mode, baseline, heldout):
    rng = np.random.default_rng(711)
    high, low, extra = rng.normal(size=6), rng.normal(size=6), rng.normal(size=4)
    scores, es = rng.normal(size=(6, 3)), rng.normal(size=(4, 3))
    refs = np.abs(scores)+1
    before = loo_acv_estimator(high, low, extra, scores, es, baseline, mode,
                               high_reference_terms=refs)[2][heldout]
    high[heldout], low[heldout] = 1e12, -1e12
    scores[heldout], refs[heldout] = 1e12, 1e24
    after = loo_acv_estimator(high, low, extra, scores, es, baseline, mode,
                              high_reference_terms=refs)[2][heldout]
    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('baseline', ['none', 'loo', 'optimal'])
def test_perfect_rom_retains_unit_gains_in_underdetermined_folds(mode, baseline):
    rng = np.random.default_rng(501)
    high, extra = rng.normal(size=3), rng.normal(size=5)
    scores, es = rng.normal(size=(3, 7)), rng.normal(size=(5, 7))
    _, _, gains = loo_acv_estimator(high, high, extra, scores, es, baseline, mode)
    expected = np.eye(7) if mode == 'matrix' else np.ones(gains.shape[1:])
    np.testing.assert_allclose(gains, np.broadcast_to(expected, gains.shape), atol=1e-12)


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
def test_extreme_gain_guard_stabilizes_nearly_constant_lf_fold(mode):
    high, low = np.array([0., 1., 0.]), np.array([0., 1., 1.0001])
    scores = np.ones((3, 1))
    _, _, stable = loo_acv_estimator(high, low, np.array([2.]), scores, scores[:1], 'loo', mode)
    _, _, raw = loo_acv_estimator(high, low, np.array([2.]), scores, scores[:1], 'loo', mode,
                                gain_regularization=0)
    assert np.max(np.abs(raw)) > 1000
    assert np.max(np.abs(stable)) <= 1 + np.sqrt(2)/2 + 1e-12


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
def test_lf_only_samples_cannot_change_bounded_gain(mode):
    rng = np.random.default_rng(52)
    high, low = rng.normal(size=6), rng.normal(size=6)
    scores = rng.normal(size=(6, 3))
    before = loo_acv_estimator(high, low, np.ones(4), scores, np.ones((4, 3)), 'loo', mode)[2]
    after = loo_acv_estimator(high, low, np.full(4, 1e12), scores, np.full((4, 3), 1e12), 'loo', mode)[2]
    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('hessian', [False, True])
def test_bounded_loo_baselines_unbiased_by_exact_score_enumeration(mode, hessian):
    import itertools
    p = np.array([0.2, 0.65])
    states = np.array(list(itertools.product((0., 1.), repeat=2)))
    probabilities = np.prod(np.where(states, p, 1-p), axis=1)
    score = states-p
    if hessian:
        score = score[:, :, None]*score[:, None, :] - np.diag(p*(1-p))[None, :, :]
    high = states[:, 0] + 2*states[:, 1] + states[:, 0]*states[:, 1] + 0.7
    low = 0.6*states[:, 0] - 0.5*states[:, 1] + 0.8*states[:, 0]*states[:, 1] + 0.2
    target = np.einsum('n,n,n...->...', probabilities, high, score)
    expectation = np.zeros_like(target)
    for indices in itertools.product(range(4), repeat=5):
        main, extra = np.array(indices[:3]), np.array(indices[3:])
        estimate, _, _ = loo_acv_estimator(high[main], low[main], low[extra],
                                          score[main], score[extra], 'loo', mode)
        expectation += np.prod(probabilities[list(indices)])*estimate
    np.testing.assert_allclose(expectation, target, rtol=1e-11, atol=1e-12)


def test_extreme_gain_guard_reduces_small_fold_gaussian_gradient_variance():
    rng = np.random.default_rng(836)
    estimates = []
    for _ in range(400):
        main, extra = rng.normal(size=3), rng.normal(size=48)
        high = -0.75*main**2 + main
        low, low_extra = -0.8*main**2 + 0.9*main, -0.8*extra**2 + 0.9*extra
        values = []
        for strength in (0., 1.):
            estimate, _, _ = loo_acv_estimator(high, low, low_extra, main[:, None], extra[:, None],
                                              'loo', 'scalar', gain_regularization=strength)
            values.append(estimate.item())
        estimates.append(values)
    values = np.asarray(estimates)
    # The exact mean gradient is 1. Pair both estimators on identical batches.
    mse = np.mean((values-1)**2, axis=0)
    assert mse[1] < mse[0]/4
    standard_error = values[:, 1].std(ddof=1)/np.sqrt(len(values))
    assert abs(values[:, 1].mean()-1) < 6*standard_error


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
def test_constant_hf_weight_with_loo_baseline_does_not_add_cv_noise(mode):
    rng = np.random.default_rng(173)
    scores, extra_scores = rng.normal(size=(6, 3)), rng.normal(size=(8, 3))
    estimate, _, gains = loo_acv_estimator(np.full(6, 1e8), rng.normal(size=6), rng.normal(size=8),
                                          scores, extra_scores, 'loo', mode)
    np.testing.assert_array_equal(gains, np.zeros_like(gains))
    np.testing.assert_array_equal(estimate, np.zeros(3))


@pytest.mark.parametrize('mode', ['scalar', 'componentwise', 'matrix'])
@pytest.mark.parametrize('lf_scale', [0.5, 1., 10., 100., -1.])
def test_proportional_rom_has_correct_gain_and_shared_outer_baseline(mode, lf_scale):
    rng = np.random.default_rng(21)
    x, z = rng.normal(size=(8, 7)), rng.normal(size=(64, 7))
    high = -0.75*np.sum(x*x, axis=1)+x[:, 0]
    high_extra = -0.75*np.sum(z*z, axis=1)+z[:, 0]
    s, es = x, z
    loo, _, gains = loo_acv_estimator(high, lf_scale*high, lf_scale*high_extra,
                                     s, es, 'loo', mode)
    hbase = (high-mf._compute_leave_one_out_baseline(high))[:, None]*s
    # The extra mean for each correction uses its corresponding paired-fold baseline.
    baselines = mf._compute_leave_one_out_baseline(high)
    extra_means = ((high_extra[None, :, None]-baselines[:, None, None])*es[None, :, :]).mean(axis=1)
    expected = (hbase + 64/72*(extra_means-hbase)).mean(axis=0)
    np.testing.assert_allclose(loo, expected, rtol=1e-10, atol=1e-10)
    target_gain = np.eye(7)/lf_scale if mode == 'matrix' else np.full(gains.shape[1:], 1/lf_scale)
    np.testing.assert_allclose(gains, np.broadcast_to(target_gain, gains.shape), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('baseline', ['loo', 'optimal'])
@pytest.mark.parametrize('shape', [(3,), (2, 2)])
@pytest.mark.parametrize('heldout', [0, 5])
def test_outer_baselines_use_only_other_seven_pairs(baseline, shape, heldout):
    rng = np.random.default_rng(271)
    high, low, extra = rng.normal(size=8), rng.normal(size=8), rng.normal(size=11)
    scores, es = rng.normal(size=(8,)+shape), rng.normal(size=(11,)+shape)
    _, _, gains, baselines = loo_acv_estimator(high, low, extra, scores, es, baseline, 'scalar',
                                              return_baselines=True)
    keep = np.arange(8) != heldout
    if baseline == 'loo':
        expected_high = np.full(shape, high[keep].mean())
        expected_low = np.full(shape, low[keep].mean())
    else:
        expected_high = np.sum(high[keep].reshape((7,)+(1,)*len(shape))*scores[keep]**2, axis=0)/np.sum(scores[keep]**2, axis=0)
        expected_low = np.sum(low[keep].reshape((7,)+(1,)*len(shape))*scores[keep]**2, axis=0)/np.sum(scores[keep]**2, axis=0)
    np.testing.assert_allclose(baselines['high'][heldout], expected_high, atol=1e-14)
    np.testing.assert_allclose(baselines['low'][heldout], expected_low, atol=1e-14)
    high[heldout], low[heldout], scores[heldout] = 1e12, -1e12, 1e12
    _, _, changed_gains, changed_baselines = loo_acv_estimator(
        high, low, extra*100, scores, es*100, baseline, 'scalar', return_baselines=True,
    )
    np.testing.assert_array_equal(gains[heldout], changed_gains[heldout])
    for key in ('high', 'low'):
        np.testing.assert_array_equal(baselines[key][heldout], changed_baselines[key][heldout])


@pytest.mark.parametrize('shape', [(3,), (2, 2)])
@pytest.mark.parametrize('mode', ['scalar', 'componentwise'])
def test_gain_fitting_uses_nested_six_sample_baselines(shape, mode):
    rng = np.random.default_rng(444)
    high, low, extra = rng.normal(size=8), rng.normal(size=8), rng.normal(size=9)
    scores, es = rng.normal(size=(8,)+shape), rng.normal(size=(9,)+shape)
    gains = loo_acv_estimator(high, low, extra, scores, es, 'loo', mode, gain_regularization=0)[2]
    for i in range(8):
        fit_high, fit_low = [], []
        fitting = np.arange(8)[np.arange(8) != i]
        assert len(fitting) == 7
        for j in fitting:
            inner = fitting[fitting != j]
            assert len(inner) == 6
            hb, lb = high[inner].mean(), low[inner].mean()
            fit_high.append((high[j]-hb)*scores[j])
            fit_low.append((low[j]-lb)*scores[j])
        expected = mf._compute_mfmc_alpha(np.array(fit_high), np.array(fit_low), mode=mode)
        np.testing.assert_allclose(gains[i], expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize('baseline', ['loo', 'optimal'])
@pytest.mark.parametrize('shape', [(3,), (2, 2)])
@pytest.mark.parametrize('m', [1, 11])
def test_paired_and_extra_correction_use_identical_outer_lf_baseline(baseline, shape, m):
    rng = np.random.default_rng(321)
    high, low, extra = rng.normal(size=8), rng.normal(size=8), 20+rng.normal(size=m)
    scores, es = rng.normal(size=(8,)+shape), rng.normal(size=(m,)+shape)
    estimate, _, _, baselines = loo_acv_estimator(high, low, extra, scores, es, baseline, 'scalar',
                                                 force_unit_gain=True, return_baselines=True)
    expected = []
    for i in range(8):
        high_term = (high[i]-baselines['high'][i])*scores[i]
        low_term = (low[i]-baselines['low'][i])*scores[i]
        extra_mean = ((extra.reshape((m,)+(1,)*len(shape))-baselines['low'][i])*es).mean(axis=0)
        expected.append(high_term+m/(8+m)*(extra_mean-low_term))
    np.testing.assert_allclose(estimate, np.mean(expected, axis=0), rtol=1e-12, atol=1e-12)


def test_optimal_fold_baseline_zero_denominator_falls_back_to_zero():
    scores = np.array([[2., 1.], [0., 2.], [0., 3.]])
    _, _, _, baselines = loo_acv_estimator(np.array([7., 8., 9.]), np.array([-1., -2., -3.]),
                                          np.ones(2), scores, np.ones((2, 2)), 'optimal', 'scalar',
                                          return_baselines=True)
    assert baselines['high'][0, 0] == 0
    assert baselines['low'][0, 0] == 0
