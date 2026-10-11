import numpy as np
import pytest

from romtools.workflows.inverse import mf_vi_drivers, vi_drivers


def _gradient_vector(result):
    return np.concatenate([result[0], result[1]])


def test_joint_gradient_can_vanish_while_model_hessian_remains_nonzero():
    """The joint strategy applies to the gradient, not the Hessian curvature model."""
    rng = np.random.default_rng(396)
    mean = np.array([0.25, -0.4])
    std = np.array([0.8, 1.1])
    samples = rng.normal(size=(256, 2)) * std + mean

    log_q = vi_drivers._compute_variational_log_densities(samples, mean, std)
    log_joint = log_q + 2.75

    joint_gradient = vi_drivers._compute_reinforce_gradients(
        samples,
        mean,
        std,
        log_joint,
        baseline_method="loo",
        score_function_entropy_strategy="joint",
    )
    analytic_gradient = vi_drivers._compute_reinforce_gradients(
        samples,
        mean,
        std,
        log_joint,
        baseline_method="loo",
        score_function_entropy_strategy="analytic",
    )
    model_hessian = vi_drivers._compute_reinforce_hessian_full(
        samples,
        mean,
        std,
        log_joint,
        baseline_method="loo",
    )

    assert np.linalg.norm(_gradient_vector(joint_gradient)) < 1e-13
    assert np.linalg.norm(_gradient_vector(analytic_gradient)) > 1e-4
    assert np.all(np.isfinite(model_hessian))
    assert np.linalg.norm(model_hessian) > 1e-3


@pytest.mark.parametrize("mode", ["componentwise", "scalar", "matrix"])
def test_mf_joint_gradient_fixed_point_protection_does_not_remove_hessian(mode):
    """Gradient CV protection at a fixed point must not disable MF curvature."""
    rng = np.random.default_rng(397)
    mean = np.array([0.2])
    std = np.array([0.9])
    fom_samples = rng.normal(mean, std, size=(48, 1))
    rom_extra_samples = rng.normal(mean, std, size=(96, 1))

    fom_log_q = vi_drivers._compute_variational_log_densities(
        fom_samples, mean, std
    )
    rom_extra_log_q = vi_drivers._compute_variational_log_densities(
        rom_extra_samples, mean, std
    )

    fom_log_joint = fom_log_q + 1.5
    rom_base_log_joint = fom_log_q + 0.2 * fom_samples[:, 0]
    rom_extra_log_joint = rom_extra_log_q + 0.2 * rom_extra_samples[:, 0]

    gradient = mf_vi_drivers._compute_mfmc_reinforce_gradients(
        fom_samples,
        fom_samples,
        rom_extra_samples,
        mean,
        std,
        fom_log_joint,
        rom_base_log_joint,
        rom_extra_log_joint,
        baseline_method="loo",
        use_mfmc_control_variate=True,
        mfmc_control_variate_mode=mode,
        score_function_entropy_strategy="joint",
    )
    hessian = mf_vi_drivers._compute_mfmc_reinforce_hessian_full(
        fom_samples,
        fom_samples,
        rom_extra_samples,
        mean,
        std,
        fom_log_joint,
        rom_base_log_joint,
        rom_extra_log_joint,
        baseline_method="loo",
        use_mfmc_control_variate=True,
        mfmc_control_variate_mode=mode,
    )

    assert np.linalg.norm(_gradient_vector(gradient)) < 1e-13
    assert np.linalg.norm(np.asarray(gradient[4])) == 0.0
    assert np.linalg.norm(np.asarray(gradient[5])) == 0.0
    assert np.all(np.isfinite(hessian))
    assert np.linalg.norm(hessian) > 1e-3


@pytest.mark.parametrize("mode", ["componentwise", "scalar", "matrix"])
def test_mf_hessian_reduces_to_single_fidelity_without_extra_rom_samples(mode):
    """No LF samples means no MF correction, including for Newton curvature."""
    rng = np.random.default_rng(398)
    mean = np.array([0.1, -0.3])
    std = np.array([0.7, 1.2])
    samples = rng.normal(size=(64, 2)) * std + mean
    log_joint = (
        -0.3 * samples[:, 0] ** 2
        - 0.7 * samples[:, 1] ** 2
        + 0.15 * samples[:, 0]
    )

    sf_hessian = vi_drivers._compute_reinforce_hessian_full(
        samples,
        mean,
        std,
        log_joint,
        baseline_method="loo",
    )
    mf_hessian = mf_vi_drivers._compute_mfmc_reinforce_hessian_full(
        samples,
        samples,
        np.zeros((0, 2)),
        mean,
        std,
        log_joint,
        log_joint,
        np.zeros(0),
        baseline_method="loo",
        use_mfmc_control_variate=True,
        mfmc_control_variate_mode=mode,
    )

    assert np.allclose(mf_hessian, sf_hessian, rtol=1e-13, atol=1e-13)
