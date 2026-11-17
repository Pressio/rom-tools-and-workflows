"""Regression tests for the Donoghue--Yano multifidelity EKI covariance."""

import numpy as np
import pytest

from romtools.workflows.inverse.mf_eki_drivers import compute_mf_eki_update


def _sample_cross_covariance(a, b):
    """Reference sample cross-covariance; columns are ensemble members."""
    assert a.shape[1] == b.shape[1]
    a_centered = a - np.mean(a, axis=1, keepdims=True)
    b_centered = b - np.mean(b, axis=1, keepdims=True)
    return a_centered @ b_centered.T / (a.shape[1] - 1)


def _make_results(parameters, qois, observations, model_at_mean):
    return {
        "qois": qois,
        "mean-qoi": model_at_mean(parameters.mean(axis=0)),
        "errors": observations[:, None] - qois,
    }


@pytest.mark.mpi_skip
def test_mf_eki_gain_uses_member_counts_empirical_means_and_half_noise():
    # Different ensemble sizes (5 and 7) and parameter dimension (2) expose
    # the old shape[1] error. Nonlinear QoIs distinguish E[G(theta)] from
    # G(E[theta]), which is also evaluated and stored in mean-qoi.
    parameters_fom = np.array([
        [-1.0, 0.2],
        [0.0, 0.9],
        [0.5, -0.5],
        [1.2, 1.0],
        [2.1, -0.2],
    ])
    parameters_rom = np.array([
        [-1.2, 0.3],
        [-0.5, 1.1],
        [0.1, -0.9],
        [0.6, 0.8],
        [1.1, -0.4],
        [1.8, 0.7],
        [2.5, -0.1],
    ])

    def fom(p):
        u, v = np.asarray(p).T
        return np.vstack((u**2 + 0.2 * v, np.exp(0.3 * v) - 0.1 * u))

    def rom(p):
        u, v = np.asarray(p).T
        return np.vstack((0.9 * u**2 + 0.1 * v, np.exp(0.28 * v) + 0.05 * u))

    def fom_at_mean(p):
        return fom(np.atleast_2d(p))[:, 0]

    def rom_at_mean(p):
        return rom(np.atleast_2d(p))[:, 0]

    observations = np.array([0.7, 1.2])
    gamma = np.array([[0.2, 0.03], [0.03, 0.4]])
    regularization = 0.035
    q_fom = fom(parameters_fom)
    q_rom_paired = rom(parameters_fom)
    q_rom_extra = rom(parameters_rom)

    assert not np.allclose(q_fom.mean(axis=1), fom_at_mean(parameters_fom.mean(axis=0)))

    sample_sets = [parameters_fom, parameters_rom]
    fom_results = [_make_results(parameters_fom, q_fom, observations, fom_at_mean)]
    rom_results = [
        _make_results(parameters_fom, q_rom_paired, observations, rom_at_mean),
        _make_results(parameters_rom, q_rom_extra, observations, rom_at_mean),
    ]

    observed = compute_mf_eki_update(
        sample_sets, fom_results, rom_results, gamma, regularization,
    )

    # Independent construction of the two total-variate covariances
    # (Donoghue--Yano, 2022, Section 2.4).
    c_yy = (
        _sample_cross_covariance(q_fom, q_fom)
        + 0.25 * _sample_cross_covariance(q_rom_paired, q_rom_paired)
        - 0.5 * _sample_cross_covariance(q_fom, q_rom_paired)
        - 0.5 * _sample_cross_covariance(q_rom_paired, q_fom)
        + 0.25 * _sample_cross_covariance(q_rom_extra, q_rom_extra)
    )
    p0 = parameters_fom.T
    p1 = parameters_rom.T
    c_py = (
        _sample_cross_covariance(p0, q_fom)
        + 0.25 * _sample_cross_covariance(p0, q_rom_paired)
        - 0.5 * _sample_cross_covariance(p0, q_rom_paired)
        - 0.5 * _sample_cross_covariance(p0, q_fom)
        + 0.25 * _sample_cross_covariance(p1, q_rom_extra)
    )
    lhs = c_yy + 0.5 * gamma + regularization * np.eye(2)
    expected_fom = (c_py @ np.linalg.solve(lhs, observations[:, None] - q_fom)).T
    expected_rom = (c_py @ np.linalg.solve(lhs, observations[:, None] - q_rom_extra)).T

    np.testing.assert_allclose(observed[0], expected_fom, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(observed[1], expected_rom, rtol=1e-12, atol=1e-12)


@pytest.mark.mpi_skip
def test_perfect_rom_reduces_to_empirical_single_fidelity_gain():
    # Matching paired and ancillary ensembles, with exact ROM predictions.
    # For a perfect ROM, C_yy^MF = C_yy^FOM/2 and C_py^MF = C_py^FOM/2;
    # Gamma/2 restores the single-fidelity Kalman gain (lambda = 0).
    parameters = np.array([[-1.4], [-0.2], [0.5], [1.0], [2.4]])
    qois = (parameters[:, 0]**2 + 0.3 * parameters[:, 0])[None, :]
    observations = np.array([0.7])
    gamma = np.array([[0.3]])

    def model_at_mean(p):
        return np.array([p[0]**2 + 0.3 * p[0]])

    result = _make_results(parameters, qois, observations, model_at_mean)
    observed = compute_mf_eki_update(
        [parameters.copy(), parameters.copy()],
        [result],
        [result, result],
        gamma,
        regularization_parameter=0.0,
    )

    c_yy = _sample_cross_covariance(qois, qois)
    c_py = _sample_cross_covariance(parameters.T, qois)
    expected = (c_py @ np.linalg.solve(
        c_yy + gamma, observations[:, None] - qois,
    )).T
    np.testing.assert_allclose(observed[0], expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(observed[1], expected, atol=1e-12, rtol=1e-12)


@pytest.mark.mpi_skip
def test_mf_eki_rejects_ensembles_with_fewer_than_two_members():
    one = np.array([[0.3]])
    two = np.array([[0.3], [0.8]])

    def results(samples):
        return {
            "qois": samples.T,
            "mean-qoi": samples.mean(axis=0),
            "errors": -samples.T,
        }

    with pytest.raises(ValueError, match="at least two samples"):
        compute_mf_eki_update(
            [one, two],
            [results(one)],
            [results(one), results(two)],
            np.eye(1),
            0.0,
        )
