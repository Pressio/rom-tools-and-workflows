import numpy as np
import pytest

import romtools.linalg as romtools_linalg
from romtools.linalg import DistributedSvd


#############################
### Test the API exposure ###
#############################

def test_distributed_svd_is_exposed_by_linalg_public_api():
    assert romtools_linalg.DistributedSvd is DistributedSvd


####################
### Define Tests ###
####################

def _serial_matrices():
    rng = np.random.default_rng(314159)

    tall_float64 = rng.normal(size=(13, 5))
    wide_float32 = rng.normal(size=(4, 11)).astype(np.float32)
    complex128 = (rng.normal(size=(9, 4))
                  + 1j*rng.normal(size=(9, 4)))
    complex64 = (rng.normal(size=(3, 8))
                 + 1j*rng.normal(size=(3, 8))).astype(np.complex64)

    rank_three = rng.normal(size=(10, 3)) @ rng.normal(size=(3, 7))
    rank_deficient = np.column_stack(
        (rank_three, rank_three[:, 0] - 2.0*rank_three[:, 1])
    )

    return [
        pytest.param(tall_float64, id="tall-float64"),
        pytest.param(wide_float32, id="wide-float32"),
        pytest.param(complex128, id="tall-complex128"),
        pytest.param(complex64, id="wide-complex64"),
        pytest.param(rank_deficient, id="rank-deficient"),
        pytest.param(np.empty((0, 5)), id="zero-rows"),
        pytest.param(np.empty((5, 0)), id="zero-columns"),
        pytest.param(np.empty((0, 0)), id="empty"),
    ]


@pytest.mark.parametrize("matrix", _serial_matrices())
def test_serial_distributed_svd_matches_numpy_factors(matrix):
    expected_u, expected_s, expected_vh = np.linalg.svd(
        matrix, full_matrices=False, compute_uv=True, hermitian=False
    )

    actual_u, actual_s, actual_vh = DistributedSvd()(  # comm=None
        matrix, full_matrices=False, compute_uv=True, hermitian=False
    )

    tolerance = 5e-6 if matrix.dtype in (np.float32, np.complex64) else 1e-12
    np.testing.assert_allclose(actual_u, expected_u, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(actual_s, expected_s, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(actual_vh, expected_vh, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("full_matrices", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64,
                                    np.complex128])
def test_serial_compute_uv_false_matches_numpy(full_matrices, dtype):
    rng = np.random.default_rng(271828)
    matrix = rng.normal(size=(7, 4)).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        matrix += 1j*rng.normal(size=matrix.shape).astype(dtype)

    expected = np.linalg.svd(
        matrix,
        full_matrices=full_matrices,
        compute_uv=False,
        hermitian=False,
    )
    actual = DistributedSvd()(
        matrix,
        full_matrices=full_matrices,
        compute_uv=False,
        hermitian=False,
    )

    tolerance = 5e-6 if dtype in (np.float32, np.complex64) else 1e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def test_serial_distributed_svd_preserves_input():
    rng = np.random.default_rng(161803)
    matrix = rng.normal(size=(12, 5))
    matrix_before = matrix.copy()

    DistributedSvd()(matrix, full_matrices=False)

    np.testing.assert_array_equal(matrix, matrix_before)


def test_serial_rank_deficient_reconstruction_and_orthonormality():
    rng = np.random.default_rng(141421)
    core = rng.normal(size=(15, 2)) @ rng.normal(size=(2, 6))
    matrix = np.column_stack((core, core[:, 0] + core[:, 1]))

    u_matrix, singular_values, vh_matrix = DistributedSvd()(
        matrix, full_matrices=False
    )

    reconstruction = (u_matrix*singular_values) @ vh_matrix
    np.testing.assert_allclose(reconstruction, matrix, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        u_matrix.conj().T @ u_matrix,
        np.eye(singular_values.size),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.count_nonzero(singular_values > 1e-12) == 2


###########################
### Test error handling ###
###########################

@pytest.mark.parametrize(
    "matrix",
    [
        pytest.param(np.arange(5.0), id="one-dimensional"),
        pytest.param(np.zeros((2, 3, 4)), id="three-dimensional"),
        pytest.param(np.array([["not", "numeric"]]), id="non-numeric"),
    ],
)
def test_serial_invalid_inputs_raise_value_error(matrix):
    with pytest.raises(ValueError, match="invalid distributed SVD input"):
        DistributedSvd()(matrix, full_matrices=False)


def test_serial_full_matrices_true_with_vectors_is_rejected():
    with pytest.raises(NotImplementedError, match="full_matrices=False"):
        DistributedSvd()(np.eye(4), full_matrices=True, compute_uv=True)


@pytest.mark.parametrize("compute_uv", [False, True])
def test_serial_hermitian_true_is_rejected(compute_uv):
    with pytest.raises(NotImplementedError, match="hermitian=True"):
        DistributedSvd()(
            np.eye(4),
            full_matrices=False,
            compute_uv=compute_uv,
            hermitian=True,
        )


def test_default_numpy_signature_fails_clearly_for_unsupported_full_svd():
    with pytest.raises(NotImplementedError, match="full_matrices=False"):
        DistributedSvd()(np.eye(3))
