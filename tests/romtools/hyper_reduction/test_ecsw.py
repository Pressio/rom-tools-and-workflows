import numpy as np
import pytest
from romtools.hyper_reduction import ecsw

@pytest.mark.mpi_skip
def test_ecsw_nnls():
    # test NNLS routine for small random matrix
    full_mesh_lhs = np.random.normal(size=(5, 10))
    full_mesh_rhs = np.sum(full_mesh_lhs, axis=1)

    nnls = ecsw.ECSWsolverNNLS()
    sample_mesh_indices, sample_mesh_weights = nnls(full_mesh_lhs, full_mesh_rhs, 1e-20)
    full_mesh_weights = np.zeros(full_mesh_lhs.shape[1])
    full_mesh_weights[sample_mesh_indices] = sample_mesh_weights

    # test weights from NNLS
    assert np.allclose(full_mesh_lhs@full_mesh_weights, full_mesh_rhs)

    # test sample mesh inices indices from NNLS
    sample_mesh_lhs = full_mesh_lhs[:, sample_mesh_indices]

    assert np.allclose(full_mesh_lhs@full_mesh_weights, sample_mesh_lhs@sample_mesh_weights)


@pytest.mark.mpi_skip
def test_ecsw_matrix():

    # test matrix construction for scalar case
    residual_snapshots = np.random.normal(size=(10, 5))
    test_basis = np.random.normal(size=(10, 3))

    full_mesh_lhs, full_mesh_rhs = ecsw._construct_linear_system(residual_snapshots, test_basis, 1, 'C')

    # Check that left-hand-side is correctly constructed
    assert np.allclose((np.sum(full_mesh_lhs, axis=1)).reshape((3, 5), order='F'), (test_basis.T)@residual_snapshots)

    # Check right-hand-side is correctly constructed
    assert np.allclose(full_mesh_rhs.reshape((3, 5), order='F'), (test_basis.T)@residual_snapshots)


@pytest.mark.mpi_skip
def test_full_ecsw():
    # test ECSW
    nnls = ecsw.ECSWsolverNNLS()
    residual_snapshots = np.random.normal(size=(10, 5))
    test_basis = np.random.normal(size=(10, 3))

    sample_mesh_indices, sample_mesh_weights = ecsw.ecsw_fixed_test_basis(nnls, residual_snapshots, test_basis, 1, 'C', 1e-2)
    full_mesh_weights = np.zeros(residual_snapshots.shape[0])
    full_mesh_weights[sample_mesh_indices] = sample_mesh_weights

    # Check that the full approximation of the residual snapshots is correct
    exact = (test_basis.T)@residual_snapshots
    approx = (test_basis.T)@(np.diag(full_mesh_weights)@residual_snapshots)

    assert np.allclose(exact, approx)

    # Check that the sample mesh indices are correct
    sample_mesh_test_basis = test_basis[sample_mesh_indices, :]
    sample_mesh_residual_snapshots = residual_snapshots[sample_mesh_indices, :]

    sample_mesh_approx = (sample_mesh_test_basis.T)@(np.diag(sample_mesh_weights)@sample_mesh_residual_snapshots)

    assert np.allclose(sample_mesh_approx, approx)


if __name__ == "__main__":
    test_full_ecsw()
    test_ecsw_nnls()
    test_ecsw_matrix()
