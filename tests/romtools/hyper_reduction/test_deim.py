import numpy as np
import pytest
from romtools.hyper_reduction import deim


@pytest.mark.mpi_skip
def test_deim_approximation():
    U = np.random.normal(size=(10, 5))
    indices = np.arange(0, 5)
    np.random.shuffle(indices)
    Uhat = deim.deim_get_approximation_matrix(U, indices)
    # Check exact reconstruction for a function in our basis
    U_approx = Uhat @ U[indices, 0]
    assert np.allclose(U_approx, U[:, 0])

    # Check projections are correct
    Phi = np.random.normal(size=(10, 3))
    deimPhi = deim.deim_get_test_basis(Phi, U, indices)
    assert np.allclose(deimPhi, (Phi.transpose() @ Uhat).transpose())


@pytest.mark.mpi_skip
def test_deim_basis():
    U = np.random.normal(size=(10, 5))
    Phi = np.random.normal(size=(10, 3))

    indices = deim.deim_get_indices(U)
    # Test over sampling
    for i in range(0, 10):
        if i not in indices:
            indices = np.append(indices, i)
            break
    assert indices.size == U.shape[1]+1
    deimPhi = deim.deim_get_test_basis(Phi, U, indices)
    assert deimPhi.shape[0] == indices.size
    assert deimPhi.shape[1] == Phi.shape[1]


@pytest.mark.mpi_skip
def test_full_deim():
    U = np.random.normal(size=(5, 5))
    indices = deim.deim_get_indices(U)
    # Confirm that we get all indices
    assert indices.size == 5
    # Confirm test index
    assert indices[0] == np.argmax(np.abs(U[:, 0]))
    assert np.allclose(np.sort(indices), np.arange(0, 5))


if __name__ == "__main__":
    test_full_deim()
    test_deim_basis()
    test_deim_approximation()
