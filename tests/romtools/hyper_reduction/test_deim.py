
import numpy as np
import pytest
import scipy.linalg as scipy_linalg
from romtools.hyper_reduction import deim
from romtools.hyper_reduction import DEIM, QDEIM
from romtools.vector_space.utils.truncater import BasisSizeTruncater


@pytest.mark.mpi_skip
def test_deim_class_from_basis():
    function_basis = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.25],
        [0.25, 0.5],
    ])

    reducer = DEIM.from_basis(function_basis)

    assert np.array_equal(reducer.sample_indices, np.array([0, 1]))
    assert np.allclose(reducer.function_basis, function_basis)
    assert np.allclose(
        reducer.reconstruction_matrix(),
        function_basis @ np.linalg.pinv(function_basis[[0, 1], :]),
    )


@pytest.mark.mpi_skip
def test_deim_class_from_snapshots():
    function_snapshots = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ])

    reducer = DEIM.from_snapshots(
        function_snapshots,
        truncater=BasisSizeTruncater(2),
    )

    assert reducer.function_basis.shape == (4, 2)
    assert np.array_equal(reducer.sample_indices, np.array([0, 1]))


@pytest.mark.mpi_skip
def test_qdeim_get_indices_matches_pivoted_qr():
    function_basis = np.array([
        [0.1, 0.2, 0.0],
        [1.2, 0.1, 0.3],
        [0.0, 1.5, 0.2],
        [0.3, 0.1, 1.7],
        [0.6, 0.4, 0.5],
    ])
    _, _, pivots = scipy_linalg.qr(
        function_basis.transpose(), mode="economic", pivoting=True
    )

    indices = deim.qdeim_get_indices(function_basis)

    assert np.array_equal(indices, pivots[:function_basis.shape[1]])


@pytest.mark.mpi_skip
def test_qdeim_class_from_basis_reconstructs_basis_vectors():
    function_basis = np.array([
        [0.1, 0.2, 0.0],
        [1.2, 0.1, 0.3],
        [0.0, 1.5, 0.2],
        [0.3, 0.1, 1.7],
        [0.6, 0.4, 0.5],
    ])
    reducer = QDEIM.from_basis(function_basis)
    coefficients = np.array([0.5, -1.0, 2.0])
    function = function_basis @ coefficients

    assert np.array_equal(
        reducer.sample_indices,
        deim.qdeim_get_indices(function_basis),
    )
    assert np.allclose(
        reducer.reconstruct(function[reducer.sample_indices]),
        function,
    )


@pytest.mark.mpi_skip
def test_qdeim_class_from_snapshots():
    function_snapshots = np.diag([4.0, 3.0, 2.0, 1.0])

    reducer = QDEIM.from_snapshots(
        function_snapshots,
        truncater=BasisSizeTruncater(3),
    )

    assert reducer.function_basis.shape == (4, 3)
    assert np.array_equal(
        reducer.sample_indices,
        deim.qdeim_get_indices(reducer.function_basis),
    )


@pytest.mark.mpi_skip
def test_deim_class_reconstructs_basis_vectors_and_projects_test_basis():
    function_basis = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.25],
        [0.25, 0.5],
    ])
    sample_indices = np.array([0, 1, 2])
    reducer = DEIM.from_basis(function_basis, sample_indices=sample_indices)

    coefficients = np.array([2.0, -3.0])
    function = function_basis @ coefficients
    assert np.allclose(
        reducer.reconstruct(function[sample_indices]),
        function,
    )

    test_basis = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
    ])
    expected = (
        test_basis.transpose() @ reducer.reconstruction_matrix()
    ).transpose()
    assert np.allclose(reducer.project_test_basis(test_basis), expected)


@pytest.mark.mpi_skip
def test_deim_class_validates_inputs():
    with pytest.raises(ValueError, match="rank-2"):
        DEIM.from_basis(np.ones(3))

    with pytest.raises(ValueError, match="linearly independent"):
        DEIM.from_basis(np.ones((3, 2)))

    function_basis = np.eye(3, 2)
    with pytest.raises(TypeError, match="integers"):
        DEIM.from_basis(function_basis, sample_indices=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="duplicates"):
        DEIM.from_basis(function_basis, sample_indices=np.array([0, 0]))
    with pytest.raises(ValueError, match="out-of-bounds"):
        DEIM.from_basis(function_basis, sample_indices=np.array([0, 3]))
    with pytest.raises(ValueError, match="first extent"):
        DEIM.from_basis(function_basis).reconstruct(np.ones(3))


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
def test_multi_state_deim_basis():
    U = np.random.normal(size=(3,10, 5))
    Phi = np.random.normal(size=(3,10, 3))
    indices = deim.multi_state_deim_get_indices(U)
    deimPhi = deim.multi_state_deim_get_test_basis(Phi, U, indices)
    b1 = deim.deim_get_test_basis(Phi[0],U[0],indices)
    b2 = deim.deim_get_test_basis(Phi[1],U[1],indices)
    b3 = deim.deim_get_test_basis(Phi[2],U[2],indices)
    b = np.append(b1[None],np.append(b2[None],b3[None],axis=0),axis=0)
    assert(np.allclose(b,deimPhi))

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

@pytest.mark.mpi_skip
def test_multi_state_deim_samples():
    U = np.random.normal(size=(3,5, 5))
    indices_one = deim.deim_get_indices(U[0])
    indices_two = deim.deim_get_indices(U[1])
    indices_three = deim.deim_get_indices(U[2])
    ms_indices = deim.multi_state_deim_get_indices(U)
    # Confirm that we get all indices
    all_indices = np.append(indices_one,indices_two)
    all_indices = np.unique(np.append(all_indices,indices_three))
    assert(np.allclose(np.sort(ms_indices),np.sort(all_indices)))

if __name__ == "__main__":
    test_full_deim()
    test_deim_basis()
    test_deim_approximation()
    test_multi_state_deim_samples()
    test_multi_state_deim_basis()
