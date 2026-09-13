import numpy as np
import pytest
import scipy.linalg as scipy_linalg
from romtools.hyper_reduction import deim
from romtools.hyper_reduction import DEIM, QDEIM
from romtools.vector_space.utils.truncater import BasisSizeTruncater


def _flatten_tensor(tensor):
    return np.reshape(
        tensor,
        (tensor.shape[0] * tensor.shape[1], tensor.shape[2]),
        order="C",
    )


def _unique_preserve_order(indices):
    indices = np.asarray(indices, dtype=int)
    _, first = np.unique(indices, return_index=True)
    return indices[np.sort(first)]


@pytest.mark.mpi_skip
def test_deim_class_from_basis_uses_tensor_api_and_shared_points():
    function_basis = np.zeros((2, 3, 2))
    function_basis[0, 1, 0] = 1.0
    function_basis[1, 2, 1] = 1.0

    reducer = DEIM.from_basis(function_basis)

    assert reducer.function_basis.shape == (2, 3, 2)
    assert np.array_equal(reducer.sample_indices, np.array([1, 2]))
    assert reducer.sample_all_states

    function_basis_matrix = _flatten_tensor(function_basis)
    # C-order state-major layout gives rows [1, 2] for state 0 and
    # [4, 5] for state 1 at spatial points [1, 2].
    flat_sample_indices = np.array([1, 2, 4, 5])
    expected = function_basis_matrix @ np.linalg.pinv(
        function_basis_matrix[flat_sample_indices, :]
    )
    assert np.allclose(reducer.reconstruction_matrix(), expected)


@pytest.mark.mpi_skip
def test_deim_class_from_snapshots():
    function_snapshots = np.array([[
        [3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]])

    reducer = DEIM.from_snapshots(
        function_snapshots,
        truncater=BasisSizeTruncater(2),
    )

    assert reducer.function_basis.shape == (1, 4, 2)
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
    function_basis_matrix = np.array([
        [0.1, 0.2, 0.0],
        [1.2, 0.1, 0.3],
        [0.0, 1.5, 0.2],
        [0.3, 0.1, 1.7],
        [0.6, 0.4, 0.5],
    ])
    function_basis = np.reshape(
        function_basis_matrix, (1, 5, 3), order="C"
    )
    reducer = QDEIM.from_basis(function_basis)
    coefficients = np.array([0.5, -1.0, 2.0])
    function = np.einsum("ijk,k->ij", function_basis, coefficients)

    assert np.array_equal(
        reducer.sample_indices,
        deim.qdeim_get_indices(function_basis_matrix),
    )
    assert np.allclose(
        reducer.reconstruct(function[:, reducer.sample_indices]),
        function,
    )


@pytest.mark.mpi_skip
def test_qdeim_multistate_samples_all_states_at_selected_points():
    function_basis = np.zeros((2, 3, 2))
    function_basis[0, 0, 0] = 2.0
    function_basis[1, 2, 1] = 3.0
    function_basis_matrix = _flatten_tensor(function_basis)
    raw_indices = deim.qdeim_get_indices(function_basis_matrix)
    expected_points = _unique_preserve_order(raw_indices % 3)

    reducer = QDEIM.from_basis(function_basis)

    assert np.array_equal(reducer.sample_indices, expected_points)
    assert reducer.reconstruction_matrix().shape[1] == 2 * expected_points.size


@pytest.mark.mpi_skip
def test_qdeim_class_from_snapshots():
    function_snapshots = np.diag([4.0, 3.0, 2.0, 1.0])[None, :, :]

    reducer = QDEIM.from_snapshots(
        function_snapshots,
        truncater=BasisSizeTruncater(3),
    )

    assert reducer.function_basis.shape == (1, 4, 3)
    function_basis_matrix = _flatten_tensor(reducer.function_basis)
    assert np.array_equal(
        reducer.sample_indices,
        deim.qdeim_get_indices(function_basis_matrix),
    )


@pytest.mark.mpi_skip
def test_deim_class_reconstructs_tensor_values_and_projects_test_basis():
    function_basis = np.array([
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.25]],
        [[0.25, 0.5], [1.0, 1.0], [2.0, -1.0]],
    ])
    sample_indices = np.array([0, 1])
    reducer = DEIM.from_basis(
        function_basis,
        sample_indices=sample_indices,
    )

    coefficients = np.array([2.0, -3.0])
    function = np.einsum("ijk,k->ij", function_basis, coefficients)
    assert np.allclose(
        reducer.reconstruct(function[:, sample_indices]),
        function,
    )

    coefficient_snapshots = np.array([
        [2.0, -1.0, 0.5],
        [-3.0, 4.0, 1.5],
    ])
    function_snapshots = np.einsum(
        "ijk,ks->ijs", function_basis, coefficient_snapshots
    )
    assert np.allclose(
        reducer.reconstruct(function_snapshots[:, sample_indices, :]),
        function_snapshots,
    )

    test_basis = np.array([
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [[2.0, -1.0], [0.5, 0.25], [1.5, 2.0]],
    ])
    test_basis_matrix = _flatten_tensor(test_basis)
    expected_matrix = (
        test_basis_matrix.transpose() @ reducer.reconstruction_matrix()
    ).transpose()
    expected = np.reshape(
        expected_matrix,
        (2, sample_indices.size, test_basis.shape[2]),
        order="C",
    )
    projected = reducer.project_test_basis(test_basis)
    assert projected.shape == expected.shape
    assert np.allclose(projected, expected)


@pytest.mark.mpi_skip
def test_deim_class_uses_c_order_for_shared_state_expansion():
    function_basis = np.zeros((2, 3, 2))
    function_basis[0, 1, 0] = 2.0
    function_basis[1, 1, 1] = 3.0

    reducer = DEIM.from_basis(
        function_basis,
        sample_indices=np.array([1]),
    )

    function_basis_matrix = np.reshape(
        function_basis, (6, 2), order="C"
    )
    # For C-order state-major flattening, spatial point 1 maps to rows 1 and 4.
    expected = function_basis_matrix @ np.linalg.pinv(
        function_basis_matrix[[1, 4], :]
    )
    assert np.allclose(reducer.reconstruction_matrix(), expected)

    # A Fortran-style mapping would use rows 2 and 3 and is intentionally wrong.
    wrong = function_basis_matrix @ np.linalg.pinv(
        function_basis_matrix[[2, 3], :]
    )
    assert not np.allclose(reducer.reconstruction_matrix(), wrong)


@pytest.mark.mpi_skip
def test_deim_class_can_opt_out_of_shared_state_sampling():
    function_basis = np.array([
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.25]],
        [[0.25, 0.5], [1.0, 1.0], [2.0, -1.0]],
    ])
    function_basis_matrix = _flatten_tensor(function_basis)
    raw_indices = np.atleast_1d(
        deim.deim_get_indices(function_basis_matrix)
    ).astype(int)

    reducer = DEIM.from_basis(
        function_basis,
        sample_all_states=False,
    )

    assert not reducer.sample_all_states
    assert np.array_equal(reducer.sample_indices, raw_indices)

    coefficients = np.array([2.0, -3.0])
    function = np.einsum("ijk,k->ij", function_basis, coefficients)
    function_flat = np.reshape(function, (-1,), order="C")
    assert np.allclose(
        reducer.reconstruct(function_flat[raw_indices]),
        function,
    )


@pytest.mark.mpi_skip
def test_deim_class_validates_tensor_inputs():
    with pytest.raises(ValueError, match="rank-3"):
        DEIM.from_basis(np.ones((3, 2)))

    with pytest.raises(ValueError, match="linearly independent"):
        DEIM.from_basis(np.ones((1, 3, 2)))

    function_basis = np.zeros((2, 3, 2))
    function_basis[0, 1, 0] = 1.0
    function_basis[1, 1, 1] = 1.0

    with pytest.raises(TypeError, match="integers"):
        DEIM.from_basis(
            function_basis,
            sample_indices=np.array([1.0]),
        )
    with pytest.raises(ValueError, match="duplicates"):
        DEIM.from_basis(
            function_basis,
            sample_indices=np.array([1, 1]),
        )
    with pytest.raises(ValueError, match="spatial index"):
        DEIM.from_basis(
            function_basis,
            sample_indices=np.array([3]),
        )
    with pytest.raises(TypeError, match="bool"):
        DEIM.from_basis(function_basis, sample_all_states="yes")

    reducer = DEIM.from_basis(
        function_basis,
        sample_indices=np.array([1]),
    )
    with pytest.raises(ValueError, match="leading shape"):
        reducer.reconstruct(np.ones((2, 2)))


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
