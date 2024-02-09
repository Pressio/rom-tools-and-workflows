import copy
import pytest
import numpy as np
import romtools as rt
import romtools.vector_space.utils as utils


#@pytest.mark.mpi_skip
#def test_list_snapshots_to_array():
#    snapshots = np.random.normal(size=(15,7))
#    snapshot_data = PythonSnapshotData(snapshots)
#    matrix = snapshot_data.getSnapshotsAsArray()
#    assert matrix.shape[0] == 15
#    assert matrix.shape[1] == 7


def _tensor_to_matrix(tensor_input):
    return tensor_input.reshape(tensor_input.shape[0]*tensor_input.shape[1],
                                tensor_input.shape[2])


@pytest.mark.mpi_skip
def test_dictionary_vector_space():
    snapshots = np.random.normal(size=(3, 8, 6))
    original_snapshots = snapshots.copy()
    # default test
    my_shifter = utils.create_noop_shifter(snapshots)
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    my_vector_space = rt.DictionaryVectorSpace(snapshots,
                                             my_shifter,
                                             my_splitter,
                                             my_orthogonalizer)
    assert np.allclose(my_vector_space.get_basis().flatten(),
                       snapshots.flatten())
    assert np.allclose(my_vector_space.get_shift_vector(), 0)
    assert np.allclose(my_vector_space.get_dimension(), 6)

    # test with a shift
    my_shifter = utils.create_average_shifter(snapshots)
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    my_vector_space = rt.DictionaryVectorSpace(snapshots,
                                             my_shifter,
                                             my_splitter,
                                             my_orthogonalizer)
    assert np.allclose(my_vector_space.get_basis().flatten(),
                      (original_snapshots - np.mean(original_snapshots, axis=2)[:, :, None]).flatten())
    assert np.allclose(my_vector_space.get_shift_vector(),
                       np.mean(original_snapshots, axis=2))
    assert np.allclose(my_vector_space.get_dimension(), 6)

    # test with a shift and splitting
    my_shifter = utils.create_average_shifter(snapshots)
    my_splitter = utils.BlockSplitter([[0], [1, 2]], 3)
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    my_vector_space = rt.DictionaryVectorSpace(snapshots,
                                             my_shifter,
                                             my_splitter,
                                             my_orthogonalizer)
    assert np.allclose(my_vector_space.get_shift_vector(),
                       np.mean(snapshots, axis=2))
    assert np.allclose(my_vector_space.get_dimension(), 12)

    # test with a shift, splitting, and orthogonalization
    my_shifter = utils.create_average_shifter(snapshots)
    my_splitter = utils.BlockSplitter([[0], [1, 2]], 3)
    my_orthogonalizer = utils.EuclideanL2Orthogonalizer()
    my_vector_space = rt.DictionaryVectorSpace(snapshots,
                                             my_shifter,
                                             my_splitter,
                                             my_orthogonalizer)
    assert np.allclose(my_vector_space.get_shift_vector(),
                       np.mean(snapshots, axis=2))
    assert np.allclose(my_vector_space.get_dimension(), 12)
    basis = my_vector_space.get_basis()
    basis = _tensor_to_matrix(basis)
    assert np.allclose(basis.transpose() @ basis, np.eye(12))


@pytest.mark.mpi_skip
def test_vector_space_from_pod():
    snapshots = np.random.normal(size=(3, 8, 6))
    original_snapshots = snapshots.copy()
    my_vector_space = rt.VectorSpaceFromPOD(snapshots)
    # truth vector space
    snapshotMatrix = _tensor_to_matrix(original_snapshots)
    u, s, v = np.linalg.svd(snapshotMatrix, full_matrices=False)
    basis_tensor = my_vector_space.get_basis()
    assert np.allclose(u.reshape(basis_tensor.shape), basis_tensor)
    assert np.allclose(6, my_vector_space.get_dimension())
    assert np.allclose(0, my_vector_space.get_shift_vector())

    # test with a shift
    my_shifter = utils.create_average_shifter(snapshots)
    my_vector_space = rt.VectorSpaceFromPOD(snapshots, shifter=my_shifter)
    u, s, v = np.linalg.svd(snapshotMatrix - np.mean(snapshotMatrix, axis=1)[:, None], full_matrices=False)
    basis_tensor = my_vector_space.get_basis()
    assert np.allclose(u.reshape(basis_tensor.shape), basis_tensor) # FAILS
    assert np.allclose(my_vector_space.get_shift_vector(),
                       np.mean(original_snapshots, axis=2))
    assert np.allclose(my_vector_space.get_dimension(), 6)

    # test with a shift and splitting
    snapshots = np.random.normal(size=(3, 8, 6))
    original_snapshots = snapshots.copy()
    snapshotMatrix = _tensor_to_matrix(original_snapshots)
    my_shifter = utils.create_average_shifter(snapshots)
    my_splitter = utils.BlockSplitter([[0], [1, 2]], 3)
    my_vector_space = rt.VectorSpaceFromPOD(snapshots, shifter=my_shifter, splitter=my_splitter)
    u, s, v = np.linalg.svd(my_splitter(snapshotMatrix - np.mean(snapshotMatrix, axis=1)[:, None]), full_matrices=False)
    basis_tensor = my_vector_space.get_basis()
    assert np.allclose(u.reshape(basis_tensor.shape), basis_tensor)
    assert np.allclose(my_vector_space.get_shift_vector(),
                       np.mean(original_snapshots, axis=2))
    assert np.allclose(my_vector_space.get_dimension(), 12)

    # test with a shift, splitting, and orthogonalization
    snapshots = np.random.normal(size=(3, 8, 6))
    original_snapshots = snapshots.copy()
    snapshotMatrix = _tensor_to_matrix(original_snapshots)
    my_shifter = utils.create_average_shifter(snapshots)
    my_splitter = utils.BlockSplitter([[0], [1, 2]], 3)
    weighting = np.abs(np.random.normal(size=24))
    my_orthogonalizer = utils.EuclideanVectorWeightedL2Orthogonalizer(weighting)
    my_vector_space = rt.VectorSpaceFromPOD(snapshots, shifter=my_shifter, splitter=my_splitter, orthogonalizer=my_orthogonalizer)
    u, s, v = np.linalg.svd(my_splitter(snapshotMatrix - np.mean(snapshotMatrix, axis=1)[:, None]), full_matrices=False)
    u = my_orthogonalizer(u)
    basis_tensor = my_vector_space.get_basis()
    assert np.allclose(u.reshape(basis_tensor.shape), basis_tensor)
    assert np.allclose(my_vector_space.get_shift_vector(),
                       np.mean(original_snapshots, axis=2))
    assert np.allclose(my_vector_space.get_dimension(), 12)


@pytest.mark.mpi_skip
def test_trial_space_from_scaled_pod():
    snapshots = np.random.normal(size=(3, 8, 6))
    my_scaler = utils.VariableScaler('max_abs')
    my_vector_space = rt.VectorSpaceFromPOD(copy.deepcopy(snapshots), scaler=my_scaler)
    scaled_snapshots = my_scaler.pre_scaling(snapshots)
    snapshotMatrix = _tensor_to_matrix(scaled_snapshots)
    u, s, v = np.linalg.svd(snapshotMatrix, full_matrices=False)
    basis_tensor = my_vector_space.get_basis()
    u = u.reshape(basis_tensor.shape)
    u = my_scaler.post_scaling(u)
    assert np.allclose(u, basis_tensor), print(u, my_vector_space.get_basis())
    assert np.allclose(6, my_vector_space.get_dimension())
    assert np.allclose(0, my_vector_space.get_shift_vector())

    # test with a shift
    snapshots = np.random.normal(size=(3, 8, 6))
    shifted_snapshots = snapshots.copy()
    original_snapshots = snapshots.copy()
    my_shifter = utils.create_average_shifter(snapshots)
    my_scaler = utils.VariableScaler('max_abs')
    my_vector_space = rt.VectorSpaceFromPOD(snapshots, shifter=my_shifter, scaler=my_scaler)
    my_shifter.apply_shift(shifted_snapshots)
    scaled_shifted_snapshots = my_scaler.pre_scaling(shifted_snapshots)
    snapshot_matrix = _tensor_to_matrix(scaled_shifted_snapshots)
    u, s, v = np.linalg.svd(snapshot_matrix, full_matrices=False)
    basis_tensor = my_vector_space.get_basis()
    u = u.reshape(basis_tensor.shape)
    u = my_scaler.post_scaling(u)
    assert np.allclose(basis_tensor, u) # FAILS
    assert np.allclose(my_vector_space.get_shift_vector(),
                       np.mean(original_snapshots, axis=2))
    assert np.allclose(my_vector_space.get_dimension(), 6)

    # test with a shift and splitting
    snapshots = np.random.normal(size=(3, 8, 6))
    shifted_snapshots = snapshots.copy()
    original_snapshots = snapshots.copy()
    my_scaler = utils.VariableScaler('max_abs')
    my_shifter = utils.create_average_shifter(snapshots)
    my_splitter = utils.BlockSplitter([[0], [1, 2]], 3)
    my_vector_space = rt.VectorSpaceFromPOD(snapshots, shifter=my_shifter, splitter=my_splitter, scaler=my_scaler)
    my_shifter.apply_shift(shifted_snapshots)
    scaled_shifted_snapshots = my_scaler.pre_scaling(shifted_snapshots)
    snapshot_matrix = _tensor_to_matrix(scaled_shifted_snapshots)
    snapshot_matrix = my_splitter(snapshot_matrix)
    u, s, v = np.linalg.svd(snapshot_matrix, full_matrices=False)
    basis_tensor = my_vector_space.get_basis()
    u = u.reshape(basis_tensor.shape)
    u = my_scaler.post_scaling(u)
    assert np.allclose(basis_tensor, u)
    assert np.allclose(my_vector_space.get_shift_vector(),
                       np.mean(original_snapshots, axis=2))
    assert np.allclose(my_vector_space.get_dimension(), 12)

    # test with a shift, splitting, and orthogonalization
    snapshots = np.random.normal(size=(3, 8, 6))
    shifted_snapshots = snapshots.copy()
    original_snapshots = snapshots.copy()
    my_scaler = utils.VariableScaler('max_abs')
    my_shifter = utils.create_average_shifter(snapshots)
    my_splitter = utils.BlockSplitter([[0], [1, 2]], 3)
    weighting = np.abs(np.random.normal(size=24))
    my_orthogonalizer = utils.EuclideanVectorWeightedL2Orthogonalizer(weighting)
    my_vector_space = rt.VectorSpaceFromPOD(snapshots, shifter=my_shifter, splitter=my_splitter, scaler=my_scaler, orthogonalizer=my_orthogonalizer)
    my_shifter.apply_shift(shifted_snapshots)
    my_scaler = utils.VariableScaler('max_abs')
    scaled_shifted_snapshots = my_scaler.pre_scaling(shifted_snapshots)
    snapshot_matrix = _tensor_to_matrix(scaled_shifted_snapshots)
    snapshot_matrix = my_splitter(snapshot_matrix)
    u, s, v = np.linalg.svd(snapshot_matrix, full_matrices=False)
    ushp = u.shape
    basis_tensor = my_vector_space.get_basis()
    u = u.reshape(basis_tensor.shape)
    u = my_scaler.post_scaling(u)
    u = my_orthogonalizer(u.reshape(ushp))
    u = u.reshape(basis_tensor.shape)
    assert np.allclose(basis_tensor, u)
    assert np.allclose(my_vector_space.get_shift_vector(),
                       np.mean(original_snapshots, axis=2))
    assert np.allclose(my_vector_space.get_dimension(), 12)


if __name__ == "__main__":
    test_dictionary_vector_space()
    test_vector_space_from_pod()
    test_vector_space_from_scaled_pod()
