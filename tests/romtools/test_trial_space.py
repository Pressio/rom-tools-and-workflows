import copy
import pytest
import numpy as np
import romtools as rt
import romtools.trial_space.utils as utils



#@pytest.mark.mpi_skip
#def test_list_snapshots_to_array():
#    snapshots = np.random.normal(size=(15,7))
#    snapshot_data = PythonSnapshotData(snapshots)
#    matrix = snapshot_data.getSnapshotsAsArray()
#    assert matrix.shape[0] == 15
#    assert matrix.shape[1] == 7


def tensor_to_matrix(tensor_input):
    return tensor_input.reshape(tensor_input.shape[0]*tensor_input.shape[1],
                                tensor_input.shape[2])


@pytest.mark.mpi_skip
def test_dictionary_trial_space():
    snapshots = np.random.normal(size=(3, 8, 6))
    original_snapshots = snapshots.copy()
    # default test
    my_shifter = utils.create_noop_shifter(snapshots)
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    my_trial_space = rt.DictionaryTrialSpace(snapshots,
                                             my_shifter,
                                             my_splitter,
                                             my_orthogonalizer)
    assert np.allclose(my_trial_space.get_basis().flatten(),
                       snapshots.flatten())
    assert np.allclose(my_trial_space.get_shift_vector(), 0)
    assert np.allclose(my_trial_space.get_dimension(), 6)

    # test with a shift
    my_shifter = utils.create_average_shifter(snapshots)
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    my_trial_space = rt.DictionaryTrialSpace(snapshots,
                                             my_shifter,
                                             my_splitter,
                                             my_orthogonalizer)
    assert np.allclose(my_trial_space.get_basis().flatten(),
                      (original_snapshots - np.mean(original_snapshots, axis=2)[:, :, None]).flatten())
    assert np.allclose(my_trial_space.get_shift_vector(),
                       np.mean(original_snapshots, axis=2))
    assert np.allclose(my_trial_space.get_dimension(), 6)

    # test with a shift and splitting
    my_shifter = utils.create_average_shifter(snapshots)
    my_splitter = utils.BlockSplitter([[0], [1, 2]], 3)
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    my_trial_space = rt.DictionaryTrialSpace(snapshots,
                                             my_shifter,
                                             my_splitter,
                                             my_orthogonalizer)
    assert np.allclose(my_trial_space.get_shift_vector(),
                       np.mean(snapshots, axis=2))
    assert np.allclose(my_trial_space.get_dimension(), 12)

    # test with a shift, splitting, and orthogonalization
    my_shifter = utils.create_average_shifter(snapshots)
    my_splitter = utils.BlockSplitter([[0], [1, 2]], 3)
    my_orthogonalizer = utils.EuclideanL2Orthogonalizer()
    my_trial_space = rt.DictionaryTrialSpace(snapshots,
                                             my_shifter,
                                             my_splitter,
                                             my_orthogonalizer)
    assert np.allclose(my_trial_space.get_shift_vector(),
                       np.mean(snapshots, axis=2))
    assert np.allclose(my_trial_space.get_dimension(), 12)
    basis = my_trial_space.get_basis()
    basis = tensor_to_matrix(basis)
    assert np.allclose(basis.transpose() @ basis, np.eye(12))


@pytest.mark.mpi_skip
def test_trial_space_from_pod():
    snapshots = np.random.normal(size=(3, 8, 6))
    original_snapshots = snapshots.copy()
    my_truncater = utils.NoOpTruncater()
    my_shifter = utils.create_noop_shifter(original_snapshots)
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    my_trial_space = rt.TrialSpaceFromPOD(original_snapshots,
                                          my_shifter,
                                          my_truncater,
                                          my_splitter,
                                          my_orthogonalizer)
    # truth trial space
    snapshotMatrix = tensor_to_matrix(original_snapshots)
    u, s, v = np.linalg.svd(snapshotMatrix, full_matrices=False)
    basis_tensor = my_trial_space.get_basis()
    assert np.allclose(u.reshape(basis_tensor.shape), basis_tensor)
    assert np.allclose(6, my_trial_space.get_dimension())
    assert np.allclose(0, my_trial_space.get_shift_vector())

    # test with a shift
    my_shifter = utils.create_average_shifter(original_snapshots)
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    my_trial_space = rt.TrialSpaceFromPOD(original_snapshots,
                                          my_shifter,
                                          my_truncater,
                                          my_splitter,
                                          my_orthogonalizer)
    u, s, v = np.linalg.svd(snapshotMatrix - np.mean(snapshotMatrix, axis=1)[:, None], full_matrices=False)
    basis_tensor = my_trial_space.get_basis()
    assert np.allclose(u.reshape(basis_tensor.shape), basis_tensor)
    assert np.allclose(my_trial_space.get_shift_vector(),
                       np.mean(original_snapshots, axis=2))
    assert np.allclose(my_trial_space.get_dimension(), 6)

    # test with a shift and splitting
    my_shifter = utils.create_average_shifter(original_snapshots)
    my_splitter = utils.BlockSplitter([[0], [1, 2]], 3)
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    my_trial_space = rt.TrialSpaceFromPOD(original_snapshots,
                                          my_shifter,
                                          my_truncater,
                                          my_splitter,
                                          my_orthogonalizer)
    u, s, v = np.linalg.svd(my_splitter(snapshotMatrix - np.mean(snapshotMatrix, axis=1)[:, None]), full_matrices=False)
    basis_tensor = my_trial_space.get_basis()
    assert np.allclose(u.reshape(basis_tensor.shape), basis_tensor)
    assert np.allclose(my_trial_space.get_shift_vector(),
                       np.mean(original_snapshots, axis=2))
    assert np.allclose(my_trial_space.get_dimension(), 12)

    # test with a shift, splitting, and orthogonalization
    my_shifter = utils.create_average_shifter(original_snapshots)
    my_splitter = utils.BlockSplitter([[0], [1, 2]], 3)
    weighting = np.abs(np.random.normal(size=24))
    my_orthogonalizer = utils.EuclideanVectorWeightedL2Orthogonalizer(weighting)
    my_trial_space = rt.TrialSpaceFromPOD(original_snapshots,
                                          my_shifter,
                                          my_truncater,
                                          my_splitter,
                                          my_orthogonalizer)
    u, s, v = np.linalg.svd(my_splitter(snapshotMatrix - np.mean(snapshotMatrix, axis=1)[:, None]), full_matrices=False)
    u = my_orthogonalizer(u)
    basis_tensor = my_trial_space.get_basis()
    assert np.allclose(u.reshape(basis_tensor.shape), basis_tensor)
    assert np.allclose(my_trial_space.get_shift_vector(),
                       np.mean(original_snapshots, axis=2))
    assert np.allclose(my_trial_space.get_dimension(), 12)


@pytest.mark.mpi_skip
def test_trial_space_from_scaled_pod():
    snapshots = np.random.normal(size=(3, 8, 6))
    my_truncater = utils.NoOpTruncater()
    my_shifter = utils.create_noop_shifter(snapshots)
    my_scaler = utils.VariableScaler('max_abs')
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    my_trial_space = rt.TrialSpaceFromScaledPOD(copy.deepcopy(snapshots),
                                                my_shifter,
                                                my_truncater,
                                                my_scaler,
                                                my_splitter,
                                                my_orthogonalizer)
    scaled_snapshots = my_scaler.pre_scaling(snapshots)
    snapshotMatrix = tensor_to_matrix(scaled_snapshots)
    u, s, v = np.linalg.svd(snapshotMatrix, full_matrices=False)
    basis_tensor = my_trial_space.get_basis()
    u = u.reshape(basis_tensor.shape)
    u = my_scaler.post_scaling(u)
    assert np.allclose(u, basis_tensor), print(u, my_trial_space.get_basis())
    assert np.allclose(6, my_trial_space.get_dimension())
    assert np.allclose(0, my_trial_space.get_shift_vector())

    # test with a shift
    snapshots = np.random.normal(size=(3, 8, 6))
    original_snapshots = snapshots.copy()
    my_shifter = utils.create_average_shifter(snapshots)
    my_splitter = utils.NoOpSplitter()
    my_scaler = utils.VariableScaler('max_abs')
    my_orthogonalizer = utils.NoOpOrthogonalizer()

    my_trial_space = rt.TrialSpaceFromScaledPOD(snapshots,
                                                my_shifter,
                                                my_truncater,
                                                my_scaler,
                                                my_splitter,
                                                my_orthogonalizer)
    my_shifter.apply_shift()
    my_scaler = utils.VariableScaler('max_abs')
    scaled_shifted_snapshots = my_scaler.pre_scaling(snapshots)
    snapshot_matrix = tensor_to_matrix(scaled_shifted_snapshots)
    u, s, v = np.linalg.svd(snapshot_matrix, full_matrices=False)
    basis_tensor = my_trial_space.get_basis()
    u = u.reshape(basis_tensor.shape)
    u = my_scaler.post_scaling(u)
    assert np.allclose(basis_tensor, u)
    assert np.allclose(my_trial_space.get_shift_vector(),+
                       np.mean(original_snapshots, axis=2))
    assert np.allclose(my_trial_space.get_dimension(), 6)

    # test with a shift and splitting
    snapshots = np.random.normal(size=(3, 8, 6))
    original_snapshots = snapshots.copy()
    my_scaler = utils.VariableScaler('max_abs')
    my_shifter = utils.create_average_shifter(snapshots)
    my_splitter = utils.BlockSplitter([[0], [1, 2]], 3)
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    my_trial_space = rt.TrialSpaceFromScaledPOD(snapshots,
                                                my_shifter,
                                                my_truncater,
                                                my_scaler,
                                                my_splitter,
                                                my_orthogonalizer)
    my_shifter.apply_shift()
    my_scaler = utils.VariableScaler('max_abs')
    scaled_shifted_snapshots = my_scaler.pre_scaling(snapshots)
    snapshot_matrix = tensor_to_matrix(scaled_shifted_snapshots)
    snapshot_matrix = my_splitter(snapshot_matrix)
    u, s, v = np.linalg.svd(snapshot_matrix, full_matrices=False)
    basis_tensor = my_trial_space.get_basis()
    u = u.reshape(basis_tensor.shape)
    u = my_scaler.post_scaling(u)
    assert np.allclose(basis_tensor, u)
    assert np.allclose(my_trial_space.get_shift_vector(),
                       np.mean(original_snapshots, axis=2))
    assert np.allclose(my_trial_space.get_dimension(), 12)

    # test with a shift, splitting, and orthogonalization
    snapshots = np.random.normal(size=(3, 8, 6))
    original_snapshots = snapshots.copy()
    my_scaler = utils.VariableScaler('max_abs')
    my_shifter = utils.create_average_shifter(snapshots)
    my_splitter = utils.BlockSplitter([[0], [1, 2]], 3)
    weighting = np.abs(np.random.normal(size=24))
    my_orthogonalizer = utils.EuclideanVectorWeightedL2Orthogonalizer(weighting)
    my_trial_space = rt.TrialSpaceFromScaledPOD(snapshots,
                                                my_shifter,
                                                my_truncater,
                                                my_scaler,
                                                my_splitter,
                                                my_orthogonalizer)
    my_shifter.apply_shift()
    my_scaler = utils.VariableScaler('max_abs')
    scaled_shifted_snapshots = my_scaler.pre_scaling(snapshots)
    snapshot_matrix = tensor_to_matrix(scaled_shifted_snapshots)
    snapshot_matrix = my_splitter(snapshot_matrix)
    u, s, v = np.linalg.svd(snapshot_matrix, full_matrices=False)
    ushp = u.shape
    basis_tensor = my_trial_space.get_basis()
    u = u.reshape(basis_tensor.shape)
    u = my_scaler.post_scaling(u)
    u = my_orthogonalizer(u.reshape(ushp))
    u = u.reshape(basis_tensor.shape)
    assert np.allclose(basis_tensor, u)
    assert np.allclose(my_trial_space.get_shift_vector(),
                       np.mean(original_snapshots, axis=2))
    assert np.allclose(my_trial_space.get_dimension(), 12)


if __name__ == "__main__":
    test_dictionary_trial_space()
    test_trial_space_from_pod()
    test_trial_space_from_scaled_pod()
