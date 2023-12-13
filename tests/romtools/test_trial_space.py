import romtools as rt
import romtools.trial_space_utils as utils
import copy
import numpy as np
import pytest

class PythonSnapshotData(rt.AbstractSnapshotData):

    def __init__(self, snapshots):
#        if isinstance(snapshots, list):
            self.snapshots = snapshots
#        else:
#            self.snapshots = [snapshots[:,i] for i in range(0,snapshots.shape[1])]

    def getMeshGids(self):
        return np.arange(0,8)

    def getSnapshotTensor(self):
        return self.snapshots


#@pytest.mark.mpi_skip
#def test_list_snapshots_to_array():
#    snapshots = np.random.normal(size=(15,7))
#    snapshot_data = PythonSnapshotData(snapshots)
#    matrix = snapshot_data.getSnapshotsAsArray()
#    assert matrix.shape[0] == 15
#    assert matrix.shape[1] == 7

def tensor_to_matrix(tensor_input):
  return tensor_input.reshape(tensor_input.shape[0]*tensor_input.shape[1],tensor_input.shape[2])

@pytest.mark.mpi_skip
def test_dictionary_trial_space():
    snapshots = np.random.normal(size=(3,8,6))
    snapshot_data = PythonSnapshotData(snapshots)
    # default test
    my_shifter = utils.NoOpShifter()
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.DictionaryTrialSpace(snapshot_data,my_shifter,my_splitter,my_orthogonalizer)
    assert(np.allclose(myTrialSpace.getBasis().flatten(),snapshots.flatten()))
    assert(np.allclose(myTrialSpace.getShiftVector(),0))
    assert(np.allclose(myTrialSpace.getDimension(),6))

    # test with a shift
    my_shifter = utils.AverageShifter()
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.DictionaryTrialSpace(snapshot_data,my_shifter,my_splitter,my_orthogonalizer)
    assert(np.allclose(myTrialSpace.getBasis().flatten(),(snapshots - np.mean(snapshots,axis=2)[:,:,None]).flatten()))
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=2)))
    assert(np.allclose(myTrialSpace.getDimension(),6))

    # test with a shift and splitting
    my_shifter = utils.AverageShifter()
    my_splitter = utils.BlockSplitter([[0],[1,2]],3)
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.DictionaryTrialSpace(snapshot_data,my_shifter,my_splitter,my_orthogonalizer)
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=2)))
    assert(np.allclose(myTrialSpace.getDimension(),12))

    # test with a shift, splitting, and orthogonalization
    my_shifter = utils.AverageShifter()
    my_splitter = utils.BlockSplitter([[0],[1,2]],3)
    my_orthogonalizer = utils.EuclideanL2Orthogonalizer()
    myTrialSpace = rt.DictionaryTrialSpace(snapshot_data,my_shifter,my_splitter,my_orthogonalizer)
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=2)))
    assert(np.allclose(myTrialSpace.getDimension(),12))
    basis = myTrialSpace.getBasis()
    basis = tensor_to_matrix(basis)
    assert(np.allclose( basis.transpose() @ basis, np.eye(12)))

@pytest.mark.mpi_skip
def test_trial_space_from_pod():
    snapshots = np.random.normal(size=(3,8,6))
    snapshot_data = PythonSnapshotData(snapshots)
    my_truncater = utils.NoOpTruncater()
    my_shifter = utils.NoOpShifter()
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.TrialSpaceFromPOD(snapshot_data,my_truncater,my_shifter,my_splitter,my_orthogonalizer)
    ## truth trial space
    snapshotMatrix = tensor_to_matrix(snapshots)
    u,s,v = np.linalg.svd(snapshotMatrix,full_matrices=False)
    basis_tensor = myTrialSpace.getBasis()
    assert(np.allclose(u.reshape(basis_tensor.shape),basis_tensor))
    assert(np.allclose(6,myTrialSpace.getDimension()))
    assert(np.allclose(0,myTrialSpace.getShiftVector()))

    # test with a shift
    my_shifter = utils.AverageShifter()
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.TrialSpaceFromPOD(snapshot_data,my_truncater,my_shifter,my_splitter,my_orthogonalizer)
    u,s,v = np.linalg.svd(snapshotMatrix - np.mean(snapshotMatrix,axis=1)[:,None],full_matrices=False)
    basis_tensor = myTrialSpace.getBasis()
    assert(np.allclose(u.reshape(basis_tensor.shape),basis_tensor))
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=2)))
    assert(np.allclose(myTrialSpace.getDimension(),6))

    # test with a shift and splitting
    my_shifter = utils.AverageShifter()
    my_splitter = utils.BlockSplitter([[0],[1,2]],3)
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.TrialSpaceFromPOD(snapshot_data,my_truncater,my_shifter,my_splitter,my_orthogonalizer)
    u,s,v = np.linalg.svd(my_splitter(snapshotMatrix - np.mean(snapshotMatrix,axis=1)[:,None]),full_matrices=False)
    basis_tensor = myTrialSpace.getBasis()
    assert(np.allclose(u.reshape(basis_tensor.shape),basis_tensor))
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=2)))
    assert(np.allclose(myTrialSpace.getDimension(),12))


    # test with a shift, splitting, and orthogonalization
    my_shifter = utils.AverageShifter()
    my_splitter = utils.BlockSplitter([[0],[1,2]],3)
    weighting = np.abs(np.random.normal(size=24))
    my_orthogonalizer = utils.EuclideanVectorWeightedL2Orthogonalizer(weighting)
    myTrialSpace = rt.TrialSpaceFromPOD(snapshot_data,my_truncater,my_shifter,my_splitter,my_orthogonalizer)
    u,s,v = np.linalg.svd(my_splitter(snapshotMatrix - np.mean(snapshotMatrix,axis=1)[:,None]),full_matrices=False)
    u = my_orthogonalizer(u)
    basis_tensor = myTrialSpace.getBasis()
    assert(np.allclose(u.reshape(basis_tensor.shape),basis_tensor))
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=2)))
    assert(np.allclose(myTrialSpace.getDimension(),12))

@pytest.mark.mpi_skip
def test_trial_space_from_scaled_pod():
    n_var = 3
    snapshots = np.random.normal(size=(3,8,6))
    snapshot_data = PythonSnapshotData(copy.deepcopy(snapshots))
    my_truncater = utils.NoOpTruncater()
    my_shifter = utils.NoOpShifter()
    my_scaler = utils.VariableScaler('max_abs')
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.TrialSpaceFromScaledPOD(snapshot_data,my_truncater,my_shifter,my_scaler,my_splitter,my_orthogonalizer)
    scaled_snapshots = my_scaler.preScaling(snapshots)
    snapshotMatrix = tensor_to_matrix(scaled_snapshots) 
    u,s,v = np.linalg.svd(snapshotMatrix,full_matrices=False)
    basis_tensor = myTrialSpace.getBasis()
    u = u.reshape(basis_tensor.shape)
    u = my_scaler.postScaling(u)
    assert np.allclose(u,basis_tensor) , print(u,myTrialSpace.getBasis())
    assert(np.allclose(6,myTrialSpace.getDimension()))
    assert(np.allclose(0,myTrialSpace.getShiftVector()))

    # test with a shift
    snapshots = np.random.normal(size=(3,8,6))
    snapshot_data = PythonSnapshotData(copy.deepcopy(snapshots))
    my_shifter = utils.AverageShifter()
    my_splitter = utils.NoOpSplitter()
    my_scaler = utils.VariableScaler('max_abs')
    my_orthogonalizer = utils.NoOpOrthogonalizer()

    myTrialSpace = rt.TrialSpaceFromScaledPOD(snapshot_data,my_truncater,my_shifter,my_scaler,my_splitter,my_orthogonalizer)
    shifted_snapshots,shift_vector = my_shifter(snapshots)
    my_scaler = utils.VariableScaler('max_abs')
    scaled_shifted_snapshots = my_scaler.preScaling(shifted_snapshots)
    snapshot_matrix = tensor_to_matrix(scaled_shifted_snapshots)
    u,s,v = np.linalg.svd(snapshot_matrix,full_matrices=False)
    basis_tensor = myTrialSpace.getBasis()
    u = u.reshape(basis_tensor.shape)
    u = my_scaler.postScaling(u)
    assert(np.allclose(basis_tensor,u))
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=2)))
    assert(np.allclose(myTrialSpace.getDimension(),6))

    # test with a shift and splitting
    snapshots = np.random.normal(size=(3,8,6))
    snapshot_data = PythonSnapshotData(copy.deepcopy(snapshots))
    my_scaler = utils.VariableScaler('max_abs')
    my_shifter = utils.AverageShifter()
    my_splitter = utils.BlockSplitter([[0],[1,2]],3)
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.TrialSpaceFromScaledPOD(snapshot_data,my_truncater,my_shifter,my_scaler,my_splitter,my_orthogonalizer)
    shifted_snapshots,shift_vector = my_shifter(snapshots)
    my_scaler = utils.VariableScaler('max_abs')
    scaled_shifted_snapshots = my_scaler.preScaling(shifted_snapshots)
    snapshot_matrix = tensor_to_matrix(scaled_shifted_snapshots)
    snapshot_matrix = my_splitter(snapshot_matrix)
    u,s,v = np.linalg.svd(snapshot_matrix,full_matrices=False)
    basis_tensor = myTrialSpace.getBasis()
    u = u.reshape(basis_tensor.shape)
    u = my_scaler.postScaling(u)
    assert(np.allclose(basis_tensor,u))
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=2)))
    assert(np.allclose(myTrialSpace.getDimension(),12))


    # test with a shift, splitting, and orthogonalization
    snapshots = np.random.normal(size=(3,8,6))
    snapshot_data = PythonSnapshotData(copy.deepcopy(snapshots))
    my_scaler = utils.VariableScaler('max_abs')
    my_shifter = utils.AverageShifter()
    my_splitter = utils.BlockSplitter([[0],[1,2]],3)
    weighting = np.abs(np.random.normal(size=24))
    my_orthogonalizer = utils.EuclideanVectorWeightedL2Orthogonalizer(weighting)
    myTrialSpace = rt.TrialSpaceFromScaledPOD(snapshot_data,my_truncater,my_shifter,my_scaler,my_splitter,my_orthogonalizer)
    shifted_snapshots,shift_vector = my_shifter(snapshots)
    my_scaler = utils.VariableScaler('max_abs')
    scaled_shifted_snapshots = my_scaler.preScaling(shifted_snapshots)
    snapshot_matrix = tensor_to_matrix(scaled_shifted_snapshots)
    snapshot_matrix = my_splitter(snapshot_matrix)
    u,s,v = np.linalg.svd(snapshot_matrix,full_matrices=False)
    ushp = u.shape
    basis_tensor = myTrialSpace.getBasis()
    u = u.reshape(basis_tensor.shape)
    u = my_scaler.postScaling(u)
    u = my_orthogonalizer(u.reshape(ushp))
    u = u.reshape(basis_tensor.shape)
    assert(np.allclose(basis_tensor,u))
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=2)))
    assert(np.allclose(myTrialSpace.getDimension(),12))


if __name__=="__main__":
    test_dictionary_trial_space()
    test_trial_space_from_pod()
    test_trial_space_from_scaled_pod()
